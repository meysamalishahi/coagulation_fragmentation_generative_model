#!/usr/bin/env python3

import random
import unittest

import numpy as np
import torch

import calorimeter_shape as shape
import gpu_model as gm


class CalorimeterShapeTests(unittest.TestCase):
    def setUp(self):
        self.features = np.asarray(
            [
                [-1.0, -0.5, 0.0],
                [-1.0, 0.5, 0.0],
                [-1.0, 0.0, 0.5],
                [1.0, -0.5, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 0.0, 0.5],
            ],
            dtype=np.float32,
        )
        self.layers = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        self.config = shape.ShapeModelConfig(
            voxel_count=6,
            layer_count=2,
            condition_dim=3,
            core_components=2,
            hidden_dim=12,
            transformer_layers=1,
            attention_heads=3,
            mixture_components=2,
            residual_hidden_dim=16,
            residual_layers=3,
        )

    def test_joint_loss_is_finite(self):
        showers = np.asarray(
            [[4.0, 2.0, 1.0, 3.0, 2.0, 1.0], [1.0, 0.0, 0.0, 5.0, 2.0, 1.0]],
            dtype=np.float32,
        )
        states, fractions = shape.dense_to_core_states(showers, self.features, 2)
        model = shape.HybridVoxelShapeModel(self.config, self.features, self.layers)
        conditions = torch.randn(2, 3)
        corruption = gm.CorruptionConfig(max_components=2, split_rate=0.0, gamma=0.0)
        core = model.sampled_core_nll(
            states, conditions, corruption, torch.Generator().manual_seed(7)
        )
        residual = model.residual_nll(states, conditions, fractions)
        loss = core["loss"] + residual["loss"]
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_vectorized_residual_matches_scalar_reference(self):
        showers = np.asarray(
            [
                [4.0, 2.0, 1.0, 3.0, 2.0, 1.0],
                [1.0, 0.0, 0.5, 5.0, 2.0, 1.0],
                [0.2, 0.4, 0.8, 0.3, 0.7, 1.1],
            ],
            dtype=np.float32,
        )
        states, fractions = shape.dense_to_core_states(showers, self.features, 2)
        model = shape.HybridVoxelShapeModel(self.config, self.features, self.layers)
        conditions = torch.randn(3, self.config.condition_dim)

        vectorized = model.residual_nll(states, conditions, fractions)
        concentration, core_masses, occupied = model.residual_parameters(states, conditions)
        targets = (fractions.to(model.dtype) - core_masses).clamp_min(0.0)
        scalar_terms = []
        scalar_cross_entropies = []
        for row in range(len(states)):
            for layer in range(self.config.layer_count):
                eligible = (model.layer_indices == layer) & ~occupied[row]
                count = int(eligible.sum())
                if count < 2:
                    continue
                values = targets[row, eligible]
                total = values.sum()
                if float(total.detach()) <= 1e-12:
                    continue
                smoothed = (values + self.config.target_smoothing) / (
                    total + self.config.target_smoothing * count
                )
                alpha = concentration[row, eligible]
                log_probability = (
                    torch.lgamma(alpha.sum())
                    - torch.lgamma(alpha).sum()
                    + ((alpha - 1.0) * smoothed.log()).sum()
                )
                scalar_terms.append(-log_probability / count)
                expected = alpha / alpha.sum()
                scalar_cross_entropies.append(
                    -(smoothed * expected.clamp_min(1e-12).log()).sum()
                )

        scalar_nll = torch.stack(scalar_terms).mean()
        scalar_cross_entropy = torch.stack(scalar_cross_entropies).mean()
        torch.testing.assert_close(
            vectorized["residual_nll"], scalar_nll, atol=2e-6, rtol=2e-6
        )
        torch.testing.assert_close(
            vectorized["residual_cross_entropy"],
            scalar_cross_entropy,
            atol=2e-6,
            rtol=2e-6,
        )

    def test_sample_respects_layer_budgets_and_geometry(self):
        model = shape.HybridVoxelShapeModel(self.config, self.features, self.layers)
        conditions = torch.randn(4, 3)
        budgets = torch.tensor([[0.4, 0.6], [0.2, 0.8], [0.9, 0.1], [0.5, 0.5]])
        corruption = gm.CorruptionConfig(max_components=2, split_rate=0.0, gamma=0.0)
        fractions, states = model.sample_fractions(
            conditions, budgets, corruption, random.Random(13)
        )
        self.assertEqual(tuple(fractions.shape), (4, 6))
        self.assertTrue(bool(torch.all(fractions >= 0.0)))
        layer_zero = fractions[:, :3].sum(dim=-1)
        layer_one = fractions[:, 3:].sum(dim=-1)
        torch.testing.assert_close(layer_zero, budgets[:, 0], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(layer_one, budgets[:, 1], atol=2e-6, rtol=2e-6)
        official = {tuple(row) for row in self.features.tolist()}
        for state in states:
            self.assertTrue(all(tuple(row.tolist()) in official for row in state.features.cpu()))

    def test_tiny_unborn_mass_keeps_core_likelihood_finite(self):
        model = shape.HybridVoxelShapeModel(self.config, self.features, self.layers)
        state = gm.WeightedSetState(
            torch.tensor([1.0, 1e-12], dtype=torch.float32),
            torch.tensor(self.features[[0, 1]]),
            0.0,
            1.0,
            (0, 1),
        )
        conditions = torch.zeros(1, self.config.condition_dim)
        corruption = gm.CorruptionConfig(max_components=2, split_rate=0.0, gamma=0.0)
        for seed in range(32):
            terms = model.sampled_core_nll(
                [state], conditions, corruption, torch.Generator().manual_seed(seed)
            )
            self.assertTrue(bool(torch.isfinite(terms["loss"])), msg=f"seed={seed}")

    def test_official_parameter_budget_is_within_one_percent(self):
        config = shape.ShapeModelConfig()
        features = np.stack(
            (
                np.linspace(-1.0, 1.0, config.voxel_count),
                np.sin(np.arange(config.voxel_count)),
                np.cos(np.arange(config.voxel_count)),
            ),
            axis=1,
        ).astype(np.float32)
        layers = np.arange(config.voxel_count) % config.layer_count
        model = shape.HybridVoxelShapeModel(config, features, layers)
        parameters = sum(value.numel() for value in model.parameters() if value.requires_grad)
        self.assertLess(abs(parameters - 25_982_005) / 25_982_005, 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
