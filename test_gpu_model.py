#!/usr/bin/env python3
"""Correctness and CUDA smoke tests for the complete neural model."""

import math
import random
import unittest

import torch

import gpu_model as gm


def ordered_by_lineage(state: gm.WeightedSetState):
    order = sorted(range(state.n), key=lambda index: state.lineages[index])
    indices = torch.tensor(order, device=state.masses.device)
    return state.masses[indices], state.features[indices]


class ConservativeHistoryTests(unittest.TestCase):
    def setUp(self):
        self.initial = gm.make_state(
            [0.7, 1.2, 0.9],
            [[-0.8, 0.3], [0.2, -0.4], [1.1, 0.7]],
            reservoir=0.6,
        )
        self.corruption = gm.CorruptionConfig(
            split_rate=3.0,
            gamma=0.7,
            max_components=10,
            separation_scale=0.25,
        )

    def test_forward_history_reverses_exactly(self):
        history = gm.simulate_forward_history(self.initial, self.corruption, random.Random(17))
        expected_masses, expected_features = ordered_by_lineage(self.initial)
        actual_masses, actual_features = ordered_by_lineage(history.endpoint)
        torch.testing.assert_close(actual_masses, expected_masses, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(actual_features, expected_features, rtol=1e-11, atol=1e-11)
        self.assertAlmostEqual(history.endpoint.reservoir, self.initial.reservoir, places=10)
        self.assertLess(history.endpoint.conservation_residual(), 1e-12)
        self.assertEqual(len(history.births), history.leaf_budget)

    def test_every_intermediate_state_is_conservative(self):
        history = gm.simulate_forward_history(self.initial, self.corruption, random.Random(29))
        states = [event.state for event in history.births]
        states += [event.state for event in history.merges]
        states += [interval.state_at_start for interval in history.merge_intervals]
        states.append(history.endpoint)
        self.assertTrue(states)
        self.assertLess(max(state.conservation_residual() for state in states), 1e-11)
        self.assertTrue(all(bool(torch.all(state.masses > 0)) for state in states if state.n))

    def test_float32_boundary_birth_clamps_rounding_drift(self):
        state = gm.WeightedSetState(
            torch.tensor([0.9975794753408991], dtype=torch.float32),
            torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            0.00242052465910092,
            1.0,
            (1,),
        )
        rebuilt = gm.birth_component(
            state, 0.0024206300731748343, torch.tensor([0.0, 0.0]), lineage=0
        )
        self.assertEqual(rebuilt.reservoir, 0.0)
        self.assertLess(rebuilt.conservation_residual(), 1e-7)
        with self.assertRaises(ValueError):
            gm.birth_component(state, 0.003, torch.tensor([0.0, 0.0]), lineage=0)

    def test_float32_flow_projects_only_rounding_overshoot(self):
        state = gm.WeightedSetState(
            torch.tensor([0.5, 0.5], dtype=torch.float32),
            torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
            0.0,
            1.0,
            (0, 1),
        )
        projected = gm._replace_reservoir(
            state, torch.tensor([0.7, 0.3000001], dtype=torch.float32)
        )
        self.assertEqual(projected.reservoir, 0.0)
        self.assertLess(projected.conservation_residual(), 1e-7)
        with self.assertRaises(ValueError):
            gm._replace_reservoir(state, torch.tensor([0.7, 0.31], dtype=torch.float32))


class NeuralModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(4)
        cls.config = gm.NeuralConfig(
            feature_dim=2,
            max_components=10,
            hidden_dim=48,
            embedding_dim=48,
            transformer_layers=2,
            attention_heads=4,
            mixture_components=3,
            k_max=3.0,
        )

    def new_model(self):
        return gm.CoagulationModel(self.config).to(self.device)

    def test_permutation_equivariance(self):
        model = self.new_model().eval()
        state = gm.make_state(
            [0.4, 0.8, 0.5, 0.6],
            [[-0.7, 0.2], [0.1, 0.9], [0.8, -0.4], [1.4, 0.3]],
            reservoir=0.7,
        )
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = state.permute(permutation)
        with torch.no_grad():
            first = model.merge_outputs([state], torch.tensor([0.8]), torch.tensor([4]))[0]
            second = model.merge_outputs([permuted], torch.tensor([0.8]), torch.tensor([4]))[0]
        self.assertAlmostEqual(float(first.total_rate.cpu()), float(second.total_rate.cpu()), places=5)

        def probabilities_by_lineage(current, output):
            result = {}
            for pair, probability in zip(output.pairs.cpu().tolist(), output.pair_probabilities.cpu().tolist()):
                key = tuple(sorted((current.lineages[pair[0]], current.lineages[pair[1]])))
                result[key] = probability
            return result

        first_probs = probabilities_by_lineage(state, first)
        second_probs = probabilities_by_lineage(permuted, second)
        self.assertEqual(first_probs.keys(), second_probs.keys())
        for key in first_probs:
            self.assertAlmostEqual(first_probs[key], second_probs[key], places=5)

    def test_path_loss_is_finite_and_differentiable(self):
        initial_states = [
            gm.make_state(
                [0.7, 1.2, 0.9],
                [[-0.8, 0.3], [0.2, -0.4], [1.1, 0.7]],
                reservoir=0.6,
            ),
            gm.make_state(
                [0.5, 0.8],
                [[-0.2, -0.5], [0.9, 0.4]],
                reservoir=0.7,
            ),
        ]
        corruption = gm.CorruptionConfig(split_rate=2.0, gamma=0.5, max_components=10)
        histories = [
            gm.simulate_forward_history(state, corruption, random.Random(seed))
            for state, seed in zip(initial_states, (31, 47))
        ]
        model = self.new_model().train()
        terms = model.path_loss(histories, random.Random(3))
        self.assertTrue(all(torch.isfinite(value) for value in terms.values()))
        terms["loss"].backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)

    def test_sampler_preserves_mass(self):
        model = self.new_model().eval()
        corruption = gm.CorruptionConfig(split_rate=2.0, gamma=0.5, max_components=10)
        sample = model.sample(3.5, corruption, rng=random.Random(101))
        self.assertLess(sample.conservation_residual(), 2e-6)
        self.assertTrue(math.isfinite(sample.reservoir))
        self.assertGreaterEqual(sample.reservoir, 0.0)
        self.assertTrue(bool(torch.all(sample.masses > 0)))
        self.assertLessEqual(sample.n, self.config.max_components)

    def test_batched_samplers_preserve_mass(self):
        corruption = gm.CorruptionConfig(split_rate=2.0, gamma=0.5, max_components=10)
        models = (self.new_model().eval(), gm.BirthOnlyModel(self.config).to(self.device).eval())
        for model in models:
            samples = model.sample_batch(
                [1.0, 2.0, 3.5, 0.75], corruption, rng=random.Random(202)
            )
            self.assertEqual(len(samples), 4)
            self.assertTrue(all(sample.conservation_residual() < 2e-6 for sample in samples))
            self.assertTrue(all(sample.n <= self.config.max_components for sample in samples))

    def test_birth_only_ablation_is_matched_and_differentiable(self):
        coagulation = self.new_model()
        birth_only = gm.BirthOnlyModel(self.config).to(self.device)
        self.assertEqual(
            sum(parameter.numel() for parameter in coagulation.parameters()),
            sum(parameter.numel() for parameter in birth_only.parameters()),
        )
        state = gm.make_state(
            [0.7, 1.2, 0.9],
            [[-0.8, 0.3], [0.2, -0.4], [1.1, 0.7]],
            reservoir=0.6,
        )
        corruption = gm.CorruptionConfig(split_rate=0.0, gamma=0.5, max_components=10)
        histories = [
            gm.simulate_forward_history(state, corruption, random.Random(seed))
            for seed in (31, 47)
        ]
        self.assertTrue(all(not history.merges for history in histories))
        terms = birth_only.path_loss(histories)
        terms["loss"].backward()
        self.assertTrue(torch.isfinite(terms["loss"]))
        self.assertEqual(float(terms["merge_nll"]), 0.0)
        self.assertEqual(float(terms["survival"]), 0.0)
        # The repurposed pair/rate capacity must be active in the birth loss.
        self.assertIsNotNone(next(birth_only.pair_head.parameters()).grad)
        self.assertIsNotNone(next(birth_only.rate_head.parameters()).grad)
        sample = birth_only.sample(3.5, corruption, rng=random.Random(101))
        self.assertLess(sample.conservation_residual(), 2e-6)

    def test_voxel_birth_model_uses_unique_official_cells(self):
        cells = torch.tensor(
            [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
            dtype=torch.float32,
        )
        config = gm.NeuralConfig(
            feature_dim=2,
            max_components=4,
            hidden_dim=48,
            embedding_dim=48,
            transformer_layers=2,
            attention_heads=4,
            mixture_components=3,
        )
        model = gm.VoxelBirthOnlyModel(config, cells).to(self.device)
        state = gm.WeightedSetState(
            torch.tensor([0.2], device=self.device),
            cells[[0]].to(self.device),
            0.8,
            1.0,
            (0,),
        )
        for rank in range(1, 4):
            mass, feature = model.sample_birth(state, 0.1 * rank, 4)
            index = int(torch.cdist(feature[None], cells.to(self.device)).argmin())
            occupied = {
                int(torch.cdist(value[None], cells.to(self.device)).argmin())
                for value in state.features
            }
            self.assertNotIn(index, occupied)
            state = gm.birth_component(state, mass, feature, index)
        self.assertEqual(state.n, 4)
        self.assertEqual(len(set(state.lineages)), 4)
        self.assertTrue(all(
            float(torch.cdist(feature[None], cells.to(self.device)).min()) < 1e-6
            for feature in state.features
        ))

    def test_voxel_birth_path_loss_is_differentiable(self):
        cells = torch.tensor(
            [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
            dtype=torch.float32,
        )
        config = gm.NeuralConfig(
            feature_dim=2, max_components=4, hidden_dim=48, embedding_dim=48,
            transformer_layers=2, attention_heads=4, mixture_components=3,
        )
        model = gm.VoxelBirthOnlyModel(config, cells).to(self.device)
        initial = gm.make_state(
            [0.2, 0.3, 0.4], cells[[0, 2, 3]], reservoir=0.1, lineages=[0, 2, 3],
            dtype=torch.float32,
        )
        history = gm.simulate_forward_history(
            initial.to(self.device), gm.CorruptionConfig(split_rate=0.0, max_components=4),
            random.Random(91),
        )
        terms = model.path_loss([history])
        self.assertTrue(torch.isfinite(terms["loss"]))
        terms["loss"].backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))

    def test_voxel_lookup_accepts_exact_float32_geometry_rows(self):
        angles = torch.linspace(-3.0, 3.0, 368, dtype=torch.float32)
        cells = torch.stack((
            torch.linspace(-1.0, 1.0, 368),
            0.97 * torch.cos(angles),
            0.97 * torch.sin(angles),
        ), dim=-1)
        config = gm.NeuralConfig(
            feature_dim=3, max_components=48, hidden_dim=24, embedding_dim=24,
            transformer_layers=1, attention_heads=3, mixture_components=2,
        )
        model = gm.VoxelBirthOnlyModel(config, cells).to(self.device)
        selected = torch.tensor([0, 37, 183, 367], device=self.device)
        indices = model._cell_indices(model.voxel_features[selected])
        torch.testing.assert_close(indices, selected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_execution(self):
        model = self.new_model()
        self.assertEqual(next(model.parameters()).device.type, "cuda")
        logits = model.leaf_logits(torch.tensor([1.0, 2.0], device=self.device))
        self.assertEqual(logits.device.type, "cuda")
        self.assertEqual(tuple(logits.shape), (2, self.config.max_components + 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
