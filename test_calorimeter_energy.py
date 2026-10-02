#!/usr/bin/env python3

import unittest

import numpy as np
import torch

import calorimeter_energy as energy


class EnergyTransformTests(unittest.TestCase):
    def setUp(self):
        incidents = np.asarray([20.0, 40.0, 80.0, 160.0])
        layers = np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [2.0, 3.0, 4.0, 5.0, 6.0],
                [4.0, 5.0, 6.0, 7.0, 8.0],
                [8.0, 7.0, 6.0, 5.0, 4.0],
            ]
        )
        self.incidents = torch.tensor(incidents, dtype=torch.float32)
        self.layers = torch.tensor(layers, dtype=torch.float32)
        self.transform = energy.fit_energy_transform(incidents, layers, range(5))

    def test_energy_transform_round_trip(self):
        encoded, nonempty = self.transform.encode_targets(self.incidents, self.layers)
        total, decoded = self.transform.decode_targets(self.incidents, encoded)
        self.assertTrue(bool(nonempty.all()))
        torch.testing.assert_close(decoded, self.layers, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(total, self.layers.sum(dim=-1), rtol=2e-5, atol=2e-5)

    def test_model_matches_reference_energy_parameter_budget(self):
        model = energy.LayerEnergyModel(energy.EnergyModelConfig())
        count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        self.assertEqual(count, 1_951_789)
        self.assertLess(abs(count - 1_956_225) / 1_956_225, 0.01)

    def test_energy_loss_is_finite_and_differentiable(self):
        model = energy.LayerEnergyModel(
            energy.EnergyModelConfig(hidden_dim=32, hidden_layers=2, mixture_components=3)
        )
        targets, nonempty = self.transform.encode_targets(self.incidents, self.layers)
        terms = model.nll(self.transform.condition(self.incidents), targets, nonempty)
        self.assertTrue(all(torch.isfinite(value) for value in terms.values()))
        terms["loss"].backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients))

    def test_sampled_layers_are_nonnegative_and_conservative(self):
        model = energy.LayerEnergyModel(
            energy.EnergyModelConfig(hidden_dim=32, hidden_layers=2, mixture_components=3)
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.network[-1].bias[0] = 100.0
        total, layers, nonempty = model.sample(self.incidents, self.transform)
        self.assertTrue(bool(nonempty.all()))
        self.assertTrue(bool(torch.all(layers >= 0.0)))
        torch.testing.assert_close(layers.sum(dim=-1), total, rtol=1e-6, atol=1e-6)

    def test_empty_showers_have_finite_loss(self):
        model = energy.LayerEnergyModel(
            energy.EnergyModelConfig(hidden_dim=32, hidden_layers=2, mixture_components=3)
        )
        layers = self.layers.clone()
        layers[0] = 0.0
        targets, nonempty = self.transform.encode_targets(self.incidents, layers)
        terms = model.nll(self.transform.condition(self.incidents), targets, nonempty)
        self.assertTrue(torch.isfinite(terms["loss"]))
        self.assertFalse(bool(nonempty[0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
