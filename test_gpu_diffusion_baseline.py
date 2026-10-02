#!/usr/bin/env python3

import unittest

import torch

import gpu_diffusion_baseline as diffusion
import gpu_synthetic_benchmark as bench


class DiffusionBaselineTests(unittest.TestCase):
    def test_loss_and_sampling(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        examples = bench.generate_dataset(12, 17, bench.SyntheticConfig(max_cardinality=6))
        states = [example.state for example in examples]
        mean, std = diffusion.compute_normalization(states)
        config = diffusion.DiffusionConfig(
            max_cardinality=6,
            hidden_dim=32,
            transformer_layers=1,
            diffusion_steps=4,
        )
        model = diffusion.CardinalityDiffusion(config, mean, std).to(device)
        terms = model.loss(states[:6])
        self.assertTrue(torch.isfinite(terms["loss"]))
        terms["loss"].backward()
        samples = model.sample(8, 91)
        self.assertEqual(len(samples), 8)
        self.assertTrue(all(sample.conservation_residual() < 1e-6 for sample in samples))
        self.assertTrue(all(sample.n >= config.min_cardinality for sample in samples))
        self.assertTrue(all(sample.n <= config.max_cardinality for sample in samples))


if __name__ == "__main__":
    unittest.main(verbosity=2)
