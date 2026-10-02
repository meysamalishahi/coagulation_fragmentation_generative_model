#!/usr/bin/env python3
"""Unit tests for the continuous synthetic GPU benchmark."""

import unittest

import numpy as np
import torch

import gpu_synthetic_benchmark as bench
import gpu_model as gm


class SyntheticDataTests(unittest.TestCase):
    def test_hierarchical_data_conserve_mass_and_centroid(self):
        config = bench.SyntheticConfig()
        rng = np.random.default_rng(8)
        for _ in range(100):
            example = bench.generate_hierarchical_example(rng, config)
            state = example.state
            self.assertLess(state.conservation_residual(), 1e-7)
            self.assertEqual(state.n, example.target_cardinality)
            self.assertGreaterEqual(state.n, config.min_cardinality)
            self.assertLessEqual(state.n, config.max_cardinality)
            self.assertTrue(np.all(state.masses.numpy() > 0.0))
            self.assertEqual(len(example.genealogy), state.n - 1)

    def test_dataset_is_reproducible(self):
        config = bench.SyntheticConfig()
        first = bench.generate_dataset(10, 91, config)
        second = bench.generate_dataset(10, 91, config)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left.state.masses.numpy(), right.state.masses.numpy())
            np.testing.assert_array_equal(left.state.features.numpy(), right.state.features.numpy())
            self.assertEqual(left.genealogy, right.genealogy)

    def test_identical_sample_metrics_are_ideal(self):
        examples = bench.generate_dataset(30, 17, bench.SyntheticConfig())
        states = [example.state for example in examples]
        metrics, _ = bench.evaluate_samples(states, states, 8, 4)
        self.assertAlmostEqual(metrics["cardinality_tv"], 0.0)
        self.assertAlmostEqual(metrics["component_mass_w1"], 0.0)
        self.assertAlmostEqual(metrics["pair_distance_w1"], 0.0)
        self.assertAlmostEqual(metrics["summary_mmd"], 0.0)
        self.assertLess(abs(metrics["summary_energy_distance"]), 0.05)

    def test_cardinality_overflow_is_not_dropped(self):
        examples = bench.generate_dataset(10, 27, bench.SyntheticConfig(max_cardinality=8))
        reference = [example.state for example in examples]
        generated = list(reference)
        generated[0] = bench.generate_dataset(
            1, 44, bench.SyntheticConfig(min_cardinality=10, max_cardinality=10, cardinality_mean=10)
        )[0].state
        distance, rows = bench.cardinality_tv(reference, generated, 8)
        self.assertGreater(distance, 0.0)
        self.assertEqual(rows[-1]["cardinality_label"], ">8")
        self.assertAlmostEqual(rows[-1]["generated_probability"], 0.1)

    def test_latent_genealogy_is_evaluation_only(self):
        examples = bench.generate_dataset(12, 51, bench.SyntheticConfig())
        model = gm.CoagulationModel(
            gm.NeuralConfig(
                feature_dim=2,
                max_components=12,
                hidden_dim=32,
                embedding_dim=32,
                transformer_layers=1,
                attention_heads=4,
                mixture_components=2,
            )
        )
        metrics = bench.latent_genealogy_metrics(model, examples, batch_size=6)
        self.assertEqual(metrics["sets_evaluated"], len(examples))
        self.assertAlmostEqual(metrics["terminal_sibling_candidate_recall"], 1.0)
        self.assertGreater(metrics["terminal_sibling_uniform_probability"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
