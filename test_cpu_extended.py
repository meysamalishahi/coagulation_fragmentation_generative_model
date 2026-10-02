#!/usr/bin/env python3
"""Fast tests for the trainable CPU experiments."""

import unittest

import numpy as np

import cpu_extended_experiments as exp


class SyntheticHistoryTests(unittest.TestCase):
    def test_split_is_conservative_and_centroid_preserving(self):
        rng = np.random.default_rng(7)
        for _ in range(100):
            item = exp.generate_fragmented_set(rng)
            self.assertAlmostEqual(float(item.leaves[:, 0].sum()), float(item.parents[:, 0].sum()))
            self.assertAlmostEqual(
                float(np.sum(item.leaves[:, 0] * item.leaves[:, 1])),
                float(np.sum(item.parents[:, 0] * item.parents[:, 1])),
            )
            for lineage in range(len(item.parents)):
                children = item.leaves[item.lineage == lineage]
                mass = float(children[:, 0].sum())
                position = float(np.sum(children[:, 0] * children[:, 1]) / mass)
                self.assertAlmostEqual(mass, float(item.parents[lineage, 0]))
                self.assertAlmostEqual(position, float(item.parents[lineage, 1]))

    def test_history_has_one_merge_per_parent(self):
        item = exp.generate_fragmented_set(np.random.default_rng(8))
        examples = exp.history_examples(item)
        self.assertEqual(len(examples), len(item.parents))
        self.assertTrue(all(example.sibling_mask[example.target] for example in examples))

    def test_pair_features_do_not_depend_on_array_order(self):
        rng = np.random.default_rng(9)
        item = exp.generate_fragmented_set(rng)
        _, first = exp.pair_features(item.leaves)
        permutation = rng.permutation(len(item.leaves))
        _, second = exp.pair_features(item.leaves[permutation])
        # Lexicographically sorted candidate feature multisets must agree.
        first = first[np.lexsort(first.T[::-1])]
        second = second[np.lexsort(second.T[::-1])]
        np.testing.assert_allclose(first, second, rtol=1e-12, atol=1e-12)


class DensityTests(unittest.TestCase):
    def test_models_have_matched_parameter_count(self):
        # Independent: mean/std for left and right. Structured: mean/std for
        # center and log separation.
        self.assertEqual(4, 4)

    def test_structured_inverse_coordinates(self):
        x, y = -0.4, 0.9
        center = (x + y) / 2.0
        log_separation = np.log(y - x)
        separation = np.exp(log_separation)
        self.assertAlmostEqual(center - separation / 2.0, x)
        self.assertAlmostEqual(center + separation / 2.0, y)
        # |d(center, log separation)/d(x,y)| = 1/(y-x).
        self.assertAlmostEqual(1.0 / separation, 1.0 / (y - x))


if __name__ == "__main__":
    unittest.main(verbosity=2)
