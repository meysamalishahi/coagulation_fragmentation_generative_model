#!/usr/bin/env python3
"""Fast standard-library unit tests for cpu_validation.py."""

import math
import random
import unittest

import cpu_validation as cv


class EventMapTests(unittest.TestCase):
    def test_worked_split_example(self):
        state = cv._canonical(2.0, [(3.0, (1.0,)), (5.0, (4.0,))])
        split, children = cv.split_component(state, 1, 0.4, (2.0,))
        self.assertAlmostEqual(cv.total_mass(split), 10.0)
        self.assertEqual([round(m, 12) for m, _ in split[1]], [2.0, 3.0, 3.0])
        recovered = cv.merge_components(split, *children)
        self.assertEqual(recovered, state)

    def test_birth_delete_inverse(self):
        state = cv._canonical(1.0, [(2.0, (0.0,)), (3.0, (1.0,))])
        deleted, component = cv.delete_component(state, 0)
        self.assertEqual(cv.birth_component(deleted, component), state)

    def test_flow_inverse(self):
        state = cv._canonical(2.0, [(3.0, (1.0,)), (5.0, (4.0,))])
        transformed = cv.flow_state(state, 10.0, 0.7)
        recovered = cv.flow_state(transformed, 10.0, -0.7)
        self.assertLess(abs(recovered[0] - state[0]), 1e-12)
        self.assertLess(max(abs(a[0] - b[0]) for a, b in zip(recovered[1], state[1])), 1e-12)


class FormulaTests(unittest.TestCase):
    def test_split_jacobian(self):
        self.assertLess(cv.check_split_jacobian(random.Random(1), 20), 2e-7)

    def test_dequantization_jacobian(self):
        self.assertLess(cv.check_dequantization_jacobian(random.Random(2), 20), 1e-6)

    def test_flow_jacobian(self):
        self.assertLess(cv.check_flow_jacobian(random.Random(3), 20), 1e-6)

    def test_finite_reversal(self):
        metric, detail = cv.finite_reverse_metrics()
        self.assertLess(metric, 2e-10, detail)
        self.assertGreater(detail["positive_reverse_edges"], 0)

    def test_encoder_decoder(self):
        state = cv.encode_weights([0.2, 0.3, 0.5], [(0.0,), (1.0,), (2.0,)], 7.0, 0.15)
        decoded_by_feature = {z: w for w, (_, z) in zip(cv.decode_weights(state), state[1])}
        self.assertAlmostEqual(decoded_by_feature[(0.0,)], 0.2)
        self.assertAlmostEqual(decoded_by_feature[(1.0,)], 0.3)
        self.assertAlmostEqual(decoded_by_feature[(2.0,)], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
