import unittest

import numpy as np

from zephyr.benchmarks import facemap_ridge as fr


class FacemapRidgeTests(unittest.TestCase):
    def test_motion_matrix_is_absolute_difference(self):
        frames = np.array([[[0.0]], [[2.0]], [[1.0]]], np.float32)
        np.testing.assert_allclose(fr.motion_matrix(frames)[:, 0], [0.0, 2.0, 1.0])

    def test_fit_basis_recovers_dominant_direction(self):
        rng = np.random.default_rng(0)
        direction = np.zeros(20)
        direction[3] = 1.0
        x = rng.normal(0, 0.1, (500, 20)) + rng.normal(0, 5, (500, 1)) * direction
        basis = fr.fit_basis([x[:250], x[250:]], n_components=2)
        self.assertGreater(abs(basis.components[:, 0] @ direction), 0.99)
        self.assertEqual(fr.project(x, basis).shape, (500, 2))

    def test_lagged_design_shifts_columns(self):
        x = np.arange(6.0)[:, None]
        design = fr.lagged_design(x, np.array([-1, 0, 1]))
        np.testing.assert_allclose(design[2], [3.0, 2.0, 1.0])

    def test_loso_ridge_recovers_weights(self):
        rng = np.random.default_rng(1)
        w_true = rng.normal(size=8)
        grams = {}
        for session in range(4):
            x = rng.normal(size=(400, 8))
            y = x @ w_true + rng.normal(0, 0.1, 400)
            grams[session] = fr.Gram.of(x, y)
        alpha, table = fr.select_alpha(grams)
        self.assertEqual(len(table), 13)
        w = fr.solve(fr.total(grams), alpha)
        np.testing.assert_allclose(w, w_true, atol=0.05)
