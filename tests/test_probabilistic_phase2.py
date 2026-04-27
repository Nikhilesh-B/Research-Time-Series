"""Tests for Phase 2 probabilistic metrics (KDE log score, sharpness–calibration)."""

from __future__ import annotations

import unittest

import numpy as np

from benchmark2 import probabilistic_metrics as pm


def _scipy_available() -> bool:
    try:
        import scipy  # noqa: F401

        return True
    except ImportError:
        return False


class TestSampleFromQuantileFan(unittest.TestCase):
    def test_gaussian_fan_mean_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        levels = np.linspace(0.1, 0.9, 9)
        try:
            from scipy.stats import norm

            qcol = norm.ppf(levels)
        except ImportError:
            # Normal quantiles without scipy
            qcol = np.array(
                [-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816]
            )
        s = pm.sample_from_quantile_fan(levels, qcol, 20_000, rng)
        self.assertEqual(s.shape, (20_000,))
        self.assertTrue(np.isfinite(s).all())
        self.assertLess(abs(float(np.mean(s))), 0.05)


@unittest.skipUnless(_scipy_available(), "scipy required")
class TestKDEMeanLogScore(unittest.TestCase):
    def test_known_gaussian_quantiles_high_log_score_at_truth(self) -> None:
        from scipy.stats import norm

        levels = np.linspace(0.05, 0.95, 19)
        qcol_template = norm.ppf(levels)
        n_o, h = 3, 2
        quantiles = np.stack(
            [qcol_template[:, np.newaxis]] * n_o,
            axis=0,
        )
        quantiles = np.broadcast_to(quantiles, (n_o, levels.size, h)).copy()
        actual = np.zeros((n_o, h))
        mls, n_c = pm.kde_mean_log_score(
            actual,
            levels,
            quantiles,
            n_samples=800,
            random_state=123,
        )
        self.assertEqual(n_c, n_o * h)
        self.assertGreater(mls, -2.0)


class TestSharpnessCalibrationTable(unittest.TestCase):
    def test_build_table_shape(self) -> None:
        levels = np.array([0.1, 0.5, 0.9])
        # (n_origins=1, Q=3, horizon=1)
        quantiles = np.array([[[0.0], [1.0], [2.0]]], dtype=np.float64)
        actual = np.array([[1.0]])
        tab = pm.build_sharpness_calibration_table(
            ["M"],
            actual,
            {"M": quantiles},
            {"M": levels},
            nominal=0.8,
        )
        self.assertEqual(len(tab), 1)
        self.assertIn("Mean_PI_width", tab.columns)
        self.assertIn("Coverage_error", tab.columns)


if __name__ == "__main__":
    unittest.main()
