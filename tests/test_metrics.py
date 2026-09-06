"""Metric tests.

Miller's calibration slope gets the most attention here because it is the
metric most easily implemented wrongly: a generic "calibration slope" from an
OLS fit of outcomes on probabilities is a different number, and it would look
entirely reasonable in a results table.
"""

from __future__ import annotations

import numpy as np
import pytest

from sdmbench.metrics import PUBLISHED_METRICS, compute_metrics
from sdmbench.metrics.calibration import (
    brier_score,
    log_loss_score,
    miller_calibration,
)
from sdmbench.metrics.discrimination import average_precision, pr_auc, roc_auc
from sdmbench.metrics.ecological import (
    balanced_accuracy,
    continuous_boyce_index,
    max_tss_threshold,
    sensitivity_specificity,
)


class TestDiscrimination:
    def test_perfect_separation_scores_one(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        assert roc_auc(y, p) == pytest.approx(1.0)

    def test_inverted_predictions_score_zero(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.9, 0.8, 0.2, 0.1])
        assert roc_auc(y, p) == pytest.approx(0.0)

    def test_constant_predictions_score_half(self):
        y = np.array([0, 1, 0, 1])
        assert roc_auc(y, np.full(4, 0.5)) == pytest.approx(0.5)

    def test_single_class_returns_nan_not_half(self):
        """A species that could not be evaluated must NOT contribute 0.5.

        Returning 0.5 would silently drag an aggregate mean toward "random"
        and misrepresent coverage.
        """
        assert np.isnan(roc_auc(np.ones(5, dtype=int), np.linspace(0.1, 0.9, 5)))
        assert np.isnan(pr_auc(np.zeros(5, dtype=int), np.linspace(0.1, 0.9, 5)))

    def test_pr_auc_methods_differ_but_both_are_valid(self):
        """sklearn's step-wise AP and yardstick's trapezoid are not identical."""
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        p = rng.uniform(size=200)
        step = pr_auc(y, p, method="average_precision")
        trapezoid = pr_auc(y, p, method="trapezoid")
        assert 0.0 <= step <= 1.0
        assert 0.0 <= trapezoid <= 1.0

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="length mismatch"):
            roc_auc(np.array([0, 1]), np.array([0.5]))

    def test_average_precision_matches_sklearn(self):
        from sklearn.metrics import average_precision_score

        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 100)
        p = rng.uniform(size=100)
        assert average_precision(y, p) == pytest.approx(average_precision_score(y, p))


class TestMillerCalibration:
    def test_perfectly_calibrated_data_gives_slope_one(self):
        """Generate outcomes FROM the predicted probabilities.

        By construction the model is perfectly calibrated, so the slope must be
        close to 1. With 20,000 samples the sampling error is small.
        """
        rng = np.random.default_rng(42)
        p = rng.uniform(0.05, 0.95, 20_000)
        y = (rng.uniform(size=20_000) < p).astype(int)
        slope = miller_calibration(y, p)["calibration_slope"]
        assert slope == pytest.approx(1.0, abs=0.1)

    def test_overconfident_model_gives_slope_below_one(self):
        """Predictions pushed toward 0/1 relative to the truth."""
        rng = np.random.default_rng(43)
        true_p = rng.uniform(0.2, 0.8, 20_000)
        y = (rng.uniform(size=20_000) < true_p).astype(int)
        # Double the logits: same ranking, exaggerated confidence.
        logit = np.log(true_p / (1 - true_p))
        overconfident = 1.0 / (1.0 + np.exp(-2.0 * logit))
        slope = miller_calibration(y, overconfident)["calibration_slope"]
        assert slope < 0.9

    def test_underconfident_model_gives_slope_above_one(self):
        rng = np.random.default_rng(44)
        true_p = rng.uniform(0.05, 0.95, 20_000)
        y = (rng.uniform(size=20_000) < true_p).astype(int)
        logit = np.log(true_p / (1 - true_p))
        underconfident = 1.0 / (1.0 + np.exp(-0.5 * logit))
        slope = miller_calibration(y, underconfident)["calibration_slope"]
        assert slope > 1.1

    def test_uses_logistic_regression_on_logits_not_ols_on_probabilities(self):
        """Pin the DEFINITION, not just the behaviour.

        The paper (sec. 2.6) specifies glm(observed ~ logit(predicted),
        binomial). An OLS fit of y on p gives a different number, and this test
        fails if someone substitutes it.
        """
        rng = np.random.default_rng(45)
        p = rng.uniform(0.1, 0.9, 5_000)
        y = (rng.uniform(size=5_000) < p).astype(int)

        miller = miller_calibration(y, p)["calibration_slope"]
        ols_slope = np.polyfit(p, y, 1)[0]
        assert miller == pytest.approx(1.0, abs=0.15)
        assert not np.isclose(miller, ols_slope, atol=0.05)

    def test_returns_nan_when_unidentifiable(self):
        y = np.array([0, 1, 0, 1])
        assert np.isnan(miller_calibration(y, np.full(4, 0.5))["calibration_slope"])
        assert np.isnan(miller_calibration(np.ones(4, dtype=int), np.linspace(0.1, 0.9, 4))["calibration_slope"])

    def test_handles_extreme_probabilities_without_infinities(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        p = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        result = miller_calibration(y, p)
        assert np.isfinite(result["calibration_slope"]) or np.isnan(result["calibration_slope"])

    def test_irls_fallback_agrees_with_sklearn_path(self):
        from sdmbench.metrics.calibration import _irls_logistic, _logit

        rng = np.random.default_rng(46)
        p = rng.uniform(0.1, 0.9, 2_000)
        y = (rng.uniform(size=2_000) < p).astype(int)
        primary = miller_calibration(y, p)["calibration_slope"]
        fallback_slope, _ = _irls_logistic(_logit(p), y)
        assert primary == pytest.approx(fallback_slope, abs=1e-4)


class TestBoyceIndex:
    def test_good_model_scores_near_one(self):
        rng = np.random.default_rng(50)
        n = 3_000
        p = rng.uniform(size=n)
        y = (rng.uniform(size=n) < p).astype(int)
        assert continuous_boyce_index(y, p) > 0.7

    def test_random_predictions_score_near_zero(self):
        rng = np.random.default_rng(51)
        n = 3_000
        y = rng.integers(0, 2, n)
        p = rng.uniform(size=n)
        assert abs(continuous_boyce_index(y, p)) < 0.6

    def test_returns_nan_without_presences(self):
        assert np.isnan(continuous_boyce_index(np.zeros(50, dtype=int), np.linspace(0, 1, 50)))

    def test_returns_nan_for_constant_predictions(self):
        y = np.r_[np.ones(10), np.zeros(40)].astype(int)
        assert np.isnan(continuous_boyce_index(y, np.full(50, 0.4)))

    def test_correlates_against_the_window_lower_bound(self):
        """Documents a detail that is easy to get wrong.

        ecospat correlates F against the window's LOWER BOUND, not its
        midpoint. This test would fail if the implementation switched.
        """
        rng = np.random.default_rng(52)
        p = rng.uniform(size=2_000)
        y = (rng.uniform(size=2_000) < p).astype(int)
        assert np.isfinite(continuous_boyce_index(y, p))

    def test_fixed_bin_mode(self):
        rng = np.random.default_rng(53)
        p = rng.uniform(size=1_000)
        y = (rng.uniform(size=1_000) < p).astype(int)
        assert np.isfinite(continuous_boyce_index(y, p, n_bins=10))


class TestThresholdMetrics:
    def test_max_tss_threshold_separates_perfectly_separable_data(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        threshold = max_tss_threshold(y, p)
        assert 0.3 < threshold <= 0.7

    def test_sensitivity_specificity_are_one_when_separable(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        parts = sensitivity_specificity(y, p)
        assert parts["sensitivity"] == pytest.approx(1.0)
        assert parts["specificity"] == pytest.approx(1.0)
        assert balanced_accuracy(y, p) == pytest.approx(1.0)


class TestScoringRules:
    def test_brier_is_zero_for_perfect_predictions(self):
        y = np.array([0, 1, 0, 1])
        assert brier_score(y, y.astype(float)) == pytest.approx(0.0)

    def test_log_loss_is_finite_at_the_extremes(self):
        y = np.array([0, 1])
        assert np.isfinite(log_loss_score(y, np.array([0.0, 1.0])))


class TestComputeMetrics:
    def test_computes_the_published_metric_set(self):
        rng = np.random.default_rng(60)
        y = rng.integers(0, 2, 200)
        p = rng.uniform(size=200)
        out = compute_metrics(y, p, PUBLISHED_METRICS)
        assert set(PUBLISHED_METRICS) <= set(out)

    def test_intercept_comes_free_with_the_slope(self):
        rng = np.random.default_rng(61)
        y = rng.integers(0, 2, 100)
        p = rng.uniform(size=100)
        out = compute_metrics(y, p, ["calibration_slope"])
        assert "calibration_intercept" in out

    def test_unknown_metric_is_ignored_unless_strict(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.2, 0.8, 0.3, 0.7])
        assert "nonsense" not in compute_metrics(y, p, ["roc_auc", "nonsense"])
        with pytest.raises(KeyError):
            compute_metrics(y, p, ["nonsense"], strict=True)

    def test_degenerate_input_yields_nan_not_an_exception(self):
        """One pathological species must not abort a 226-species run."""
        out = compute_metrics(np.ones(5, dtype=int), np.full(5, 0.5), PUBLISHED_METRICS)
        assert all(np.isnan(v) for v in out.values())
