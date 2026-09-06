"""Model adapter and ensemble tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdmbench.exceptions import MissingDependencyError, SdmbenchError
from sdmbench.models.base import (
    MODEL_SETS,
    FitResult,
    ModelAdapter,
    ModelStatus,
    get_model,
    list_models,
    resolve_model_names,
)
from sdmbench.models.ensemble import (
    PUBLISHED_K,
    PresenceBackgroundEnsemble,
    average_logits,
    average_probabilities,
)
from sdmbench.models.sklearn_models import KNN_GRID
from sdmbench.models.tuning import InnerCvTuner, expand_grid
from tests.conftest import requires_xgboost


class TestRegistry:
    def test_core_models_are_registered(self):
        names = list_models()
        for expected in ("knn", "logistic-regression", "random-forest-sklearn",
                         "maxnet", "brt", "gam", "tabpfn-sdm", "gcn"):
            assert expected in names

    def test_unknown_model_raises_a_helpful_error(self):
        with pytest.raises(KeyError, match="Available"):
            get_model("does-not-exist")

    def test_underscores_and_hyphens_are_interchangeable(self):
        assert get_model("random_forest_sklearn").name == get_model("random-forest-sklearn").name

    def test_set_aliases_expand(self):
        expanded = resolve_model_names(["published"])
        assert set(expanded) == set(MODEL_SETS["published"])

    def test_expansion_deduplicates_and_preserves_order(self):
        expanded = resolve_model_names(["knn", "classical", "knn"])
        assert expanded[0] == "knn"
        assert len(expanded) == len(set(expanded))

    def test_published_random_forest_is_not_the_sklearn_one(self):
        """They are different models and must never be conflated."""
        published = get_model("random-forest")
        sklearn_rf = get_model("random-forest-sklearn")
        assert type(published) is not type(sklearn_rf)
        assert published.family == "published"
        assert sklearn_rf.family == "classical"


class TestSklearnAdapters:
    def test_knn_defaults_match_the_leopard_configuration(self):
        """Manhattan distance, k=5 -- the configuration under test, not a claim."""
        knn = get_model("knn")
        estimator = knn.build_estimator()
        assert estimator.n_neighbors == 5
        assert estimator.metric == "manhattan"

    def test_knn_caps_k_at_the_training_size(self):
        knn = get_model("knn", n_neighbors=50)
        X = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)})
        y = np.array([0, 1] * 5)
        knn.fit(X.to_numpy(), y)
        assert knn.hyperparameters["n_neighbors"] <= 10

    def test_fit_predict_returns_probabilities_in_range(self, species_task):
        from sdmbench.benchmarks.tabpfn_sdm_2026 import TabPFNSDM2026Benchmark

        split = _prepare(species_task)
        result = get_model("logistic-regression").run(split)
        assert isinstance(result, FitResult)
        assert result.probabilities.shape == (split.n_test,)
        assert np.all((result.probabilities >= 0) & (result.probabilities <= 1))

    def test_predict_before_fit_raises(self):
        with pytest.raises(SdmbenchError, match="not fitted"):
            get_model("knn").predict_proba(np.zeros((3, 2)))

    def test_dummy_scores_exactly_chance(self, species_task):
        """The sanity floor: prevalence-only prediction is ROC-AUC 0.5."""
        from sdmbench.metrics import roc_auc

        split = _prepare(species_task)
        result = get_model("dummy").run(split)
        assert roc_auc(split.y_test, result.probabilities) == pytest.approx(0.5, abs=1e-9)

    def test_timings_are_recorded_separately(self, species_task):
        result = get_model("logistic-regression").run(_prepare(species_task))
        assert result.fit_seconds >= 0
        assert result.predict_seconds >= 0
        assert result.total_seconds == pytest.approx(
            result.fit_seconds + result.predict_seconds
        )


class TestOptionalBackends:
    def test_missing_backend_raises_a_skippable_error(self):
        """Availability failure must be skippable, never a hard crash."""
        from sdmbench.optional import have

        if have("xgboost"):
            pytest.skip("xgboost is installed; cannot exercise the missing path")
        with pytest.raises(MissingDependencyError) as excinfo:
            get_model("xgboost").check_available()
        assert excinfo.value.status == ModelStatus.SKIPPED_DEPENDENCY.value
        assert "sdmbench[boosting]" in str(excinfo.value)

    @requires_xgboost
    def test_xgboost_runs_when_installed(self, species_task):
        result = get_model("xgboost", n_estimators=10).run(_prepare(species_task))
        assert result.probabilities.shape == (len(species_task.test),)


class TestPresenceBackgroundEnsemble:
    def test_balanced_scheme_keeps_all_presences_in_every_member(self):
        """Paper sec. 2.3, step 1: 'Retain all presence records in every member'."""
        y = np.r_[np.ones(20), np.zeros(200)].astype(int)
        members = PresenceBackgroundEnsemble(n_members=8, seed=0).build_members(y)
        assert len(members) == 8
        for member in members:
            assert member.n_presence == 20

    def test_balanced_scheme_draws_absences_equal_to_presences(self):
        """Step 2: 'a balanced random sample of pseudo-absences equal in number'."""
        y = np.r_[np.ones(15), np.zeros(300)].astype(int)
        for member in PresenceBackgroundEnsemble(n_members=4, seed=1).build_members(y):
            assert member.n_background == 15

    def test_balanced_members_differ_but_may_overlap(self):
        """Step 3: 'allowing overlap between members'."""
        y = np.r_[np.ones(10), np.zeros(500)].astype(int)
        members = PresenceBackgroundEnsemble(n_members=5, seed=2).build_members(y)
        signatures = {tuple(m.indices.tolist()) for m in members}
        assert len(signatures) > 1

    def test_partition_scheme_uses_each_background_point_once(self):
        """The model card's alternative: disjoint partitions, 100% usage."""
        y = np.r_[np.ones(10), np.zeros(100)].astype(int)
        ensemble = PresenceBackgroundEnsemble(
            n_members=5, scheme="partition", max_train_size=None, seed=3
        )
        members = ensemble.build_members(y)
        background_indices = np.concatenate(
            [m.indices[np.isin(m.indices, np.flatnonzero(y == 0))] for m in members]
        )
        assert len(background_indices) == len(np.unique(background_indices)) == 100

    def test_schemes_are_genuinely_different(self):
        """Guards the documented paper/model-card discrepancy.

        If these ever produce the same partition, the distinction sdmbench
        documents has silently collapsed.
        """
        y = np.r_[np.ones(10), np.zeros(200)].astype(int)
        balanced = PresenceBackgroundEnsemble(
            n_members=4, scheme="balanced", seed=4
        ).build_members(y)
        partition = PresenceBackgroundEnsemble(
            n_members=4, scheme="partition", max_train_size=None, seed=4
        ).build_members(y)
        assert balanced[0].n_background != partition[0].n_background

    def test_max_train_size_drops_background_before_presences(self):
        """Presences are scarce; the GPU-memory cap must not spend them."""
        y = np.r_[np.ones(50), np.zeros(5_000)].astype(int)
        ensemble = PresenceBackgroundEnsemble(
            n_members=2, scheme="partition", max_train_size=100, seed=5
        )
        for member in ensemble.build_members(y):
            assert member.n_presence == 50
            assert len(member.indices) <= 100

    def test_single_class_input_yields_one_member(self):
        members = PresenceBackgroundEnsemble(n_members=8).build_members(np.ones(20, dtype=int))
        assert len(members) == 1

    def test_default_k_is_sixteen(self):
        """Paper sec. 2.3: 'The choice of K = 16 ensemble members'."""
        assert PUBLISHED_K == 16
        assert PresenceBackgroundEnsemble().n_members == 16

    def test_is_reproducible_for_a_seed(self):
        y = np.r_[np.ones(20), np.zeros(200)].astype(int)
        a = PresenceBackgroundEnsemble(n_members=4, seed=7).build_members(y)
        b = PresenceBackgroundEnsemble(n_members=4, seed=7).build_members(y)
        for left, right in zip(a, b):
            assert np.array_equal(left.indices, right.indices)

    def test_rejects_an_unknown_scheme(self):
        with pytest.raises(ValueError, match="scheme"):
            PresenceBackgroundEnsemble(scheme="nonsense")

    def test_fit_predict_averages_across_members(self):
        y = np.r_[np.ones(20), np.zeros(100)].astype(int)
        X = np.random.default_rng(0).normal(size=(120, 3))
        X_test = np.random.default_rng(1).normal(size=(30, 3))
        calls: list[int] = []

        def fit_predict(X_member, y_member, X_eval):
            calls.append(len(y_member))
            return np.full(len(X_eval), 0.5 + 0.01 * len(calls))

        out = PresenceBackgroundEnsemble(n_members=4, seed=0).fit_predict(
            fit_predict, X, y, X_test
        )
        assert len(calls) == 4
        assert out.shape == (30,)


class TestLogitAveraging:
    def test_logit_and_probability_averaging_differ(self):
        """'Average logits ... before applying softmax' (paper sec. 2.3)."""
        predictions = [np.array([0.1, 0.9]), np.array([0.4, 0.6])]
        assert not np.allclose(
            average_logits(predictions), average_probabilities(predictions)
        )

    def test_logit_average_of_identical_inputs_is_the_input(self):
        p = np.array([0.2, 0.5, 0.8])
        assert np.allclose(average_logits([p, p, p]), p, atol=1e-6)

    def test_output_stays_in_the_unit_interval(self):
        out = average_logits([np.array([0.0, 1.0]), np.array([1.0, 0.0])])
        assert np.all((out > 0) & (out < 1))

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            average_logits([])


class TestTuning:
    def test_grid_expansion_is_deterministic(self):
        grid = expand_grid({"a": [1, 2], "b": ["x", "y"]})
        assert len(grid) == 4
        assert expand_grid({"a": [1, 2], "b": ["x", "y"]}) == grid

    def test_knn_grid_covers_the_documented_options(self):
        assert KNN_GRID["n_neighbors"] == [3, 5, 7, 9, 15]
        assert set(KNN_GRID["metric"]) == {"euclidean", "manhattan"}

    def test_tuner_never_receives_test_data(self):
        """Structural guarantee: search() takes only training arrays."""
        import inspect

        parameters = set(inspect.signature(InnerCvTuner.search).parameters)
        assert not {"X_test", "y_test"} & parameters

    def test_search_selects_a_configuration(self):
        from sklearn.neighbors import KNeighborsClassifier

        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, 3))
        y = (X[:, 0] + rng.normal(0, 0.4, 120) > 0).astype(int)
        tuner = InnerCvTuner(
            lambda params: KNeighborsClassifier(**params),
            {"n_neighbors": [3, 9]},
            n_folds=3,
        )
        result = tuner.search(X, y)
        assert result.best_params.get("n_neighbors") in {3, 9}
        assert result.used_test_data is False


def _prepare(task):
    """Run a task through the published recipe to get a PreparedSplit."""
    from sdmbench.benchmarks.tabpfn_sdm_2026 import TabPFNSDM2026Benchmark

    benchmark = TabPFNSDM2026Benchmark.__new__(TabPFNSDM2026Benchmark)
    from sdmbench.config import RunConfig

    benchmark.config = RunConfig(benchmark="tabpfn-sdm-2026")
    benchmark.buffer_m = 10_000.0
    return benchmark.prepare(task, "nonspatial")
