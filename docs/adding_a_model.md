# Adding a model

A model adapter's job is deliberately narrow:

> take a `PreparedSplit`, fit on the training part, return a probability of
> presence for each test row.

It does **not** choose the split, the preprocessing, the metrics or the test
set. The benchmark owns those. That separation is what makes the leaderboard a
comparison rather than a collection of anecdotes.

---

## The minimum

```python
from sdmbench.models.base import SklearnAdapter, register_model

@register_model("my-model", "my-alias")
class MyModelAdapter(SklearnAdapter):
    family = "classical"

    def build_estimator(self):
        from mylib import MyClassifier
        return MyClassifier(random_state=self.seed, **self.estimator_kwargs())
```

That is enough for anything with a scikit-learn interface. `SklearnAdapter`
handles fitting, `predict_proba`, `decision_function` fallback, timings and
version recording.

---

## Full control

Override `run()` when fitting and prediction are coupled — as with TabPFN's
in-context learning, or an R model that does both in one subprocess call.

```python
from sdmbench.models.base import FitResult, ModelAdapter, register_model

@register_model("my-model")
class MyAdapter(ModelAdapter):
    family = "neural"
    requires_gpu = False
    supports_categorical = False     # True -> receives DataFrames, not arrays
    uses_test_features = False       # True only for transductive methods

    def check_available(self):
        from sdmbench.optional import require
        require("mylib")             # raises MissingDependencyError -> SKIPPED
        super().check_available()

    def run(self, split) -> FitResult:
        X_train, X_test = self.prepare_features(split)
        ...
        return FitResult(
            probabilities=probs,     # shape (n_test,), in [0, 1]
            fit_seconds=...,
            predict_seconds=...,
            device=self.device_used(),
        )

    def version(self):
        from sdmbench.optional import package_version
        return package_version("mylib")
```

---

## Rules

### Never touch `split.y_test`

The one legitimate use of test *features* is transductive inference (some graph
methods). Declare it with `uses_test_features = True` so the auditor checks the
graph accordingly.

### Optional dependencies must skip, not crash

```python
from sdmbench.optional import require

def check_available(self):
    require("mylib")     # -> MissingDependencyError -> SKIPPED_DEPENDENCY row
```

Never `import mylib` at module scope. A benchmark spanning thousands of jobs
must not die because one adapter's backend is absent — and the skipped model
must still appear in the results, so coverage is honest.

Available statuses: `SKIPPED_DEPENDENCY`, `SKIPPED_NO_GPU`, `SKIPPED_LICENSE`,
`SKIPPED_DATA`.

### Tune on training data only

If your adapter tunes, set `tunes_hyperparameters = True` and use
`InnerCvTuner`, whose signature accepts only training arrays. Use spatially
blocked inner folds for the spatial scenario, so the selected configuration is
the one that generalises across space rather than the one that best exploits
spatial autocorrelation.

### Early stopping monitors training loss

Stopping on test performance is selection on the test set, even though the
labels never enter the loss.

### Don't reimplement the algorithm

sdmbench owns benchmark orchestration, splitting, leakage protection,
aggregation and reporting. It does **not** own learning algorithms. Use
scikit-learn, XGBoost, PyTorch Geometric, the official `tabpfn` package. A
reviewer should be able to verify what "our KNN" is by reading twenty lines.

### Name honestly

`random-forest` is the *published* R model with per-class down-sampling.
`random-forest-sklearn` is a different model and is reported separately.
Silently swapping one for the other would misattribute a published number.

---

## Categoricals

| `supports_categorical` | `prepare_features` returns |
|---|---|
| `False` (default) | numeric arrays, categoricals integer-coded on **training** levels; unseen test levels become `-1` |
| `True` | the DataFrames, categories intact |

TabPFN is a special case: it needs numeric arrays **plus** explicit
`categorical_features_indices`, which `split.categorical_indices` provides.

---

## Registering a model set

```python
# sdmbench/models/base.py
MODEL_SETS["my-family"] = ("my-model", "my-other-model")
```

Set aliases work anywhere a model list is accepted:

```bash
sdmbench benchmark --models my-family,knn
```

---

## Testing

```python
def test_runs_and_returns_valid_probabilities(species_task):
    result = get_model("my-model").run(_prepare(species_task))
    assert result.probabilities.shape == (species_task.test.shape[0],)
    assert np.all((result.probabilities >= 0) & (result.probabilities <= 1))

def test_missing_backend_skips_cleanly():
    with pytest.raises(MissingDependencyError) as e:
        get_model("my-model").check_available()
    assert e.value.status == "SKIPPED_DEPENDENCY"
```

Mark tests needing an optional backend with the skip helpers in `conftest.py`
(`requires_torch`, `requires_xgboost`, …) so the default suite stays runnable
with core dependencies alone.

---

## Checklist

- [ ] Registered with `@register_model`
- [ ] `family` set (`published`, `classical`, `boosting`, `foundation`, `graph`, `neural`, `automl`)
- [ ] `check_available()` raises a skippable error, no module-scope import
- [ ] Never reads `split.y_test`
- [ ] `uses_test_features` declared if transductive
- [ ] `version()` returns the backend version
- [ ] `effective_hyperparameters()` reports what was *actually* used
- [ ] Tests cover the happy path and the missing-backend path
