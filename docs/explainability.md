# Explainability

Explanation must operate at the level of the **methodology**, not just the
model. The questions worth answering are ecological.

| Question | Tool |
|---|---|
| Which **modality** mattered? | `modality_ablation` |
| Which **variable** mattered? | `permutation_importance`, `shap_values` |
| Which **scale** mattered? | `scale_ablation` |
| Did **graph structure** help? | stage-5 comparison, edge attribution |
| Where does it **fail geographically**? | `spatial_residuals` |

---

## Ablation is the primary tool

A SHAP value tells you what the *fitted model* used. An ablation tells you what
the *benchmark* would have lost without it.

Only the second answers **"should we have collected this data?"** — which is
usually the question a conservation practitioner is actually asking.

### Marginal value vs unique contribution

Stage 1 runs both, and the disagreement between them is the finding:

| Comparison | Question | Naming |
|---|---|---|
| **alone** | what can this modality do by itself? | `only-<modality>` |
| **leave-one-out** | what is lost by removing it from the full set? | `<base>-minus-<modality>` |

A modality that scores well alone but adds nothing leave-one-out is
**redundant** with the others — an argument against collecting it. A framework
that only reported one view could not distinguish redundancy from irrelevance.

```python
from sdmbench.explain import modality_ablation

modality_ablation(results, baseline_model="all-modalities", metric="roc_auc")
```

```
modality        comparison       baseline_mean  variant_mean  difference  n_species  wins  losses
climate         leave_one_out            0.762         0.701       0.061        226   198      28
terrain         leave_one_out            0.762         0.749       0.013        226   141      85
soil            leave_one_out            0.762         0.760       0.002        226   118     108
```

Read that as: climate is essential; terrain helps consistently but modestly;
soil is doing nothing here — a near-even win/loss split is noise, not a small
effect.

---

## Grouped permutation importance

Permuting features one at a time **understates** a modality's importance when
its features are correlated, because the others still carry the signal.

```python
from sdmbench.explain import permutation_importance

permutation_importance(
    model, X, y,
    feature_groups={
        "climate": ["bio01", "bio04", "bio12"],
        "terrain": ["slope", "rugosity", "topo"],
    },
)
```

Grouped mode permutes a whole modality with **one** ordering, preserving the
within-group correlation structure while destroying its relationship to the
target. Model-agnostic: it needs only `predict_proba`.

---

## Scale attribution

```python
from sdmbench.explain import scale_ablation

scale_ablation(results, modality="terrain", metric="roc_auc")
```

```
modality  scale    mean    sd     ci_low  ci_high  n_species
terrain   5km      0.771   0.09   0.759   0.783    226
terrain   1km      0.764   0.09   0.752   0.776    226
terrain   25km     0.758   0.10   0.745   0.771    226
terrain   250m     0.749   0.10   0.736   0.762    226
```

"Terrain matters at ~5 km for these species" is an ecological result. When
`multiscale-cnn` uses attention fusion, the learned per-scale weights give the
same signal per species rather than in aggregate.

---

## Spatial residuals

```python
from sdmbench.explain import spatial_residuals

spatial_residuals(coords, y_true, y_prob, n_bins=10)
```

Bins observations on a grid and reports the mean residual per cell. Strong
spatial structure in the residuals means the model is missing something that
varies over space — an unmeasured variable, a dispersal limit, or sampling bias.

This is often more actionable than a global metric: a model at 0.76 overall that
fails systematically in one region is a different problem from one that is
uniformly mediocre.

---

## Attention is not explanation

Attention weights from the fusion and graph models are reported as
`attention_weights`, **never** as `importance`.

Attention indicates what the model attended to. It is a well-documented result
in the interpretability literature that attention weights are not reliable
feature attributions. They are useful as a hypothesis generator; they are not
evidence on their own.

Fusion strategies that learn genuine per-modality scalars (`weighted`, `gated`,
`moe`) do expose `modality_importance()`, and those are interpretable as
"how much the model relied on this modality".

---

## Optional dependencies

SHAP and Captum are optional (`sdmbench[explain]`). Missing ones raise
`MissingDependencyError`, never an import error at collection time.

`shap_values` subsamples by default (500 rows): exact SHAP is exponential in
features, and even the sampling approximations cost minutes on a full background
sample.

---

## The compute-performance frontier

```python
from sdmbench.reporting.plots import plot_compute_frontier

plot_compute_frontier(leaderboard, metric="roc_auc")
```

Marks the Pareto front — models no other model beats on both accuracy and time.
The question it answers: **does the expensive model earn its cost?** A
gradient-boosted tree at 0.72 in 2 seconds and a transformer at 0.73 in an hour
are not obviously ranked, and a leaderboard sorted only by ROC-AUC hides that.

The time axis is log-scaled because the models compared span sub-second
inference to hours.
