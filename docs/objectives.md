# Learning objectives

**What the model is asked to learn** is a benchmark axis, and one the SDM
literature has largely collapsed.

---

## The problem with binary classification

Almost every machine-learning SDM does this:

```
presences      → class 1
background     → class 0
minimise cross-entropy
```

But background points are **not absences**. They are samples of available
environment, drawn to characterise what the landscape offers. Some of them are
certainly suitable habitat where the species simply was not recorded.

Treating them as negatives is an *assumption*, not a fact — and it is baked so
deeply into most pipelines that it is never tested.

---

## Point processes

Presence-only data is a realisation of a point process: the species occurs at a
rate that varies over space, and we observe some of the points. Under that view,
background points are **quadrature points** approximating the integral of the
intensity function over the study area.

This is established theory, not a reinterpretation invented here:

- **Warton & Shepherd (2010)** — presence-background logistic regression
  converges to a Poisson point process as background density increases.
- **Renner & Warton (2013)** — MaxEnt with default settings **is** a fitted
  inhomogeneous Poisson process.

So the two families are not competing heuristics; one is a special case of the
other, and the difference is testable.

---

## The registry

| Objective | Background points are... | Output |
|---|---|---|
| `bce` | negative examples | probability |
| `weighted-bce` | negative examples, down-weighted | re-balanced probability |
| `focal` | negative examples, difficulty-weighted | probability |
| `pairwise-ranking` | lower-ranked comparisons | **score, not a probability** |
| `maxent` | quadrature for the normaliser | Gibbs score |
| `poisson` | quadrature for the intensity integral | **intensity** |
| `deepmaxent` | shared quadrature across species | per-species intensity |

### `weighted-bce`

Inverse-frequency class weights. Equivalent in spirit to the presence-background
weighting the published BRT and GAM baselines use (down-weighting background by
the presence:background ratio), so the neural models are treated consistently
with them rather than each being tuned differently.

### `focal`

Down-weights easy examples. Motivated for SDM by the observation that most
background points are trivially unsuitable; the informative ones sit near the
niche boundary.

### `pairwise-ranking`

Directly targets what ROC-AUC measures — that a random presence outscores a
random background point — *without* asserting that background points are
absences. A useful middle ground.

**Consequence:** the output is a score, so Miller's calibration slope is not
meaningful for a model trained this way. The objective declares it.

### `maxent`

The Gibbs form:

```
L = -(1/n_p) · Σ_presences f(x)  +  log( (1/n_b) · Σ_background e^{f(x)} )
```

Background enters only through the normalising constant.

### `poisson`

The Berman-Turner down-weighted Poisson form. Presence points carry a small
weight; background points share the study area between them, approximating the
intensity integral.

Output is an intensity. `to_probability` applies the complementary log-log link
(`1 - exp(-λ)`) — the probability of at least one point, which is the transform
MaxEnt uses for its own output.

---

## Metrics under a point-process objective

This matters and is easy to get wrong:

| Metric | Comparable across objectives? |
|---|---|
| ROC-AUC | **Yes** — rank-based |
| PR-AUC | **Yes** — rank-based |
| Boyce index | **Yes** — rank-based |
| Miller's calibration slope | **No** — an intensity is not a probability |
| Brier, log loss | **No** — proper scoring rules assume probabilities |

sdmbench records the objective in every result row so a leaderboard mixing
families can be read correctly.

---

## DeepMaxent — status

`DeepMaxentObjective` combines three ideas:

1. **Shared representation** — one neural feature extractor across every
   species, so rare species borrow strength from common ones.
2. **Per-species intensity heads.**
3. **Per-species normalisation** — the Poisson likelihood is normalised within
   each species, so a species with 900 presences does not dominate one with 12.

### ⚠️ Not verified against upstream

> This implements the objective from its **published description**, not from
> verified upstream source. At the time of writing the reference implementation
> had not been located and cross-checked.
>
> It is registered with `maturity="experimental"` and
> `verified_against_upstream=False`, and `info()` says so.
>
> **Do not report a DeepMaxent reproduction from this implementation** until the
> parity check below has been run.

### The verification path

1. Locate the paper and its reference implementation. Record both in
   `references/papers.yaml`.
2. `sdmbench upstream fetch deepmaxent-2026` (add the source to
   `sdmbench/upstream/metadata.py` first).
3. Run both implementations on the same species with the same seed and
   background sample.
4. Compare **predicted intensities**, not just ROC-AUC — rank metrics can agree
   while the fitted intensity differs by a constant factor, which would hide a
   real discrepancy.
5. If they agree, retag the objective `PAPER` / `UPSTREAM_CODE`. If not, fix
   ours and document what differed.

The same procedure applies to the GNN-SDM patch graph, whose Self-Organizing Map
patching step is currently **substituted** with k-means.

---

## Which models can vary the objective

Only neural and graph models. Classical adapters — scikit-learn, XGBoost, R
MaxNet, `gbm.step` — have fixed internal objectives and simply report which one
they use.

`Methodology.validate()` warns rather than silently ignoring the setting:

```
[WARN ] objective: model 'random-forest-sklearn' (classical) has a fixed
        internal objective; 'poisson' cannot be applied to it
     -> use a neural or graph model to vary the objective
```

A benchmark row records the objective *actually optimised*, never an
aspirational one.

---

## Adding an objective

```python
from sdmbench.objectives.base import OBJECTIVES, Objective, objective_spec

@OBJECTIVES.register(
    "my-objective",
    spec=objective_spec("One line on what it optimises.", requires=("torch",)),
)
class MyObjective(Objective):
    treats_background_as = "..."   # be precise; this is reported
    output_meaning = "..."         # says whether calibration metrics apply
    reference = "Author et al. YEAR"

    def loss(self, logits, target, *, weights=None, **kwargs):
        ...

    def to_probability(self, logits):
        """Map output to (0,1) for metrics. Rank-preserving if possible."""
        ...
```

Be honest in `treats_background_as` and `output_meaning` — they propagate into
every result row and into the reports.
