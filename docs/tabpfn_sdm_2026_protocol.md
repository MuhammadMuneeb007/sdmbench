# The TabPFN-SDM 2026 protocol

Reproducing:

> Dinnage, Russell & Dan L. Warren (2026). *A Niche in the Machine: The Promise
> of AI Foundation Models for Species Distribution Modeling.* EcoEvoRxiv.
> [doi:10.32942/X2VQ10](https://doi.org/10.32942/X2VQ10)

This page records **every protocol constant and where it came from**, so a
reader can check our implementation against the source rather than trusting it.

---

## Provenance status

> **The authors' repository is not public.** The paper (sec. 2.8) cites
> `https://github.com/rdinnager/TabPFN-SDM`, but that URL returned HTTP 404 and
> the repository does not appear in the owner's public repository list or in
> GitHub search. It is presumably still private.

So the implementation is built from:

1. **The preprint PDF** — the primary source. Its Methods section is unusually
   complete and specifies most constants verbatim.
2. **The Hugging Face model card** for `rdinnager/tabpfn-sdm-finetuned`, and
   especially its `config.json`, which independently confirms the seed, species
   split fractions, per-region categorical variables and sub-batch cap.
3. **The `disdat` R package source**, for regional CRSs, group codes and the
   data schema.

Anything none of those pins down is tagged `UNVERIFIED`:

```bash
sdmbench provenance tabpfn-sdm-2026     # exits 3 while gaps remain
sdmbench upstream fetch tabpfn-sdm-2026 # re-check once the repo is published
```

---

## 1. Data

`disdat` (Elith et al. 2020): 226 anonymised species across six regions.

| Region | CRS | Units | Groups |
|---|---|---|---|
| AWT | EPSG:28355 (UTM 55S) | metres | bird, plant |
| CAN | EPSG:4008 | **degrees** | — |
| NSW | EPSG:4326 | **degrees** | ba, db, nb, ot, ou, rt, ru, sr |
| NZ | EPSG:27200 (NZMG) | metres | — |
| SA | EPSG:4326 | **degrees** | — |
| SWI | EPSG:21781 (CH1903) | metres | — |

*Source: `disdat::disCRS` (`R/disOther.R`) and the region help pages.*

**This table is why the 10 km buffer cannot be a constant in coordinate units.**
For AWT/NZ/SWI it is a Euclidean radius of 10000; for CAN/NSW/SA it must be a
geodesic distance. See §4.

Each species has:

- **`po`** — presence-only training records
- **`bg`** — 10,000 background points per region (available environment, *not*
  absences)
- **`pa`** — independent presence-absence survey data, from *different surveys*
- **`env`** — covariates at the survey sites

Training = this species' presences + the region's background.
Test = the independent survey for **this species' own group**.

### 1.1 Predictors — a subset, not `disPredictors()`

Paper sec. 2.1.1 lists per-region predictors that are **smaller** than what
`disdat::disPredictors()` returns:

| Region | Paper | disdat | Dropped |
|---|---|---|---|
| AWT | 8 | 13 | bc01, bc17, bc20, bc31, bc33 |
| CAN | 7 | 7 | — |
| NSW | 12 | 13 | tempmin |
| NZ | 11 | 11 | — |
| SA | 8 | 8 | — |
| SWI | 12 | 12 | — |

Using the full set would change every number, so the published selection is
hard-coded in `REGION_PREDICTORS` and pinned by a test.

```
AWT  bc04 bc05 bc06 bc12 bc15 slope topo tri
CAN  alt asp2 ontprec ontslp onttemp ontveg watdist
NSW  cti disturb mi rainann raindq rugged soildepth soilfert solrad tempann topo vegsys
NZ   age deficit hillshade mas mat r2pet slope sseas toxicats tseas vpd
SA   sabio12 sabio15 sabio17 sabio18 sabio2 sabio4 sabio5 sabio6
SWI  bcc calc ccc ddeg nutri pday precy sfroyy slope sradyy swb topo
```

### 1.2 Categorical variables

Five, across four regions: `ontveg` (CAN), `vegsys` (NSW), `toxicats` and `age`
(NZ), `calc` (SWI).

*Confirmed twice: paper sec. 2.1.1, and the model card `config.json`.*

---

## 2. Preprocessing

Paper sec. 2.1.2 — a `tidymodels` recipe with exactly three sequential steps:

1. remove zero-variance numeric predictors (`step_zv`)
2. Yeo-Johnson transform all numeric predictors (`step_YeoJohnson`)
3. normalise to mean 0, unit SD (`step_normalize`)

> *"The recipe was fitted on training data alone; the same transformation
> parameters were then applied to test data to prevent information leakage."*

Two consequences encoded in `sdmbench.preprocessing.SdmRecipe`:

- Numeric steps apply to **numeric predictors only**. Categorical predictors
  pass through as categories.
- `fit()` sees training rows only, and `fitted_on` is recorded so the leakage
  auditor can *verify* this rather than assume it.

**Fidelity note.** `recipes::step_YeoJohnson` bounds the estimated lambda to
[-5, 5]; scikit-learn's `PowerTransformer` does not. sdmbench applies the bound
so the two agree.

### 2.1 Categorical handling per model

> *"MaxNet, BRT, Random Forest, and GAM all received factor variables as native
> R factors without conversion. Only the local TabPFN implementation required
> explicit specification of categorical feature indices."*

So the R adapters get DataFrames with categories intact; TabPFN gets integer
arrays plus `categorical_features_indices`.

---

## 3. Non-spatial scenario

> *"Standard evaluation used all available training data without spatial
> considerations."* (sec. 2.5)

**No random splitting happens.** Training and test are two different data
collections. `sdmbench.splits.random.identity_split` is a documented no-op, so
that nobody later "fixes" it by inserting a `train_test_split`.

n = 226 species.

---

## 4. Spatial scenario — the 10 km buffer

> *"we created spatially-filtered training sets by excluding training points
> within 10 km of any test location"* (sec. 2.1.2)
>
> *"Training points within 10 km of any test location were excluded, testing
> model transferability to novel geographic areas."* (sec. 2.5)

Note precisely what this is:

- A **one-sided filter on the training set**. Test data is untouched.
- **Not** a spatial block cross-validation.
- **Not** `train_test_split` with a spatial flavour.

Substituting either changes which species survive and every reported number.

### 4.1 Distance is not a coordinate difference

The authors built their spatial features with `sf`.
`sf::st_is_within_distance` dispatches on the CRS: planar distance for a
projected CRS, great-circle distance via **s2** for a geographic one.

`sdmbench.splits.spatial.points_within_distance` reproduces that dispatch and
uses **s2's own earth radius, 6,371,010 m**, so the Python and R filters select
the same rows to the metre. `rbridge/scripts/spatial_split.R` is the R
reference the parity test compares against.

Implementation detail: geographic coordinates are projected onto the unit
sphere and the arc threshold converted to a chord, so an exact great-circle
radius query runs through a Euclidean KD-tree. The chord is monotone in the
arc, so the two select identical point sets.

Boundary is `<=`: a point at exactly 10 km is *inside* the buffer, matching `sf`.

### 4.2 Species exclusion

> *"Approximately 41 species were excluded from spatial evaluation due to
> single-class datasets after filtering."* (sec. 2.5)

After the filter, some species retain only presences or only background, and
classification is undefined. sdmbench raises `InsufficientDataError`, which
becomes a `SKIPPED_DATA` row — the species is *recorded as excluded*, not
silently dropped.

n = 185 species.

---

## 5. Models

### 5.1 Published baselines — run in R

> *"For each method, we followed the best-practice configurations specified in
> Valavi et al. (2022), ensuring that our comparison reflects each algorithm's
> optimized rather than naive performance."* (sec. 2.2.1)

Those configurations are properties of specific R packages. Reproducing them in
Python would change the numbers.

| Model | Package | Settings (verbatim from sec. 2.2.1) |
|---|---|---|
| **MaxNet** | `maxnet` | `regmult = 1`; feature classes linear, quadratic, product, threshold, hinge (`lqpht`); cloglog output |
| **BRT** | `dismo::gbm.step` | tree complexity 1 if <50 presences else 5; lr 0.001; bag fraction 0.75; 5-fold CV; max 10,000 trees; background down-weighted by the presence:background ratio |
| **Random Forest** | `randomForest` | 1,000 trees; **per-class down-sampling** — both classes sampled to the size of the minority class; default mtry/nodesize |
| **GAM** | `mgcv` | penalised thin-plate splines for continuous, factor terms for categorical; binomial/logit; REML; same weighting as BRT |
| **MaxEnt (Java)** | `dismo::maxent` | automatic feature selection; cloglog output |

> ⚠️ The published Random Forest is **not** `sklearn.ensemble.RandomForestClassifier`.
> sdmbench registers them as `random-forest` and `random-forest-sklearn` so they
> can never be confused, and a test asserts they are different classes.

`UNVERIFIED`: the per-region GAM formulas (deferred to Valavi et al. 2022), and
whether `randomForest`'s `replace` is TRUE (its default is used).

### 5.2 TabPFN variants (sec. 2.2.2)

The paper's own names — none invented:

| Name | Configuration |
|---|---|
| `tabpfn-default` | n_estimators 8, softmax_temperature 0.9, balance_probabilities False, average_before_softmax False |
| `tabpfn-real` | the `v2_5_real` checkpoint |
| `tabpfn-balanced` | balance_probabilities **True**, average_before_softmax **True** |
| `tabpfn-ss` | subsample ensemble, K = 16 (§6 below) |
| `tabpfn-sdm` | finetuned checkpoint + subsample ensemble |

All used the Python `tabpfn` package (v2.5) via `reticulate`, on a local CUDA GPU.

---

## 6. Ensemble class balancing (sec. 2.3)

The motivating asymmetry, in the authors' words:

> *"Presence records are valuable and relatively scarce... Pseudo-absences, by
> contrast, are interchangeable samples from the background environmental
> space; any given pseudo-absence point could be replaced by another randomly
> drawn background point without loss of information."*

The published procedure:

1. Retain **all** presence records in every ensemble member
2. For each of K members, draw a balanced sample of pseudo-absences **equal in
   number to the presences**
3. K training sets, each with a different balanced subsample, **allowing overlap**
4. Fit TabPFN on each
5. **Average logits** across members before applying softmax

with K = 16, `n_estimators = 16`, `average_before_softmax = True`,
`balance_probabilities = True`.

### 6.1 A genuine discrepancy, documented not averaged

The **Hugging Face model card** describes a *different* procedure for
`run_tabpfn_finetuned_ensemble()`:

> *"Keeps ALL presences in every sub-batch; **Partitions absences across
> sub-batches** for 100% data usage; Averages predictions across sub-batches"*

These are not the same. The paper draws `n_presence` background points per
member *with overlap between members*; the model card divides the whole
background into *disjoint* blocks.

`sdmbench.models.ensemble.PresenceBackgroundEnsemble` implements **both**
(`scheme="balanced"` / `scheme="partition"`). The paper's wording is the default
for the published `tabpfn-ss` variant, and a test asserts the two schemes remain
distinguishable. Resolving which the authors actually used at inference requires
their repository — it is tagged `UNVERIFIED`.

---

## 7. Finetuning (sec. 2.4)

Not needed to *run* the benchmark — the checkpoints are released — but recorded
so the released weights can be reproduced.

**Species split** (sec. 2.4.1): 62.5% finetuning (~141), 7.5% validation (~17),
30% held out (~68). Seed `12345` (model card `config.json`; the paper says only
"a fixed random seed").

**Step 1 — cross-validation finetuning.** 3-fold CV, 5 repeats per species
(15 train/test pairs), absences subsampled to match the presence count. 40
epochs, lr 1e-5, OneCycleLR.

**Step 2 — benchmark finetuning.** The actual benchmark train/test pairs, 5
balanced samples per species per epoch. 100 epochs, lr 1e-6, initialised from
the best Step 1 checkpoint by validation ROC-AUC.

Validation every 900 batches using 16-member ensembles. Max training set 1,500
samples per split. Separate models for the non-spatial and spatial protocols.

| Checkpoint | Step 1 | Step 2 | Val ROC-AUC | Val PR-AUC |
|---|---|---|---|---|
| `tabpfn-sdm-nonspatial.pt` | 40 | 100 | 0.747 | 0.261 |
| `tabpfn-sdm-spatial.pt` | 25 | 50 | 0.653 | 0.144 |

> ⚠️ **These are checkpoint-validation metrics, not benchmark results.** They
> were measured on held-out *species* during finetuning. They are **not**
> comparable with the paper's 0.762 / 0.699 benchmark aggregates. sdmbench keeps
> them in `checkpoint_validation_metrics`, never in `published_reference`.

---

## 8. Metrics (sec. 2.6)

Computed with `yardstick`.

**ROC-AUC** — primary discrimination metric.

**PR-AUC** — complementary under imbalance.

**Miller's calibration slope** — and this one is easy to get wrong:

> *"Miller's calibration slope and intercept are obtained by regressing observed
> outcomes on the **logit-transformed predicted probabilities**."*

So the estimator is a **logistic** regression:

```r
glm(observed ~ logit(predicted), family = binomial)
```

with the slope being the coefficient on `logit(p)`. An OLS regression of
outcomes on probabilities, or a binned-means line, gives a *different number*
that would look entirely plausible in a results table. `test_metrics.py`
asserts the two differ, so a substitution fails loudly.

Only the slope is interpretable here, and the paper says why:

> *"no presence-only model can achieve absolute calibration (intercept = 0)
> because the baseline prevalence is confounded with the intercept term. We
> therefore focus on the slope as a measure of ratio calibration."*

**Known divergence.** `yardstick::pr_auc` integrates the PR curve
trapezoidally; scikit-learn's `average_precision_score` is step-wise. sdmbench
defaults to the step-wise estimator because trapezoidal PR interpolation is
optimistically biased, and exposes `method="trapezoid"` for parity. Tagged
`UNVERIFIED`.

---

## 9. Published results

Reference targets only. sdmbench never merges them into computed results.

**Non-spatial, mean ROC-AUC across 226 species:**

| Model | ROC-AUC |
|---|---|
| Finetuned TabPFN | **0.762** |
| MaxNet | 0.732 |
| Random Forest | 0.727 |
| BRT | 0.724 |
| GAM | 0.717 |

**Spatial, 185 species:** finetuned TabPFN 0.699; traditional methods 0.656–0.683.

**Calibration:** finetuned TabPFN slope 1.110.

**Held-out species** (never seen in finetuning): 0.763 — essentially identical
to 0.762, which is the paper's evidence for generalisation rather than
memorisation.

---

## 10. Reproduction status bands

`sdmbench report` classifies each comparison. The bands are heuristics for
triage, are stated in every report, and are configurable:

| Status | Band | Rationale |
|---|---|---|
| `MATCH` | ≤ 0.010 | The order of Monte-Carlo variation from the protocol's own stochastic components — background subsampling, RF bagging, BRT's internal CV folds, the K=16 ensemble draw — when the seed cannot be matched exactly. |
| `CLOSE` | ≤ 0.025 | Roughly the spread between the paper's four traditional baselines (0.717–0.732). A discrepancy this large is the size of a real between-method difference and cannot be dismissed as noise. |
| `MISMATCH` | > 0.025 | Investigate. |
| `NOT RUN` | — | No computed value. The published number is **not** echoed back. |

A `MATCH` is evidence the pipeline behaves, not proof of exact reproduction.

---

## 11. Computational environment (sec. 2.8)

R 4.x with `targets` (~7,500 targets) and `crew` — 18 workers for traditional
models, 1 sequential GPU worker, 2 for cloud API. Python via `reticulate` with
conda: `tabpfn` 2.5, `torch`, `numpy`. GPU: NVIDIA RTX 4000 ADA. Finetuning used
20-core parallelism via `furrr`.

sdmbench does not reimplement `targets`; its own resumable result store plays
the same role.

---

## Running it

```bash
sdmbench provenance tabpfn-sdm-2026            # read this first
sdmbench data fetch disdat
sdmbench reproduce tabpfn-sdm-2026 --max-species 3   # smoke test
sdmbench reproduce tabpfn-sdm-2026                   # full run
sdmbench report tabpfn-sdm-2026 --show-extra
```
