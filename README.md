# sdmbench

**A reproducible benchmarking framework for species distribution modelling.**

sdmbench is not an AutoML wrapper. Its contribution is that every model is
forced through the *same* ecological benchmark, with the same data, the same
splits, the same preprocessing and the same metrics — so the resulting
comparison means something.

---

## The distinction the whole package rests on

```
INFORMATION  ≠  REPRESENTATION  ≠  METHODOLOGY  ≠  ALGORITHM  ≠  VALIDATION
```

These are routinely conflated in the SDM literature, and conflating them is why
"we compared 12 models" so often answers nothing useful.

| Concept | What it is | Example |
|---|---|---|
| **Dataset** | The observations and covariates | `disdat`: 226 species, 6 regions |
| **Modality** | One *kind* of ecological information | climate, terrain, soil, satellite imagery |
| **Representation** | How a modality becomes features | raw values, PCA, a multi-scale CNN, an EO embedding |
| **Methodology** | Modalities + scales + representations + fusion + structure + objective + model | "climate tabular, terrain multi-scale CNN, cross-attention fusion, Poisson objective, graph transformer" |
| **Algorithm** | The learner | Random Forest, XGBoost, GCN |
| **Benchmark** | The *fixed* experimental design from a published paper | TabPFN-SDM 2026 |

"Random Forest" is an algorithm. A methodology is the whole chain. sdmbench
benchmarks **methodologies**; algorithms are one field inside them.

---

## The pipeline

```
                        DATASET
                           │
          ┌────────────────┼────────────────┐
          ↓                ↓                ↓
       CLIMATE          TERRAIN          EO/OTHER
          ↓                ↓                ↓
   representation    representation    representation
          └────────────────┼────────────────┘
                           ↓
                         FUSION
                           ↓
              SPATIAL / TEMPORAL STRUCTURE
                           ↓
                    LEARNING OBJECTIVE
                           ↓
                         MODEL
                           ↓
                      SUITABILITY
                           ↓
              INDEPENDENT EVALUATION
                           ↓
                      EXPLANATION
```

**sdmbench does not assume that more information or a more complicated model is
better.** It measures whether each methodological choice actually improves
*geographically independent* prediction — which is the only kind that matters
for conservation.

And the workflow that makes a comparison fair:

```
published benchmark  →  reconstruct exact experiment  →  reproduce baselines
        →  reproduce the published method  →  FREEZE the benchmark
        →  run additional models  →  compare identically  →  leaderboard + statistics
```

---

## Install

```bash
pip install -e .                       # core: sklearn models, metrics, splits, reporting
pip install -e ".[boosting]"           # XGBoost, LightGBM, CatBoost
pip install -e ".[deep,graph]"         # torch, torch-geometric
pip install -e ".[tabpfn]"             # TabPFN (Prior Labs License v1.1)
pip install -e ".[raster]"             # rasterio, geopandas, pyproj
pip install -e ".[all]"

# The published MaxNet / BRT / GAM / Random Forest baselines run in R:
Rscript scripts/setup_r_packages.R

sdmbench env check                     # what can and cannot run here
```

Someone who only wants scikit-learn models on a table is never made to install
torch, H2O and every R package.

---

## Quick start

```bash
sdmbench data fetch disdat                       # prepare the benchmark data
sdmbench reproduce tabpfn-sdm-2026               # strict reproduction
sdmbench report tabpfn-sdm-2026 --show-extra     # published vs reproduced
sdmbench leaderboard --scenario nonspatial
sdmbench compare tabpfn-sdm knn --metric roc_auc
```

Python:

```python
from sdmbench import Benchmark

benchmark = Benchmark.from_csv(
    "species.csv",
    target="presence",
    coordinates=("longitude", "latitude"),   # for SPLITTING, not as features
)
results = benchmark.run(models=["knn", "random-forest-sklearn", "xgboost", "tabpfn-sdm"])
print(results.leaderboard())
```

---

## Two modes

**Mode A — strict reproduction.** Reproduce Dinnage & Warren (2026) as
faithfully as the public record allows. The published baselines run in *R*
(`maxnet`, `dismo::gbm.step`, `randomForest`, `mgcv`) because their published
configurations are properties of those packages — substituting a scikit-learn
look-alike would change the numbers.

**Mode B — extended benchmark.** Identical data, splits, preprocessing and
metrics; additional algorithms. This is where KNN, graph networks, AutoML and
future methods are tested against the published state of the art on equal terms.

---

## What makes the comparison trustworthy

**Leakage is checked, not assumed.** Every job carries a `PASS`/`FAIL` audit.
The auditor detects test rows in training, coordinate duplicates, preprocessing
fitted on pooled data, graph edges crossing the train/test boundary, the target
leaking into features, and buffer violations. Its tests work by constructing
deliberate leaks and asserting they are caught.

**Coordinates are not predictors.** They build splits and graph topology.
Making them features lets a model memorise where surveys happened rather than
which environments a species occupies — it inflates interpolation scores and
destroys transferability. It requires an explicit opt-in and is recorded in the
results.

**Aggregation is across species, never the best one.** Every leaderboard row
carries SD, SE, a 95% CI, and the species and region counts. A model evaluated
on 40 species is not silently compared with one evaluated on 226.

**Superiority needs evidence, not a larger mean.** Because every model sees the
same species, comparisons are *paired*: mean and median difference, bootstrap
CI, win/tie/loss, Wilcoxon signed-rank, effect size, and multiple-comparison
correction. `compare` returns `NO_RELIABLE_DIFFERENCE` when that is the honest
answer.

**Published ≠ reproduced.** They live in separate namespaces
(`published_reference` / `reproduced_result`) and a value that was not measured
is reported `NOT RUN` — never echoed back as though it had been.

**Provenance is tracked per constant.** Every protocol value records where it
came from (`PAPER`, `MODEL_CARD`, `DISDAT`, `UNVERIFIED`). Run
`sdmbench provenance tabpfn-sdm-2026` to see it.

**Skips are visible.** A missing dependency, GPU or licence produces a
`SKIPPED_DEPENDENCY` / `SKIPPED_NO_GPU` / `SKIPPED_LICENSE` row, not a silent
omission. The rest of the benchmark continues.

---

## Honest status

> ⚠️ **The upstream repository for the reproduction target is not public.**
> Dinnage & Warren (2026) sec. 2.8 cites `github.com/rdinnager/TabPFN-SDM`, but
> it returned HTTP 404 and does not appear in the owner's public repository
> list. sdmbench's implementation is therefore built from the paper's Methods
> section (which is unusually complete) and the Hugging Face model card's
> `config.json`. Constants that could not be confirmed are tagged `UNVERIFIED`
> and listed by `sdmbench provenance`. Run
> `sdmbench upstream fetch tabpfn-sdm-2026` once it is published to re-check them.

Known gaps, all recorded in code rather than hidden:

- **GAM per-region formulas** — deferred by the paper to Valavi et al. (2022);
  reconstructed from the description. Reading that paper is the highest-value
  next step for fidelity.
- **Ensemble scheme discrepancy** — the paper (sec. 2.3) describes a *balanced
  draw with overlap*; the model card describes *partitioning* the background.
  Both are implemented and the difference is documented, not averaged away.
- **PR-AUC estimator** — the paper used `yardstick` (trapezoidal); sdmbench
  defaults to scikit-learn's step-wise `average_precision`, because trapezoidal
  PR interpolation is optimistically biased. Both are available.
- **DeepMaxent and GNN-SDM** — implemented from published *descriptions*, not
  verified upstream source. Marked experimental; do not report either as a
  reproduction until the parity checks are run.
- **Tests are written but have not been executed.**

---

## The staged benchmark

The full factorial design over modalities × scales × representations × fusion ×
graphs × objectives × models is ~415,000 configurations *per species*. That is
computationally absurd and statistically worse — thousands of spurious
"significant" results by chance alone.

So sdmbench answers one question per stage, carrying the winner forward:

| Stage | Question |
|---|---|
| 1. Information | Which modalities help? |
| 2. Scale | At what spatial scale? |
| 3. Representation | How should information be encoded? |
| 4. Fusion | How should modalities interact? |
| 5. Spatial structure | Do relationships between locations help? |
| 6. Objective | What should the model learn? |
| 7. Model | And *only now*, which algorithm? |

Model choice comes last deliberately. The usual practice — fix the data, sweep
the classifiers — answers the least interesting question first, and conditions
everything else on an arbitrary representation.

Greedy selection can miss interactions, and stage winners are optimistically
biased. Both caveats are printed with every plan; the final number must come
from a clean run, and any superiority claim from the paired tests.

See [`configs/examples/staged_ablation.yaml`](configs/examples/staged_ablation.yaml).

---

## Objectives: background points are not absences

Almost every ML-based SDM treats presence-only data as binary classification —
presences are 1, background is 0. But background points are samples of
*available environment*, and some are certainly suitable habitat.

Point-process objectives take that seriously: background points are quadrature
points approximating an integral, not negative examples. This is the
established equivalence (Warton & Shepherd 2010; Renner & Warton 2013), and
making the objective pluggable lets the benchmark ask whether it matters more
than the architecture.

```
classification:  bce, weighted-bce, focal, pairwise-ranking
point process:   maxent, poisson, deepmaxent
```

A point-process model outputs an *intensity*, not a probability. Rank metrics
stay comparable; calibration slope does not. Each objective declares this, and
the results record it.

---

## Documentation

| Page | |
|---|---|
| [`docs/concepts.md`](docs/concepts.md) | Dataset ≠ Benchmark ≠ Methodology ≠ Model |
| [`docs/tabpfn_sdm_2026_protocol.md`](docs/tabpfn_sdm_2026_protocol.md) | The reproduction protocol, constant by constant |
| [`docs/validation.md`](docs/validation.md) | Splits, leakage, and what "independent" means |
| [`docs/objectives.md`](docs/objectives.md) | Why the learning objective is a benchmark axis |
| [`docs/modalities.md`](docs/modalities.md) | The ecological data ontology |
| [`docs/representations.md`](docs/representations.md) | Encoders, scales, foundation embeddings |
| [`docs/adding_a_paper_recipe.md`](docs/adding_a_paper_recipe.md) | Adding a new published benchmark |
| [`docs/adding_a_model.md`](docs/adding_a_model.md) | Adding a model adapter |
| [`references/papers.yaml`](references/papers.yaml) | Literature, and which of it we verified |

---

## Citation

sdmbench is research software. If you use it, cite the *benchmark* you
reproduced as well as this package — see [`CITATION.cff`](CITATION.cff) and
[`references/papers.yaml`](references/papers.yaml).

## Licence

MIT for sdmbench's own code. It does **not** cover TabPFN weights (Prior Labs
License v1.1), `disdat` (GPL ≥ 3), or any R package invoked through the bridge.
sdmbench downloads these on your behalf and never redistributes them; you are
responsible for each upstream licence. It never bypasses an authentication or
licence gate.
"# sdmbench" 
