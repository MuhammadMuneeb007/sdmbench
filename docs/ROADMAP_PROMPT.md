# sdmbench — next-phase prompt

**A prompt for the next development session, written against what actually
exists in this repository.**

This is not the original greenfield brief. It is a **gap analysis**: what is
built and verified, what is written but never executed, what is wrong or
substituted, and what to do next in priority order. Paste the relevant section
into a new session.

Repository: `MuhammadMuneeb007/sdmbench`
Written: 2026-09-06, against sdmbench 0.1.0

---

## 0. Read this before writing any code

### What is already true

Do not re-derive these. They were established by reading primary sources.

1. **The upstream repository is NOT public.** Dinnage & Warren (2026) sec. 2.8
   cites `github.com/rdinnager/TabPFN-SDM`. It returns **HTTP 404**, does not
   appear in the owner's public repo list, and does not appear in GitHub search.
   Everything in the recipe comes from the preprint PDF, the Hugging Face model
   card's `config.json`, and the `disdat` R source.

2. **The paper's predictor lists are SUBSETS of `disPredictors()`.** AWT uses 8
   of 13, NSW 12 of 13 (drops `tempmin`). Hard-coded in `REGION_PREDICTORS` and
   pinned by a test. Using the full set would change every number.

3. **The paper and the model card describe DIFFERENT ensemble schemes.** Paper
   sec. 2.3: balanced draw per member, *overlap allowed*. Model card: *partition*
   the background across sub-batches. Both are implemented
   (`scheme="balanced"` / `"partition"`); the discrepancy is documented, not
   averaged away.

4. **Miller's calibration slope is `glm(observed ~ logit(p), binomial)`** — a
   *logistic* regression on logits, not OLS on probabilities. A test asserts the
   two differ.

5. **`disdat` regions differ in CRS units.** AWT/NZ/SWI are projected in metres;
   CAN/NSW/SA are geographic degrees. The 10 km buffer therefore dispatches on
   CRS and uses s2's earth radius (6,371,010 m) to match `sf`.

6. **The published Random Forest is not sklearn's.** It is R `randomForest` with
   per-class down-sampling. Registered separately as `random-forest` vs
   `random-forest-sklearn`.

### What has actually been executed

| | |
|---|---|
| Verified working | 79 modules import; end-to-end run on synthetic data (fit → predict → metrics → leaderboard → paired comparison → manifest → parquet); registries populate; spatial buffer maths checked against haversine |
| **Written, never executed** | **the entire test suite**; every R script; TabPFN/torch/graph/AutoML/raster/foundation adapters; all cross-language parity tests |

**Never report an unexecuted test as passing.**

---

## 1. Highest priority — close the fidelity gaps

These affect whether the reproduction is *correct*, and they are ordered by
value per unit of work.

### 1.1 Read Valavi et al. (2022) and fix the GAM formulas

**Why first:** it is the single largest `UNVERIFIED` item, it is freely
obtainable (unlike the upstream repo), and it blocks a faithful GAM baseline —
one of the four numbers the reproduction is judged against.

- Paper: Ecological Monographs, doi:10.1002/ecm.1486
- Extract the **per-region GAM formulas** (specifically the basis dimension `k`
  per predictor), the RF down-sampling configuration, and the BRT settings.
- Update `rbridge/scripts/gam.R` to take authoritative formulas; retag
  `gam_region_formulas` from `UNVERIFIED` to `CITED_REFERENCE`.
- Also resolve `random_forest_replace` (the paper gives the sample size but not
  whether sampling is with replacement; `randomForest`'s default TRUE is used).

### 1.2 Run the cross-language parity tests

They are written in `tests/test_cli_config_manifest.py` and marked
`@pytest.mark.r`. Running them is the first real evidence the Python
implementations are right.

```bash
Rscript scripts/setup_r_packages.R
pytest -m r -v
```

Three parity checks, in order of consequence:

1. **Spatial filter vs `sf`** — must select *identical rows*. If this
   disagrees, every spatial number is wrong.
2. **ROC-AUC / Miller slope vs `yardstick`** — should agree to ~1e-6 / 1e-4.
3. **Boyce vs `tidysdm::boyce_cont()`** — expect small disagreement; **pin the
   size** rather than forcing equality, and document it.

Expect PR-AUC to differ: `yardstick` integrates trapezoidally, sklearn's
`average_precision` is step-wise. Do not "fix" this by switching the default —
trapezoidal PR interpolation is optimistically biased. Quantify and document.

### 1.3 Run the unit suite and fix what breaks

```bash
pip install -e ".[dev]"
pytest -m "not integration and not r and not gpu and not network" -v
```

The suite was written from the implementation without execution. Expect fixture
and import errors. **Fix the tests where they are wrong; fix the code where the
test is right.** Do not delete a failing test to make the suite green.

Known likely issue: `tests/test_models_and_ensemble.py::_prepare` constructs a
benchmark via `__new__` to bypass dataset loading — fragile, and a better
fixture should replace it.

### 1.4 Smoke-test the real benchmark

```bash
sdmbench data fetch disdat          # needs R + disdat
sdmbench data info disdat           # expect total_species == 226
sdmbench reproduce tabpfn-sdm-2026 --max-species 3 --models maxnet,random-forest
```

Check specifically:
- 226 species across 6 regions
- AWT tasks have exactly 8 predictors
- Test data comes from the species' **own group** (AWT/NSW have multiple)
- The spatial scenario excludes ~18% of species (~41 of 226)

---

## 2. High priority — make the reproduction claim

### 2.1 Full reproduction run

```bash
sdmbench reproduce tabpfn-sdm-2026            # 226 species x 2 scenarios
sdmbench report tabpfn-sdm-2026 --show-extra
```

Targets (means across species; **reference only, never merge into results**):

| Model | Non-spatial | Spatial |
|---|---|---|
| Finetuned TabPFN | 0.762 | 0.699 |
| MaxNet | 0.732 | 0.656–0.683 |
| Random Forest | 0.727 | " |
| BRT | 0.724 | " |
| GAM | 0.717 | " |

Calibration slope for finetuned TabPFN: 1.110.

If a baseline is `MISMATCH`, investigate in this order: predictor subset →
preprocessing order → weighting → the R model's own settings.

### 2.2 Execute the TabPFN path

Never run. Requires `pip install -e ".[tabpfn]"` and accepting the Prior Labs
License v1.1 on the model page.

The loading procedure follows the model card exactly:
`_initialize_model_variables()` → `models_[0].load_state_dict(...)`. **If the
installed `tabpfn` exposes a different internal layout, the adapter raises
rather than loading partial weights** — that is deliberate; do not "fix" it by
guessing an alternative path, because partially loaded weights produce plausible
but wrong results.

Note the version tension: the paper used `tabpfn` **2.5**; the checkpoints
declare base model `Prior-Labs/TabPFN-v2-clf` and the card's snippet targets the
v2 API. Resolve empirically and record what worked.

Verify the checkpoint sha256 matches the values in `CHECKPOINTS`:
`e08fc4aa...` (non-spatial), `eafaf070...` (spatial).

### 2.3 Poll for the upstream repository

```bash
sdmbench upstream check tabpfn-sdm-2026
```

The moment it becomes public, `sdmbench upstream fetch tabpfn-sdm-2026` and
resolve every `UNVERIFIED` constant. Priority files: `R/run_tabpfn_finetuned.R`
(settles the ensemble scheme), `_targets.R` (the authoritative pipeline order),
`R/metrics.R`, `R/spatial.R`.

---

## 3. Medium priority — the extended benchmark (Mode B)

### 3.1 Run the extended comparison

```bash
sdmbench benchmark --config configs/extended_benchmark.yaml
sdmbench leaderboard --scenario nonspatial --common-species
sdmbench compare tabpfn-sdm knn --metric roc_auc --scenario nonspatial
sdmbench compare --reference tabpfn-sdm --all-pairs --correction holm
```

**The KNN framing matters.** It is included because it performed well in an
earlier leopard analysis — that is a reason to *test* it here, not evidence it
is good here. Do not describe it as state-of-the-art. If it loses, report that.

Use `--common-species` for the like-for-like table, and require the paired test
to agree with the bootstrap CI before claiming any difference.

### 3.2 Execute the graph and neural paths

Never run. `pip install -e ".[deep,graph]"`.

Watch for: `inductive` mode building a fresh test subgraph (verify the leakage
audit passes), early stopping on **training** loss only, and coordinates staying
out of node features.

### 3.3 Execute the AutoML adapters

Never run. Verify each respects its `time_limit_seconds` and that the budget is
recorded in the result row — an AutoML number without its budget is
uninterpretable.

---

## 4. The v2 research programme

The architecture is built; the science is not yet run.

### 4.1 Wire one real raster provider

**Recommended: Copernicus DEM.** It unlocks the most interesting untested axis
(multi-scale terrain) and is a clean public download with no credentials.

Implement `DataProvider` (see `docs/adding_a_dataset.md`), then verify the whole
raster path end to end — extraction, alignment checking, patch construction,
`multiscale-cnn`, and `plot_compute_frontier`.

`sdmbench.data.rasters` is written against the documented rasterio/pyproj APIs
but **has never been executed**. Expect real bugs.

### 4.2 Run the staged ablation

```bash
sdmbench benchmark --config configs/examples/staged_ablation.yaml
```

Stage 1 alone (information ablation) is publishable: *"which ecological
modalities actually improve geographically independent SDM prediction?"* is an
open question, and the leave-one-out vs alone contrast distinguishes redundancy
from irrelevance.

**Report the caveats the plan prints:** greedy selection can miss interactions,
and stage winners are optimistically biased. Final numbers from a clean run;
superiority claims from paired tests only.

### 4.3 Verify DeepMaxent, or stop calling it DeepMaxent

Currently implemented from a published *description*, tagged
`maturity="experimental"`, `verified_against_upstream=False`.

1. Locate the paper and reference implementation (**not yet found** — record it
   in `references/papers.yaml` when you do).
2. Compare **predicted intensities**, not just ROC-AUC — rank metrics can agree
   while the fitted intensity differs by a constant factor.
3. Retag on success; fix and document on failure.

Same for **GNN-SDM**: `PatchAdjacencyBuilder` substitutes k-means for the
published Self-Organizing Map patching step. That substitution is recorded in
`graph.metadata["patching"]` and must not be presented as a reproduction.

### 4.4 Compute AlphaEarth embeddings once

Needs *your* Earth Engine account. Compute for all `disdat` coordinates, cache
as NPZ, then use the `cached` provider — the only executable route for a
226-species run.

The question it answers is sharp: **does a general-purpose EO embedding carry
more ecological signal than twenty years of bioclim variables?**

---

## 5. Lower priority

- **Leopard recipe** — structure implemented, data loader is not. Refactor the
  existing scripts *into* the recipe; do not copy them in. Add published
  reference values before any reproduction claim.
- **Parallel execution** — the runner is sequential. Jobs are independent and
  checkpointed, so process-level parallelism is straightforward. Note R models
  parallelise poorly across processes.
- **SLURM backend** — `sdmbench submit config.yaml --backend slurm`. Not started.
- **Meta-learning recommender** — schema designed, nothing implemented. Do not
  ship recommendations until there are enough completed benchmarks to support
  them.
- **`sdmbench plan`** — the planner and DAG exist but have no CLI command.
- **`sdmbench explain`** — the module exists, no CLI command.

---

## 6. Standing rules — do not regress these

1. **Never report unexecuted tests as passing.** Say "written but not executed".
2. **Never let `published_reference` leak into `reproduced_result`.** A value
   not measured is `NOT RUN`.
3. **Coordinates are not predictors** without an explicit opt-in.
4. **Fitted preprocessing sees training rows only.** Encoders, PCA,
   autoencoders, scalers — all of them.
5. **Never bypass a licence or authentication gate.** Report and skip.
6. **Never reimplement a learning algorithm.** sdmbench owns orchestration,
   splitting, leakage protection, aggregation, reporting.
7. **Never silently substitute an implementation.** `random-forest` ≠
   `random-forest-sklearn`. Substitutions get a different name and a recorded
   note.
8. **Tag honestly.** If a constant cannot be verified, `Provenance.UNVERIFIED`.
   `sdmbench provenance` exits 3 while gaps remain — keep it that way.
9. **Aggregate across species, never the best one.** Report SD, SE, CI, n.
10. **A larger mean is not superiority.** Paired test + bootstrap CI must agree.

---

## 7. Suggested first session

```bash
# 1. Environment
pip install -e ".[dev,boosting]"
Rscript scripts/setup_r_packages.R
sdmbench env check

# 2. Run the tests that have never been run
pytest -m "not integration and not r and not gpu and not network" -v
# fix failures; do not delete tests to go green

# 3. Cross-language parity — the first real correctness evidence
pytest -m r -v

# 4. Real data
sdmbench data fetch disdat
sdmbench data info disdat            # expect 226 species

# 5. Smoke test
sdmbench reproduce tabpfn-sdm-2026 --max-species 3 --models maxnet,random-forest
sdmbench report tabpfn-sdm-2026

# 6. Then, in parallel:
#    - read Valavi et al. 2022, fix the GAM formulas (section 1.1)
#    - sdmbench upstream check tabpfn-sdm-2026   (poll for publication)
```

**Deliverable for that session:** a green unit suite, a passing spatial-filter
parity test against `sf`, and a 3-species smoke reproduction — after which the
full run is a matter of compute rather than correctness.
