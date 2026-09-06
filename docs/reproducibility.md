# Reproducibility

The standard sdmbench holds itself to: a number produced today should be
interpretable, and re-derivable, in five years.

---

## `run_manifest.json`

Written when a run starts and updated when it finishes.

```json
{
  "run_id": "tabpfn-sdm-2026-20260906T120000Z-a1b2c3d4",
  "benchmark_id": "tabpfn-sdm-2026",
  "benchmark_version": "1",
  "sdmbench_version": "0.1.0",
  "config": { ... },
  "config_hash": "sha256:...",
  "seeds": {"global": 32639},
  "dataset_hashes": {"primary": "sha256:..."},
  "model_checkpoints": {
    "tabpfn-sdm": {
      "repo_id": "rdinnager/tabpfn-sdm-finetuned",
      "filename": "tabpfn-sdm-nonspatial.pt",
      "revision": "a1b2c3...",
      "sha256": "e08fc4aa...",
      "hash_matches": true,
      "license": "Prior Labs License v1.1"
    }
  },
  "upstream": {"repo_url": "...", "commit": "...", "status": "not_public"},
  "provenance": { ... },
  "environment": {
    "platform": "...", "python_version": "3.11.8",
    "packages": {"numpy": "2.1.3", "tabpfn": "not-installed", ...},
    "gpu": {"cuda_available": true, "cuda_version": "12.1", "devices": [...]},
    "r": {"available": true, "version": "4.3.2"},
    "git_commit": "..."
  },
  "notes": ["4 protocol constant(s) are UNVERIFIED: [...]"]
}
```

The `notes` field matters: it is where a run records that the upstream
repository could not be fetched, or that the recipe contains unverified
constants, so the caveat travels **with the results** rather than living only in
documentation.

---

## Three hashes per result row

| Hash | Answers |
|---|---|
| `dataset_hash` | was this the same data? |
| `split_hash` | was this the same train/test partition? |
| `model_config_hash` | were these the same settings? |

Together they answer the question a reviewer will ask: *was this number computed
under the same conditions as that one?*

All are content-based and order-stable, so two runs on the same inputs produce
identical hashes on any machine and any Python version. The DataFrame hasher
goes through a canonical serialisation rather than `pandas.util.hash_pandas_object`,
whose output has changed across releases, and rounds floats to 12 significant
digits so last-bit noise from different BLAS builds does not change the hash.

`split_hash` describes **membership**, not the incidental order rows were
emitted in — so `hash_split([1,2,3], [4,5]) == hash_split([3,2,1], [5,4])`.

---

## Seeds

The published seed is **32639** (Dinnage & Warren 2026 sec. 2.2.1; confirmed by
the model card `config.json`). The species-split seed is **12345** (model card
only).

Ensemble member *i* uses `seed + i`, so members differ but the whole ensemble is
reproducible. A test asserts that two ensembles built with the same seed produce
identical member indices.

---

## Resumability

A full run is `226 species × N models × 2 scenarios` — thousands of jobs, many
slow. Every job writes its own JSON the moment it finishes:

```
<run_dir>/
    run_manifest.json
    results/<scenario>/<region>/<species>__<model>.json
    results.parquet          consolidated
    leakage/<job_id>.json    audit reports
```

Re-running the same command resumes: `completed_job_ids()` tells the runner what
exists and only missing jobs run. A completed successful job is never recomputed
unless `--force` is given.

Failed and skipped jobs are persisted too, so a resumed run does not retry a
model whose dependency is still missing — but they can be retried selectively.

Job ids are deterministic and filesystem-safe on Windows.

---

## Skips are recorded, not omitted

| Status | Meaning |
|---|---|
| `OK` | ran, metrics computed |
| `FAILED` | ran and errored; the message is in `status_detail` |
| `SKIPPED_DEPENDENCY` | an optional package is missing |
| `SKIPPED_NO_GPU` | a GPU-only model, no accelerator |
| `SKIPPED_LICENSE` | licence acceptance or authentication required |
| `SKIPPED_DATA` | single-class after filtering (the ~41 spatial exclusions) |

A benchmark that silently omitted every model it could not run would overstate
the coverage of its comparison. The leaderboard shows skip reasons per model.

---

## Upstream code: pinned, never vendored

sdmbench does not copy upstream source into this repository. Doing so would be a
licensing problem, a maintenance problem, and would hide which lines are the
authors' and which are ours.

```bash
sdmbench upstream check tabpfn-sdm-2026    # is it reachable?
sdmbench upstream fetch tabpfn-sdm-2026    # clone into cache, pin the SHA
sdmbench upstream info tabpfn-sdm-2026     # what was fetched, what it resolves
```

This gives attribution, provenance, and **two independent implementations to
cross-check**. Agreement between the authors' code and ours is far stronger
evidence of a correct reproduction than either alone.

---

## Provenance per constant

```bash
sdmbench provenance tabpfn-sdm-2026
```

```
  spatial_buffer_m        paper             Dinnage & Warren 2026 sec. 2.1.2, 2.5
  seed                    paper             Dinnage & Warren 2026 sec. 2.2.1
  region_crs              disdat            disdat R/disOther.R
! gam_region_formulas     unverified        the paper defers to Valavi et al. 2022
! upstream_repository     unverified        github.com/rdinnager/TabPFN-SDM -> HTTP 404

4 of 18 constants are UNVERIFIED (marked '!'). This recipe is an
approximation, not an exact reproduction.
```

Exits **3** while gaps remain, so CI can refuse to call an approximation a
reproduction.

---

## Published ≠ reproduced

Two separate namespaces, and nothing crosses between them:

```
published_reference   quoted from the paper. Never computed, never overwritten.
reproduced_result     what sdmbench measured in this run.
```

A model with no reproduced result reports `NOT RUN`. The published number is
never echoed back as though it had been measured. A test asserts this.

The `MATCH` / `CLOSE` / `MISMATCH` bands are documented, justified and printed
in every report — see
[`tabpfn_sdm_2026_protocol.md`](tabpfn_sdm_2026_protocol.md#10-reproduction-status-bands).

---

## Environment capture

`capture_environment()` records platform, Python, package versions (installed or
not), CUDA and GPU devices, R version, git commit (with a `-dirty` suffix when
the tree has uncommitted changes), and relevant environment variables.

It only observes. It installs nothing and changes nothing.

```bash
sdmbench env check
python scripts/check_environment.py --json
```

---

## What is not yet reproducible

Stated plainly:

- **Tests are written but have not been executed.**
- Cross-language parity tests (metrics vs `yardstick`, the spatial filter vs
  `sf`, Boyce vs `tidysdm`) are written and **not run**.
- The TabPFN, torch, graph, AutoML, raster and foundation adapters have **never
  been executed** — their backends were not installed.
- DeepMaxent and the GNN-SDM patch graph are implemented from published
  *descriptions*, not verified source, and are marked experimental.
- The leopard recipe's data loader is not implemented.
