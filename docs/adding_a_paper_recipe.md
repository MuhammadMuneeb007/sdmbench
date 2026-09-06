# Adding a paper benchmark

A benchmark is a published paper's **fixed experimental design**:

```
paper + dataset + preprocessing + split protocol + models + evaluation
```

Adding one means writing a `PaperBenchmark` subclass. The run engine, metrics,
leaderboard, reporting and CLI need no changes.

---

## 1. Read the paper properly first

Not the abstract — the Methods section, and the released code and model card if
they exist. You need, at minimum:

- Which species/sites, and any exclusions
- Which predictors (often a **subset** of what the dataset offers)
- The preprocessing steps, **in order**, and what they were fitted on
- The exact split rule
- Which model implementations and settings
- Which metric definitions
- Seeds

Where the paper defers to another paper or to code, follow the chain. Where you
cannot, **tag it `UNVERIFIED`** rather than guessing.

> Worked example: [`tabpfn_sdm_2026_protocol.md`](tabpfn_sdm_2026_protocol.md)
> records every constant and its source, including the ones we could not confirm
> because the authors' repository is not public.

---

## 2. Write the recipe

```python
from sdmbench.benchmarks.base import (
    PaperBenchmark, PaperMetadata, PublishedResult, ReferenceValues,
)
from sdmbench.provenance import Provenance, ProvenanceReport, Sourced


class MyBenchmark(PaperBenchmark):
    benchmark_id = "author-2027"
    benchmark_version = "1"          # bump when a change alters numbers
    scenarios = ("nonspatial", "spatial")
    metrics = ("roc_auc", "boyce")
    default_models = ("maxnet", "random-forest")
    seed = 12345

    paper = PaperMetadata(
        paper_id="author_2027",
        title="...",
        authors=("A. Author",),
        year=2027,
        doi="10.xxxx/yyyy",
        code_url="https://github.com/...",
    )

    def iter_tasks(self):
        """Yield one SpeciesTask per species, with the PAPER's predictors."""
        for region in self.dataset.regions():
            for species in self.dataset.species(region):
                yield self.dataset.get_task(
                    region, species, predictors=MY_PREDICTORS[region]
                )

    def prepare(self, task, scenario):
        """Split rule FIRST, then preprocessing fitted on the result."""
        ...

    def dataset_hash(self):
        return self.dataset.content_hash()
```

Register it in `sdmbench/benchmarks/__init__.py`.

### Order matters in `prepare`

Apply the **split rule first**, then fit the preprocessing recipe on the
training set the split produced. Fitting the recipe before a spatial filter lets
excluded points influence the scaling parameters — a subtle leak that is
invisible in the results.

```python
if scenario == "spatial":
    task, info = apply_spatial_buffer(task, buffer_m=self.buffer_m)
    task.require_both_classes(where="training data after the spatial filter")
else:
    task, info = identity_split(task)

recipe = SdmRecipe(numeric, categorical)
X_train = recipe.fit_transform(task.train)   # train only
X_test = recipe.transform(task.test)
meta = recipe.describe()
meta["fitted_on"] = "train"                  # the auditor checks this
```

---

## 3. Record the published results

**Reference targets only.** They are stored, displayed and compared against —
never merged into a computed result.

```python
def published_results(self) -> ReferenceValues:
    refs = ReferenceValues()
    refs.add(PublishedResult(
        model="maxnet", scenario="nonspatial", metric="roc_auc",
        value=0.732, n_species=226, source="Author 2027, Table 2",
    ))
    refs.notes = [
        "Means across species, the paper's stated measure of central tendency.",
    ]
    return refs
```

Use `notes` for anything a reader could misread. The TabPFN recipe uses it to
warn that the model card's validation figures are **not** benchmark aggregates.

---

## 4. Record provenance

This is what separates a reproduction from an approximation that claims to be
one.

```python
def provenance(self) -> ProvenanceReport:
    report = ProvenanceReport(benchmark_id=self.benchmark_id)
    report.add("spatial_buffer_m",
               Sourced(10_000.0, Provenance.PAPER, "Author 2027 sec. 2.5"))
    report.add("seed",
               Sourced(12345, Provenance.PAPER, "Author 2027 sec. 2.2"))
    report.add("gam_formulas",
               Sourced("reconstructed", Provenance.UNVERIFIED,
                       "the paper defers to Smith 2020",
                       note="basis dimension k could not be confirmed"))
    return report
```

| Tag | Meaning |
|---|---|
| `PAPER` | stated in the text |
| `MODEL_CARD` | from a released model card / config |
| `DISDAT` | read from the dataset package source |
| `UPSTREAM_CODE` | read from the authors' code |
| `CITED_REFERENCE` | from a cited third-party paper |
| `UNVERIFIED` | **no public source confirms it** |

`sdmbench provenance <id>` exits **3** while any `UNVERIFIED` entries remain, so
CI can refuse to call an approximation a reproduction.

---

## 5. Register the upstream repository

```python
# sdmbench/upstream/metadata.py
"author-2027": UpstreamSource(
    benchmark_id="author-2027",
    repo_url="https://github.com/author/paper-code",
    files_of_interest={
        "R/models.R": "the exact model configurations",
        "R/metrics.R": "metric definitions",
    },
    resolves=("gam_formulas",),      # which UNVERIFIED entries it would fix
    availability="public",
    license="MIT",
),
```

sdmbench **never vendors** upstream code. It clones into the cache, pins the
commit SHA, and records it in every manifest — giving attribution plus two
independent implementations to cross-check.

---

## 6. Write the config and the docs

- `configs/<benchmark_id>.yaml` — annotated with the paper section each value
  comes from.
- `docs/<benchmark_id>_protocol.md` — the constant-by-constant record.
- `references/papers.yaml` — add the entry, with `verified:` set honestly.

---

## 7. Test the protocol, not just the plumbing

Pin the things that would change every number if broken:

```python
def test_predictor_counts_match_the_paper():
    assert len(REGION_PREDICTORS["AWT"]) == 8    # not disdat's 13

def test_published_values_match_the_paper():
    assert lookup[("maxnet", "nonspatial", "roc_auc")] == 0.732

def test_recipe_is_fitted_on_training_data_only():
    # a shifted test set must NOT come out centred at zero
    ...
```

---

## Checklist

- [ ] Methods section read; constants extracted with citations
- [ ] `iter_tasks` uses the paper's predictor subset, not the dataset default
- [ ] `prepare` applies the split **before** fitting preprocessing
- [ ] Excluded species raise `InsufficientDataError` (recorded, not dropped)
- [ ] `published_results` has a source per value
- [ ] `provenance` tags every constant, `UNVERIFIED` where honest
- [ ] Upstream repository registered
- [ ] Config, protocol doc and `references/papers.yaml` updated
- [ ] Tests pin the protocol constants
