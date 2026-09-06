# Validation and leakage

Every claim sdmbench makes rests on one property: **the independent
presence-absence survey data is used for evaluation only.**

This page explains what that means precisely, how it is enforced, and the
failure modes it prevents.

---

## The three partitions

Conflating these is the most common way to produce an invalid SDM benchmark.

| Partition | Purpose | May a model see it during fitting? |
|---|---|---|
| **Train** | fitting model parameters | yes |
| **Model-selection validation** | choosing hyperparameters, early stopping, checkpoint selection | yes — **but it must be carved out of the training data** |
| **Independent test** | the reported number | **never**, in any capacity |

The subtlety is the middle row. Tuning is legitimate and necessary — but the
validation set must come from the training partition. Tuning against the
independent survey and then reporting performance on it is not a benchmark, it
is a fitted curve.

`InnerCvTuner`'s signature makes this structural rather than advisory: it takes
`X_train`, `y_train` and `coords_train`, and there is no parameter through which
test data could be passed.

---

## What must never touch the test data

- feature selection
- model selection
- hyperparameter tuning
- early stopping
- scaling and any fitted preprocessing
- representation learning (PCA, autoencoders, contrastive encoders)
- graph construction involving training nodes
- calibration fitting

---

## Split protocols

### Non-spatial (TabPFN-SDM 2026)

No random splitting at all. Training data is presence-only records plus
background; test data is an independent survey. Two different data collections.

`identity_split` is a documented no-op so that the *absence* of a random split
is a recorded decision rather than an omission someone later "fixes".

### Spatial — the 10 km buffer

Exclude training points within 10 km of **any** test location.

- A one-sided filter on the **training** set. Test data is untouched.
- Not a spatial block CV. Not a spatially flavoured `train_test_split`.
- Species left single-class are excluded and *recorded as excluded*
  (`SKIPPED_DATA`), not silently dropped. This removes ~41 of 226.

**Distance is not a coordinate difference.** `disdat` regions differ: AWT, NZ
and SWI are projected in metres, so 10 km is a Euclidean radius of 10000. CAN,
NSW and SA are in degrees, where 10000 is meaningless.

`points_within_distance` dispatches on CRS exactly as `sf::st_is_within_distance`
does, and uses s2's earth radius (6,371,010 m) so the Python and R filters
select the same rows. `rbridge/scripts/spatial_split.R` is the reference for the
parity test.

### Spatial block CV

For the paper's sec. 2.7 holdout-vs-independent comparison. Points are binned
onto a grid, whole cells are dealt to folds, so nearby points share a fold.

> sdmbench's implementation is **not** bit-compatible with
> `spatialsample::spatial_block_cv`. Results computed with it are flagged.

---

## The leakage auditor

Rather than trusting the above by convention, every job is audited and every
result row carries `leakage_audit`.

| Check | Severity | Catches |
|---|---|---|
| `test_sites_absent_from_training` | fatal | the same site id in both partitions |
| `no_duplicate_coordinates_across_split` | warning | same location, different id |
| `spatial_buffer_respected` | fatal | a retained training point inside the buffer |
| `target_not_among_features` | fatal | the label copied into X under another name |
| `coordinates_not_used_as_features` | fatal | lon/lat as predictors without opt-in |
| `train_test_features_aligned` | fatal | mismatched column layouts |
| `preprocessing_fitted_on_training_only` | fatal | a recipe fitted on pooled data |
| `no_identical_rows_across_split` | warning | duplicated feature vectors |
| `graph_edges_cross_boundary` | fatal | message passing between train and test nodes |
| `hyperparameters_selected_without_test_labels` | fatal | tuning against the test set |

`FAIL` in strict mode aborts; otherwise it is recorded loudly so the number can
never be read as clean.

**These checks are tested by constructing deliberate leaks and asserting they
are caught.** A leakage detector that has never been shown a leak is not
evidence of anything.

---

## Failure modes, and why each matters

### Coordinates as features

The single most common way to get an impressive-looking SDM that has learned
nothing. Longitude and latitude let a model memorise *where surveys happened*
rather than which environments a species occupies. Interpolation scores rise;
transferability collapses — and transferability is what conservation needs.

sdmbench uses coordinates for splits and graph topology. Making them features
requires `include_coordinates=True`, and it is recorded in the results.

### Preprocessing fitted on pooled data

Fitting the scaler, PCA or autoencoder on train+test leaks the test
distribution into the representation. The downstream classifier then looks
perfectly well-behaved, which is what makes this so hard to spot after the fact.

The auditor has a fingerprint for it: a training-fitted scaler leaves the *test*
mean free to differ from zero. Perfect centring of both is the signature of a
leak.

### Graph edges crossing the boundary

In a transductive graph, an edge between a training node and a test node lets
test environment shape the fitted representation. sdmbench builds training and
test subgraphs **separately** by default (`graph_mode="inductive"`).
Transductive mode exists, must be requested explicitly, and is flagged.

### Early stopping on the test set

Subtle because it does not require the labels to enter the loss — monitoring
test performance to decide when to stop is still selection on the test set. The
neural and graph adapters early-stop on **training** loss, and the docstrings
say so.

### AutoML carving its own test set

AutoML systems normally want to own the whole experiment. That is unacceptable
here: a system that splits the benchmark's training data and reports performance
on its own slice is not comparable with anything.

The contract: the benchmark owns the train/test definition; AutoML may do
whatever internal selection it likes **on the training partition**; compute
budgets are recorded, because an AutoML number without its time limit is
uninterpretable.

---

## Aggregation rules

Enforcing separation at the job level is necessary but not sufficient — the
same failures reappear at the reporting level.

**Never rank on a single best species.** Aggregation is across species, always.

**Report dispersion.** Every leaderboard row carries SD, SE and a 95% CI. A
mean without them implies a precision the data does not have.

**Report coverage.** `n_species` and `n_regions` are always shown, and a warning
fires when models were evaluated on different species counts. Use
`--common-species` for the strictly like-for-like table.

**Superiority needs a paired test.** Because every model sees the same species,
comparisons are paired — which removes the enormous species-to-species variation
in difficulty. `compare` reports mean and median difference, bootstrap CI,
win/tie/loss, Wilcoxon signed-rank, effect size and a multiple-comparison
correction, and returns `NO_RELIABLE_DIFFERENCE` when that is the honest answer.

A 0.03 difference in mean ROC-AUC across 226 species may be highly consistent (a
model winning on 200) or noise (winning on 115). The mean alone cannot tell you
which.

---

## Reproducibility

Every run writes `run_manifest.json`: sdmbench and package versions, git commit,
OS, Python and R versions, CUDA and GPU, dataset and split hashes, model
checkpoint hashes, seeds, the full configuration, and any caveats recorded
during the run.

Result rows carry `dataset_hash`, `split_hash` and `model_config_hash`, which
together answer the question a reviewer will ask: *was this number computed on
the same data, with the same split, under the same settings as that one?*
