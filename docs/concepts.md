# Concepts

The vocabulary sdmbench is built on, and why each distinction earns its place.

---

## Dataset ≠ Benchmark ≠ Methodology ≠ Model

### Dataset

The observations and covariates. `disdat` is a dataset: 226 species, six
regions, presence-only training records, background samples, and independent
presence-absence surveys.

A dataset does not tell you how to split it, what to preprocess, or how to
score a prediction.

### Benchmark

A **fixed experimental design taken from a published paper**:

```
paper + dataset + preprocessing + split protocol + models + evaluation
```

`tabpfn-sdm-2026` is a benchmark. It says: these 226 species, these predictor
subsets, this three-step recipe fitted on training data only, this 10 km
buffer, these three metrics, this seed.

This is the object that makes comparison meaningful. Two models evaluated under
one benchmark are comparable; two models each evaluated under their own
favourable conditions are not — and most published SDM comparisons are the
second kind.

### Methodology

The full modelling approach:

```
modalities + scales + representations + fusion
    + spatial structure + temporal structure + objective + predictor
```

*"Encode climate as raw tabular values and terrain with a multi-scale CNN at
1/5/25 km, fuse by cross-attention, build a geographic k-NN graph, and train a
Graph Transformer under a Poisson point-process objective"* is a methodology.

### Model / algorithm

The learner: Random Forest, XGBoost, GCN, TabPFN. One field inside a
methodology.

**The confusion this prevents:** "Random Forest beats deep learning for SDM" is
a claim about algorithms that is almost always really a claim about
methodologies — the deep model was given the same flat table of point-sampled
bioclim variables, which discards exactly the structure it could have exploited.
Separating the two lets you test which it was.

---

## Modality

One *kind* of ecological information: climate, terrain, soil, vegetation, land
cover, remote sensing, hydrology, human pressure, species traits, community
context, or a learned geospatial embedding.

The tempting shortcut is to concatenate everything into one wide DataFrame.
That discards four things the central research question needs:

- **Semantics.** Climate at a point and a 25 km terrain patch are different
  kinds of evidence and belong in different encoders.
- **Scale.** A modality has a resolution and a support (point vs patch). Lose it
  and "at what scale does terrain matter?" becomes unanswerable.
- **Provenance.** Source, licence, CRS, download time, checksum — without them
  the run is not reproducible.
- **Ablation.** *"Does terrain add anything beyond climate?"* is only
  well-posed if terrain is a nameable, removable unit.

So `ModalityData` keeps each source typed and separate all the way to fusion,
and `ModalityMetadata` carries its provenance.

### Structural kinds

What an encoder dispatches on:

| Kind | Shape | Example |
|---|---|---|
| `TABULAR` | (n, f) | bioclim variables at points |
| `RASTER` | (n, c, h, w) or (n, s, c, h, w) | terrain patches, multi-scale |
| `SEQUENCE` | (n, t, f) | seasonal NDVI, palaeoclimate |
| `EMBEDDING` | (n, d) | AlphaEarth vectors |
| `CATEGORICAL` | (n, f) | land cover classes |
| `GRAPH` | nodes + edges | landscape patches |

---

## Representation

How a modality becomes machine-readable features. The same information can be
represented many ways, and *which* is an empirical question:

```
raw values → PCA → autoencoder → multi-scale CNN → foundation embedding
```

Separating representation from model is what lets the benchmark ask *"is climate
better as raw values or as a contrastive embedding?"* while holding the
classifier fixed.

### The leakage contract

Encoders come in three kinds, and the difference is a leakage question:

| Kind | `fitted_on` | Can it leak? |
|---|---|---|
| Stateless (raw, patch statistics) | `none` | No — nothing is estimated |
| Fitted (PCA, autoencoder, standardisation) | `train` | **Yes**, if fitted on pooled data |
| Pretrained (frozen CNN, foundation model) | `pretrained` | No — fitted on external data |

A fitted encoder's parameters **must** come from training rows only. Fitting
PCA on the pooled train+test matrix leaks the test distribution into the basis,
and the downstream classifier will look perfectly well-behaved while doing it.
The auditor checks `fitted_on` rather than trusting it.

---

## Scale

Spatial scale is a first-class benchmark dimension, not an implementation
detail.

A species may respond to topography at 1 km (does this slope drain?), at 5 km
(is this a valley system?) and at 25 km (is this a mountain range?) — at once. A
point sample of elevation answers none of those.

```
location
   │
1 km patch  → encoder ──┐
5 km patch  → encoder ──┼→ fusion → vector
25 km patch → encoder ──┘
```

When the scales are combined by attention, the learned weights say *which scale
this species actually responds to*. That is an ecological result, not an
activation.

---

## Fusion

How encoded modalities are combined. Genuinely a research question:

| Strategy | Idea | Trade-off |
|---|---|---|
| `concat` | stack the vectors | baseline; surprisingly hard to beat |
| `early` | standardise per modality, then stack | stops a 64-d embedding swamping 5-d climate |
| `late` | one model per modality, combine probabilities | robust; **cannot** model interactions |
| `weighted` | one learned scalar per modality | interpretable |
| `gated` | weights **per observation** | terrain in mountains, climate on plains |
| `cross-attention` | modalities as tokens, attend across | most expressive, most data-hungry |
| `moe` | experts + learned router | specialisation |

Every strategy must tolerate an ablated modality, so `dataset.without("terrain")`
runs without special-casing.

---

## Objective

**Background points are not absences.**

Almost every ML-based SDM treats presence-only data as binary classification.
But background points are samples of *available environment*, and some are
certainly suitable habitat where the species simply was not recorded.

Point-process objectives take that seriously: background points are quadrature
points approximating the integral of an intensity function, not negative
examples. This is the established equivalence — MaxEnt with default settings
*is* a fitted inhomogeneous Poisson process (Renner & Warton 2013).

| Family | Objectives | Background points are... |
|---|---|---|
| Classification | `bce`, `weighted-bce`, `focal`, `pairwise-ranking` | negative examples |
| Point process | `maxent`, `poisson`, `deepmaxent` | quadrature points |

**Consequence for metrics:** a point-process model outputs an *intensity*, not
a probability. ROC-AUC and PR-AUC are rank-based and stay comparable; Miller's
calibration slope does not mean the same thing. Each objective declares this and
the results record it.

---

## Validation

What "independent" means, precisely.

**Non-spatial** — train on presence-only + background, test on an independent
presence-absence survey. Already stronger than most published evaluations,
because the test data comes from a different data collection. But test sites may
sit near training records, so it measures *interpolation*.

**Spatial** — additionally exclude training points within 10 km of any test
location. Measures *transferability* to novel geography, which is what
conservation actually needs.

The gap between the two is the honest measure of how much a same-distribution
estimate flatters a model.

### The invariant

The independent evaluation data must never influence:

- feature selection
- model selection
- hyperparameter tuning
- early stopping
- scaling or any fitted preprocessing
- representation learning
- graph construction involving training nodes
- calibration fitting

This is checked, not assumed — see [`validation.md`](validation.md).

---

## Why model choice comes last

The staged benchmark runs algorithm comparison in **stage 7**, after
information, scale, representation, fusion, spatial structure and objective.

The conventional order — fix the data, sweep the classifiers — answers the least
interesting question first, and conditions every result on an arbitrary
representation that nobody tested. If terrain matters at 5 km and you only ever
supplied point elevation, no classifier in your sweep could have used it, and
your conclusion is about your feature table rather than about the algorithms.
