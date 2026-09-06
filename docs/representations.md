# Representations, scales and fusion

How ecological information becomes machine-readable features — and why that is
a research question rather than preprocessing.

---

## The registry

```bash
sdmbench models          # model adapters
python -c "from sdmbench.representations import REPRESENTATIONS; print(REPRESENTATIONS.names())"
```

| Encoder | Accepts | Trainable | Notes |
|---|---|---|---|
| `raw-tabular` | tabular | no | the honest baseline |
| `standardized-tabular` | tabular | yes | z-score, fitted on train only |
| `categorical` | categorical | yes | one-hot on training levels |
| `pca` | tabular, embedding | yes | linear reduction |
| `ica` | tabular, embedding | yes | independent components |
| `autoencoder` | tabular | yes | denoising bottleneck |
| `vae` | tabular | yes | posterior mean (deterministic) |
| `contrastive` | tabular | yes | SimCLR-style, noise augmentation |
| `masked-feature` | tabular | yes | BERT-style feature masking |
| `patch-statistics` | raster | no | mean/SD/min/max/median per channel |
| `cnn` | raster | yes | small conv autoencoder |
| `multiscale-cnn` | raster | yes | per-scale + attention fusion |
| `vision-backbone` | raster | no (frozen) | torchvision ImageNet weights |
| `temporal-summary` | sequence | no | incl. a linear trend term |
| `temporal-lstm` | sequence | yes | recurrent |
| `temporal-cnn` | sequence | yes | dilated 1-D conv |
| `temporal-transformer` | sequence | yes | attention over time |
| `pretrained-embedding` | embedding | no | pass a foundation vector through |

---

## Why the baselines are included

`raw-tabular`, `patch-statistics` and `temporal-summary` are deliberately in the
registry, and the benchmark should run them.

If a ResNet does not beat five summary statistics per channel, **that is the
finding.** A framework that only offers the sophisticated option cannot
discover it.

`temporal-summary` includes a linear trend term, because "is climate here
changing?" is often the ecologically relevant question and a plain mean discards
it. A test asserts it recovers a known trend exactly.

---

## The leakage contract

| Kind | `fitted_on` | Leakage risk |
|---|---|---|
| Stateless | `none` | none — nothing estimated |
| Fitted | `train` | **real**, if fitted on pooled data |
| Pretrained | `pretrained` | none — fitted externally |

`BaseEncoder.fit()` is the only method allowed to see training data;
`transform()` must be a pure function of the fitted state. Calling `transform`
before `fit` on a trainable encoder raises.

A test asserts that after fitting a standardiser on training data, a *shifted*
test set does **not** come out centred at zero — the fingerprint of a pooled fit.

---

## Scale

```yaml
terrain:
  representation: multiscale-cnn
  scales: [1km, 5km, 25km]
  options:
    shared_encoder: true
    scale_fusion: attention
```

```
location
   │
1 km patch  → encoder ──┐
5 km patch  → encoder ──┼→ fusion → vector
25 km patch → encoder ──┘
```

**Shared vs separate encoders.** Shared means fewer parameters and
scale-invariant features; separate lets each scale specialise. Both available.

**Scale fusion.**

| Mode | Behaviour |
|---|---|
| `concat` | keeps scales separable — best for ablation and attribution |
| `mean` / `max` | pooling |
| `attention` | softmax over per-scale embedding norms |

Attention is the interesting one: the weights say *which scale this species
responds to*. That is an ecological result. It is parameter-free and
deterministic here, so it needs no extra training pass.

`plan_scale` runs each single scale (isolating the response) **and** the
multi-scale combination (testing whether combining beats the best single one).

---

## Fusion

| Strategy | Idea | Reports importance? |
|---|---|---|
| `concat` | stack vectors | no |
| `early` | standardise per modality, then stack | no |
| `late` | per-modality models, combine probabilities | yes (AUC weights) |
| `weighted` | one learned scalar per modality | yes |
| `gated` | weights **per observation** | yes (mean gate) |
| `cross-attention` | modalities as tokens | attention weights |
| `moe` | experts + router | router weights |

**`early` exists for a concrete reason:** without per-modality standardisation,
a 64-dimensional embedding with values in the hundreds dominates a
5-dimensional standardised climate block purely by scale. A test asserts this.

**`late` cannot model interactions** by construction — each modality gets its
own predictor. That is a limitation, and comparing it against `cross-attention`
is precisely how you measure whether cross-modal interactions matter.

**On attention as explanation.** Attention weights are reported as
`attention_weights`, never as `importance`. Attention indicates what the model
attended to; it is not by itself an explanation. For causal-ish claims use
ablation (see [`explainability.md`](explainability.md)).

Every strategy tolerates an ablated modality, so `dataset.without("terrain")`
runs unmodified. The neural strategies refuse rather than silently dropping an
input they were fitted with — an ablation must be re-fitted, not faked.

---

## Foundation embeddings

Adapters for models that turn a *location* into a vector. The question they
exist to answer:

> Does a general-purpose Earth-observation embedding carry more ecological
> signal than the bioclim variables ecologists have used for twenty years?

| Provider | Access | Status |
|---|---|---|
| `alphaearth` | Google Earth Engine (**your** account) | scaffold, never run |
| `terramind` | `terratorch` / Hugging Face | scaffold, never run |
| `prithvi` | `transformers` | scaffold, never run |
| `cached` | a local file | **works today** |

All are **frozen feature extractors**: weights are pretrained externally, so
nothing is fitted on benchmark data and no leakage is possible from them. What
they do require is provenance — model id, revision, embedding dimension, and the
date computed — because an embedding is only reproducible if the exact model
version is recorded.

### Credentials

sdmbench **never** embeds, requests or stores credentials, and never bypasses an
authentication gate. AlphaEarth detects an authenticated Earth Engine client and
otherwise raises `LicenseError` with setup instructions — which becomes a
`SKIPPED_LICENSE` row.

### The practical path

For a 226-species benchmark, compute embeddings **once**, cache them, and load
them as a precomputed modality:

```yaml
earth_observation:
  provider: cached
  representation: pretrained-embedding
  options:
    path: embeddings/alphaearth_disdat.npz
```

The file must be row-aligned with the dataset. sdmbench checks the row count and
records the assumption in the manifest; it cannot verify the correspondence
itself.

### The ImageNet caveat

`vision-backbone` applies torchvision ImageNet weights to environmental
rasters. Those weights were learned on 3-channel photographs; environmental
rasters are neither. The adapter maps channels by selection or repetition, and
this is a **real domain gap**.

It is included so the gap can be *measured* against a purpose-built alternative,
not because it is expected to win. The encoded metadata carries a
`domain_gap_note`.

---

## Adding an encoder

```python
from sdmbench.core.modality import ModalityKind
from sdmbench.representations.base import (
    REPRESENTATIONS, BaseEncoder, EncodedModality, encoder_spec,
)

@REPRESENTATIONS.register(
    "my-encoder",
    spec=encoder_spec(
        accepts=("tabular",),
        description="What it does, in one line.",
        requires=("torch",), extra="deep", trainable=True,
    ),
)
class MyEncoder(BaseEncoder):
    accepts = (ModalityKind.TABULAR,)
    trainable = True          # if True, _fit MUST only see training rows

    def _fit(self, modality, *, y=None):
        ...

    def _transform(self, modality) -> EncodedModality:
        return EncodedModality(name=modality.name, values=..., feature_names=[...])
```

Declare `accepts` honestly — `Methodology.validate()` uses it to reject
incompatible combinations *before* compute is spent, and a wrong declaration
turns a fast config error into a slow runtime one.
