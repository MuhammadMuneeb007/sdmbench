"""The TabPFN-SDM 2026 benchmark recipe.

Reproduces:

    Dinnage, Russell & Dan L. Warren (2026). "A Niche in the Machine: The
    Promise of AI Foundation Models for Species Distribution Modeling."
    EcoEvoRxiv. https://doi.org/10.32942/X2VQ10

Protocol, as encoded here
-------------------------
=================  ==========================================================
Data               ``disdat``: 226 species, 6 regions (sec. 2.1)
Predictors         The paper's per-region subsets (sec. 2.1.1) -- these are
                   *smaller* than ``disPredictors()`` returns for several
                   regions, so they are listed explicitly below
Categoricals       ``ontveg`` (CAN), ``vegsys`` (NSW), ``toxicats`` + ``age``
                   (NZ), ``calc`` (SWI) -- five variables, four regions
Preprocessing      zero-variance removal -> Yeo-Johnson -> normalise, fitted on
                   training data alone (sec. 2.1.2)
Training data      species presences + regional background sample
Test data          the independent presence-absence survey for the species'
                   own group -- never used for fitting anything
Non-spatial        all training data, no filtering (sec. 2.5), n = 226 species
Spatial            exclude training points within 10 km of any test location
                   (sec. 2.1.2, 2.5); species left single-class are dropped,
                   which removes ~41, leaving n = 185
Metrics            ROC-AUC, PR-AUC, Miller's calibration slope (sec. 2.6)
Seed               32639 (sec. 2.2.1)
=================  ==========================================================

Provenance status
-----------------
The paper's Methods section is unusually complete, so most constants here are
taken verbatim from it and are tagged ``PAPER``. Several are confirmed
independently by the Hugging Face model card's ``config.json`` (tagged
``MODEL_CARD``), notably the seed, the species-split fractions, the per-region
categorical variables and the 1,500-sample sub-batch cap.

What could **not** be verified: the authors' repository
(``github.com/rdinnager/TabPFN-SDM``, cited in sec. 2.8) returned HTTP 404 when
this recipe was written -- it is not public. Constants that depend on it are
tagged ``UNVERIFIED`` and listed by ``sdmbench report tabpfn-sdm-2026``. Run
``sdmbench upstream fetch tabpfn-sdm-2026`` once the repository appears to
re-check them.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from sdmbench.benchmarks.base import (
    PaperBenchmark,
    PaperMetadata,
    PublishedResult,
    ReferenceValues,
)
from sdmbench.config import RunConfig
from sdmbench.data.base import PreparedSplit, SpeciesTask
from sdmbench.data.disdat import CATEGORICAL_VARIABLES, REGIONS, DisdatDataset
from sdmbench.preprocessing import SdmRecipe
from sdmbench.provenance import Provenance, ProvenanceReport, Sourced
from sdmbench.splits.random import identity_split
from sdmbench.splits.spatial import DEFAULT_BUFFER_M, apply_spatial_buffer

__all__ = ["TabPFNSDM2026Benchmark", "REGION_PREDICTORS", "PUBLISHED_SEED"]

#: Seed used for every model in the study. Sec. 2.2.1: "All models used a
#: consistent random seed (32639) for reproducibility." Confirmed by the model
#: card's ``config.json`` (``"seed": 32639``).
PUBLISHED_SEED = 32639

#: Seed for the 62.5/7.5/30 species split used during finetuning. Only in the
#: model card (``"species_split_seed": 12345``); the paper says "a fixed random
#: seed" without giving it.
SPECIES_SPLIT_SEED = 12345

#: Per-region predictors, quoted from sec. 2.1.1.
#:
#: These are deliberately hard-coded rather than read from ``disPredictors()``,
#: because the paper uses a **subset**. ``disdat`` offers 13 predictors for AWT
#: and 13 for NSW; the paper uses 8 and 12 respectively (AWT drops bc01, bc17,
#: bc20, bc31, bc33; NSW drops tempmin). Using the full set would change every
#: number, so the published selection is authoritative here.
REGION_PREDICTORS: dict[str, tuple[str, ...]] = {
    "AWT": ("bc04", "bc05", "bc06", "bc12", "bc15", "slope", "topo", "tri"),
    "CAN": ("alt", "asp2", "ontprec", "ontslp", "onttemp", "ontveg", "watdist"),
    "NSW": (
        "cti", "disturb", "mi", "rainann", "raindq", "rugged", "soildepth",
        "soilfert", "solrad", "tempann", "topo", "vegsys",
    ),
    "NZ": (
        "age", "deficit", "hillshade", "mas", "mat", "r2pet", "slope", "sseas",
        "toxicats", "tseas", "vpd",
    ),
    "SA": ("sabio12", "sabio15", "sabio17", "sabio18", "sabio2", "sabio4", "sabio5", "sabio6"),
    "SWI": (
        "bcc", "calc", "ccc", "ddeg", "nutri", "pday", "precy", "sfroyy",
        "slope", "sradyy", "swb", "topo",
    ),
}

#: Species counts per scenario, from the Results section.
EXPECTED_N_SPECIES = {"nonspatial": 226, "spatial": 185}


class TabPFNSDM2026Benchmark(PaperBenchmark):
    """Strict reproduction of Dinnage & Warren (2026)."""

    benchmark_id = "tabpfn-sdm-2026"
    benchmark_version = "1"
    scenarios = ("nonspatial", "spatial")
    metrics = ("roc_auc", "pr_auc", "calibration_slope")
    seed = PUBLISHED_SEED

    #: The models the paper itself compared (sec. 2.2.1 and 2.2.2).
    default_models = (
        "maxnet",
        "random-forest",
        "brt",
        "gam",
        "tabpfn-default",
        "tabpfn-ss",
        "tabpfn-sdm",
    )

    paper = PaperMetadata(
        paper_id="dinnage_warren_2026",
        title=(
            "A Niche in the Machine: The Promise of AI Foundation Models for "
            "Species Distribution Modeling"
        ),
        authors=("Russell Dinnage", "Dan L. Warren"),
        year=2026,
        venue="EcoEvoRxiv",
        doi="10.32942/X2VQ10",
        code_url="https://github.com/rdinnager/TabPFN-SDM",
        data_source="disdat R package (Elith et al. 2020)",
    )

    def __init__(self, config: RunConfig | None = None, dataset: DisdatDataset | None = None):
        super().__init__(config)
        self.dataset = dataset or DisdatDataset()
        self.buffer_m = float(self.config.extra.get("spatial_buffer_m", DEFAULT_BUFFER_M))

    # ------------------------------------------------------------------ data --
    def regions(self) -> list[str]:
        """Regions to evaluate, honouring any restriction in the config."""
        available = self.dataset.regions()
        requested = self.config.evaluation.regions
        if not requested:
            return available
        wanted = {r.upper() for r in requested}
        unknown = wanted - set(REGIONS)
        if unknown:
            raise ValueError(f"unknown region(s): {sorted(unknown)}; expected {list(REGIONS)}")
        return [r for r in available if r in wanted]

    def iter_tasks(self) -> Iterator[SpeciesTask]:
        """Yield one task per species, with the paper's predictor subset."""
        limit = self.config.evaluation.max_species
        wanted_species = (
            {str(s) for s in self.config.evaluation.species}
            if self.config.evaluation.species
            else None
        )
        emitted = 0
        for region in self.regions():
            predictors = list(REGION_PREDICTORS[region])
            for species_id in self.dataset.species(region):
                if wanted_species is not None and species_id not in wanted_species:
                    continue
                if limit is not None and emitted >= limit:
                    return
                yield self.dataset.get_task(
                    region,
                    species_id,
                    predictors=predictors,
                    benchmark_id=self.benchmark_id,
                )
                emitted += 1

    def n_tasks(self) -> int:
        if self.config.evaluation.max_species:
            return int(self.config.evaluation.max_species)
        return sum(len(self.dataset.species(r)) for r in self.regions())

    def dataset_hash(self) -> str:
        return self.dataset.content_hash()

    # -------------------------------------------------------------- protocol --
    def prepare(self, task: SpeciesTask, scenario: str) -> PreparedSplit:
        """Apply the split rule, then the preprocessing recipe, in that order.

        Order matters: the recipe is fitted on the training set that the split
        rule produced. Fitting it before the spatial filter would let excluded
        points influence the scaling parameters -- a subtle leak that would be
        invisible in the results.
        """
        self.validate_scenario(scenario)

        if scenario == "spatial":
            task, split_info = apply_spatial_buffer(task, buffer_m=self.buffer_m)
            # Sec. 2.5: "Approximately 41 species were excluded from spatial
            # evaluation due to single-class datasets after filtering."
            task.require_both_classes(where="training data after the 10 km spatial filter")
        else:
            task, split_info = identity_split(task)
            task.require_both_classes()

        numeric = [p for p in task.predictors if p not in CATEGORICAL_VARIABLES]
        categorical = [p for p in task.predictors if p in CATEGORICAL_VARIABLES]

        recipe = SdmRecipe(
            numeric_predictors=numeric,
            categorical_predictors=categorical,
            yeo_johnson=True,
            normalize=True,
            remove_zero_variance=True,
        )
        # Fitted on training rows only -- the whole point of sec. 2.1.2.
        X_train = recipe.fit_transform(task.train)
        X_test = recipe.transform(task.test)

        preprocessing_meta = recipe.describe()
        preprocessing_meta["fitted_on"] = "train"

        return PreparedSplit(
            X_train=X_train,
            y_train=np.asarray(task.train["occ"], dtype=int),
            X_test=X_test,
            y_test=np.asarray(task.test["occ"], dtype=int),
            coords_train=task.train_coords(),
            coords_test=task.test_coords(),
            feature_names=recipe.output_columns,
            categorical_features=categorical,
            scenario=scenario,
            split_id=scenario,
            repeat_id=0,
            crs=task.crs,
            geographic=task.geographic,
            metadata={
                "region": task.region,
                "group": task.group,
                "species_id": task.species_id,
                "split": split_info,
                "preprocessing": preprocessing_meta,
                "benchmark_id": self.benchmark_id,
            },
        )

    def model_options(self, model_name: str, scenario: str) -> dict[str, Any]:
        """Scenario-specific model settings the protocol requires.

        Sec. 2.4.2: "Separate finetuned models were trained for non-spatial and
        spatial evaluation protocols." So the spatial scenario must use the
        spatial checkpoint; using the non-spatial one there would evaluate a
        model the paper never reported.
        """
        options: dict[str, Any] = {"seed": self.seed}
        if model_name in {"tabpfn-sdm", "tabpfn-finetuned"} and scenario == "spatial":
            options["variant"] = "spatial"
        return options

    # ------------------------------------------------------------ references --
    def published_results(self) -> ReferenceValues:
        """Values reported in the paper.

        Reference targets only. They are never merged into computed results.
        """
        refs = ReferenceValues()
        abstract = "Dinnage & Warren 2026, Abstract"
        results = "Dinnage & Warren 2026, Results"

        for model, value in (
            ("tabpfn-sdm", 0.762),
            ("maxnet", 0.732),
            ("random-forest", 0.727),
            ("brt", 0.724),
            ("gam", 0.717),
        ):
            refs.add(
                PublishedResult(
                    model=model,
                    scenario="nonspatial",
                    metric="roc_auc",
                    value=value,
                    n_species=226,
                    source=abstract,
                )
            )

        refs.add(
            PublishedResult(
                model="tabpfn-sdm",
                scenario="spatial",
                metric="roc_auc",
                value=0.699,
                n_species=185,
                source=abstract,
            )
        )
        refs.add(
            PublishedResult(
                model="tabpfn-sdm",
                scenario="nonspatial",
                metric="calibration_slope",
                value=1.110,
                n_species=226,
                source=abstract,
            )
        )

        refs.notes = [
            "All values are means across species, the paper's stated measure of central "
            "tendency (sec. 2.6).",
            "Under spatial separation the paper reports finetuned TabPFN at 0.699 mean "
            "ROC-AUC versus 0.656-0.683 for the traditional methods; per-model spatial "
            "figures are not individually itemised in the abstract, so only the TabPFN "
            "value is stored as a point reference.",
            "The abstract also reports 0.763 mean ROC-AUC on species held out entirely "
            "from finetuning, versus 0.762 overall -- evidence of generalisation rather "
            "than memorisation.",
            "IMPORTANT: the Hugging Face model card's validation figures (ROC-AUC 0.747 "
            "non-spatial, 0.653 spatial) are CHECKPOINT-VALIDATION metrics measured "
            "during finetuning on held-out species. They are not benchmark aggregates "
            "and must not be compared against these values.",
        ]
        return refs

    # ------------------------------------------------------------ provenance --
    def provenance(self) -> ProvenanceReport:
        report = ProvenanceReport(benchmark_id=self.benchmark_id)
        paper = "Dinnage & Warren 2026"

        report.add(
            "regions",
            Sourced(list(REGIONS), Provenance.PAPER, f"{paper} sec. 2.1"),
        )
        report.add(
            "region_predictors",
            Sourced(
                {k: list(v) for k, v in REGION_PREDICTORS.items()},
                Provenance.PAPER,
                f"{paper} sec. 2.1.1",
                note=(
                    "A documented subset of disPredictors(); AWT uses 8 of 13 and NSW 12 of 13."
                ),
            ),
        )
        report.add(
            "categorical_variables",
            Sourced(
                list(CATEGORICAL_VARIABLES),
                Provenance.PAPER,
                f"{paper} sec. 2.1.1; confirmed by the model card config.json",
            ),
        )
        report.add(
            "preprocessing_steps",
            Sourced(
                ["step_zv", "step_YeoJohnson", "step_normalize"],
                Provenance.PAPER,
                f"{paper} sec. 2.1.2",
            ),
        )
        report.add(
            "preprocessing_fitted_on",
            Sourced("train", Provenance.PAPER, f"{paper} sec. 2.1.2"),
        )
        report.add(
            "spatial_buffer_m",
            Sourced(DEFAULT_BUFFER_M, Provenance.PAPER, f"{paper} sec. 2.1.2, 2.5"),
        )
        report.add(
            "spatial_filter_target",
            Sourced(
                "training points only",
                Provenance.PAPER,
                f"{paper} sec. 2.5",
                note="Excludes training points near test locations; test data is untouched.",
            ),
        )
        report.add(
            "spatial_exclusion_rule",
            Sourced(
                "drop species left single-class after filtering",
                Provenance.PAPER,
                f"{paper} sec. 2.5 (~41 species excluded, n = 185)",
            ),
        )
        report.add(
            "metrics",
            Sourced(list(self.metrics), Provenance.PAPER, f"{paper} sec. 2.6"),
        )
        report.add(
            "calibration_definition",
            Sourced(
                "glm(observed ~ logit(predicted), binomial); slope on logit(p)",
                Provenance.PAPER,
                f"{paper} sec. 2.6",
                note="Miller's slope, not a generic calibration slope.",
            ),
        )
        report.add("seed", Sourced(PUBLISHED_SEED, Provenance.PAPER, f"{paper} sec. 2.2.1"))
        report.add(
            "ensemble_k",
            Sourced(16, Provenance.PAPER, f"{paper} sec. 2.3"),
        )
        report.add(
            "ensemble_scheme",
            Sourced(
                "balanced draw per member, overlap allowed, logits averaged",
                Provenance.PAPER,
                f"{paper} sec. 2.3",
                note=(
                    "The HF model card instead describes PARTITIONING the background across "
                    "sub-batches. The two differ; the paper's wording is used for the "
                    "published TabPFN-SS variant. See sdmbench.models.ensemble."
                ),
            ),
        )
        report.add(
            "max_train_size_per_subbatch",
            Sourced(
                1500,
                Provenance.MODEL_CARD,
                "model card config.json; also stated in the paper sec. 2.4.2",
            ),
        )
        report.add(
            "species_split_seed",
            Sourced(SPECIES_SPLIT_SEED, Provenance.MODEL_CARD, "model card config.json"),
        )
        report.add(
            "region_crs",
            Sourced(
                "per-region, from disCRS()",
                Provenance.DISDAT,
                "disdat R/disOther.R",
                note="AWT/NZ/SWI are projected in metres; CAN/NSW/SA are geographic degrees.",
            ),
        )

        # --- Constants that no public source pins down ------------------------
        report.add(
            "gam_region_formulas",
            Sourced(
                "reconstructed: s(x, bs='tp') per continuous predictor + factor terms",
                Provenance.UNVERIFIED,
                f"{paper} sec. 2.2.1 defers to Valavi et al. 2022 and the authors' repository",
                note=(
                    "Per-predictor basis dimension k could not be confirmed; the upstream "
                    "repository is not public. Pass formula= to override."
                ),
            ),
        )
        report.add(
            "random_forest_replace",
            Sourced(
                True,
                Provenance.UNVERIFIED,
                "randomForest default; the paper specifies sample size but not `replace`",
            ),
        )
        report.add(
            "pr_auc_estimator",
            Sourced(
                "average_precision (step-wise)",
                Provenance.UNVERIFIED,
                "the paper says yardstick, which integrates trapezoidally",
                note=(
                    "sklearn's step-wise estimator is used by default because trapezoidal "
                    "PR interpolation is optimistically biased. Set pr_auc method='trapezoid' "
                    "for closer yardstick parity."
                ),
            ),
        )
        report.add(
            "upstream_repository",
            Sourced(
                "unavailable",
                Provenance.UNVERIFIED,
                "github.com/rdinnager/TabPFN-SDM returned HTTP 404 (not public)",
                note=(
                    "Run `sdmbench upstream fetch tabpfn-sdm-2026` once it is published to "
                    "re-verify every UNVERIFIED constant above."
                ),
            ),
        )
        return report
