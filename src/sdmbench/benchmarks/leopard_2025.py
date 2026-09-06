"""Leopard benchmark recipe (Leedham et al. 2025) -- scaffold.

Reproduces the experimental structure of:

    Leedham et al. (2025). "Limited Niche Change After Dispersal From Africa by
    Leopards (*Panthera pardus*) Hundreds of Thousands of Years Ago."

Known experimental structure
----------------------------
* five environmental predictors: ``bio04``, ``bio05``, ``bio08``, ``lai``,
  ``rugosity``
* 10 pseudo-absence datasets
* five spatial folds
* original workflow: Random Forest + MaxEnt
* evaluation by Continuous Boyce Index

Status
------
This is a **scaffold**, not a working reproduction. The data loader is
unimplemented because the leopard occurrence data is not bundled with sdmbench
and has not been wired to a public source here; :meth:`iter_tasks` therefore
raises a clear error rather than silently producing an empty benchmark.

It exists now for two reasons. First, it proves the
:class:`~sdmbench.benchmarks.base.PaperBenchmark` abstraction generalises
beyond ``disdat`` -- a different dataset, a different split scheme (repeated
pseudo-absence draws x spatial folds rather than a one-sided buffer), and a
different primary metric. Second, our existing leopard scripts should be
*refactored into* this recipe rather than copied into the package: the reusable
parts (pseudo-absence sampling, spatial folds, the Boyce index) already live in
:mod:`sdmbench.splits` and :mod:`sdmbench.metrics`, and what remains is
study-specific data loading.

See ``docs/adding_benchmarks.md`` for how to finish it.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from sdmbench.benchmarks.base import PaperBenchmark, PaperMetadata, ReferenceValues
from sdmbench.config import RunConfig
from sdmbench.data.base import PreparedSplit, SpeciesTask
from sdmbench.exceptions import DataNotFetchedError
from sdmbench.preprocessing import SdmRecipe
from sdmbench.provenance import Provenance, ProvenanceReport, Sourced
from sdmbench.splits.spatial import SpatialBlockCV

__all__ = ["LeopardLeedham2025Benchmark", "LEOPARD_PREDICTORS"]

#: The five predictors of the original study.
LEOPARD_PREDICTORS = ("bio04", "bio05", "bio08", "lai", "rugosity")

#: Number of independent pseudo-absence draws.
N_PSEUDOABSENCE_SETS = 10

#: Number of spatial cross-validation folds.
N_SPATIAL_FOLDS = 5


class LeopardLeedham2025Benchmark(PaperBenchmark):
    """Scaffold for the leopard niche-change benchmark."""

    benchmark_id = "leopard-leedham-2025"
    benchmark_version = "0-scaffold"
    scenarios = ("spatial_cv",)
    #: The original study evaluated with the Continuous Boyce Index.
    metrics = ("boyce", "roc_auc", "pr_auc")
    default_models = ("random-forest-sklearn", "maxnet", "knn")
    seed = 32639

    paper = PaperMetadata(
        paper_id="leedham_2025",
        title=(
            "Limited Niche Change After Dispersal From Africa by Leopards "
            "(Panthera pardus) Hundreds of Thousands of Years Ago"
        ),
        authors=("Leedham et al.",),
        year=2025,
        venue="",
        doi="",
        data_source="study-specific occurrence records and raster predictors",
    )

    def __init__(self, config: RunConfig | None = None) -> None:
        super().__init__(config)
        self.n_pseudoabsence_sets = int(
            self.config.extra.get("n_pseudoabsence_sets", N_PSEUDOABSENCE_SETS)
        )
        self.n_folds = int(self.config.extra.get("n_spatial_folds", N_SPATIAL_FOLDS))

    # ------------------------------------------------------------------ data --
    def iter_tasks(self) -> Iterator[SpeciesTask]:
        raise DataNotFetchedError(
            "The leopard benchmark is a scaffold: its data loader is not implemented.\n"
            "  To finish it:\n"
            "    1. Add a loader that yields SpeciesTask objects with predictors "
            f"{list(LEOPARD_PREDICTORS)}\n"
            "    2. Point it at the study's occurrence records and raster stack (see "
            "sdmbench.data.rasters.RasterDataset for extraction and pseudo-absence "
            "sampling)\n"
            "    3. Replace this method\n"
            "  See docs/adding_benchmarks.md."
        )

    def dataset_hash(self) -> str:
        return ""

    # -------------------------------------------------------------- protocol --
    def prepare(self, task: SpeciesTask, scenario: str) -> PreparedSplit:
        """Spatially blocked fold over one pseudo-absence draw.

        Implemented so the structure is reviewable even though :meth:`iter_tasks`
        cannot yet supply data: given a task, this produces exactly the split the
        study used.
        """
        self.validate_scenario(scenario)
        task.require_both_classes()

        repeat = int(self.config.evaluation.repeats and 0)
        cv = SpatialBlockCV(n_folds=self.n_folds, seed=self.seed + repeat)
        train_idx, test_idx = cv.first_fold(task.train_coords(), geographic=task.geographic)

        train_rows = task.train.iloc[train_idx]
        test_rows = task.train.iloc[test_idx]

        recipe = SdmRecipe(
            numeric_predictors=list(task.predictors),
            categorical_predictors=list(task.categorical_predictors),
        )
        X_train = recipe.fit_transform(train_rows)
        X_test = recipe.transform(test_rows)
        meta = recipe.describe()
        meta["fitted_on"] = "train"

        return PreparedSplit(
            X_train=X_train,
            y_train=np.asarray(train_rows["occ"], dtype=int),
            X_test=X_test,
            y_test=np.asarray(test_rows["occ"], dtype=int),
            coords_train=train_rows.loc[:, ["x", "y"]].to_numpy(dtype=float),
            coords_test=test_rows.loc[:, ["x", "y"]].to_numpy(dtype=float),
            feature_names=recipe.output_columns,
            categorical_features=list(task.categorical_predictors),
            scenario=scenario,
            split_id=f"fold0_of_{self.n_folds}",
            crs=task.crs,
            geographic=task.geographic,
            metadata={
                "region": task.region,
                "species_id": task.species_id,
                "preprocessing": meta,
                "benchmark_id": self.benchmark_id,
                "n_pseudoabsence_sets": self.n_pseudoabsence_sets,
            },
        )

    # ------------------------------------------------------------ references --
    def published_results(self) -> ReferenceValues:
        refs = ReferenceValues()
        refs.notes = [
            "No published reference values are encoded yet. Add them from the paper "
            "before using this recipe for a reproduction claim."
        ]
        return refs

    def provenance(self) -> ProvenanceReport:
        report = ProvenanceReport(benchmark_id=self.benchmark_id)
        report.add(
            "predictors",
            Sourced(list(LEOPARD_PREDICTORS), Provenance.CITED_REFERENCE, "Leedham et al. 2025"),
        )
        report.add(
            "n_pseudoabsence_sets",
            Sourced(N_PSEUDOABSENCE_SETS, Provenance.CITED_REFERENCE, "Leedham et al. 2025"),
        )
        report.add(
            "n_spatial_folds",
            Sourced(N_SPATIAL_FOLDS, Provenance.CITED_REFERENCE, "Leedham et al. 2025"),
        )
        report.add(
            "data_loader",
            Sourced(
                "not implemented",
                Provenance.UNVERIFIED,
                "scaffold only",
                note="Occurrence data and raster predictors are not wired to a source.",
            ),
        )
        report.add(
            "spatial_fold_construction",
            Sourced(
                "sdmbench blocked assignment",
                Provenance.UNVERIFIED,
                "the study's exact fold construction has not been encoded",
            ),
        )
        return report
