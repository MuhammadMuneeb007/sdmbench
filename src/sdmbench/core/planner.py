"""The staged benchmark planner.

The combinatorial problem
-------------------------
With 8 modality subsets x 4 scales x 6 representations x 6 fusion strategies x
5 graph strategies x 6 objectives x 12 models, the full factorial design is
about 415,000 configurations -- per species. Across 226 species that is
absurd computationally, and worse statistically: at alpha = 0.05 you would
expect thousands of spurious "significant" results by chance alone.

The staged design
-----------------
Answer one question at a time, carry the winner forward, and only expand the
axis currently under test. Seven stages, each with a question:

1. **Information** -- which modalities help? (one reference model, one
   representation)
2. **Scale** -- at what spatial scale do the informative modalities help?
3. **Representation** -- how should that information be encoded?
4. **Fusion** -- how should the modalities interact?
5. **Spatial structure** -- do relationships between locations help?
6. **Objective** -- what should the model be asked to learn?
7. **Model** -- and only now, which algorithm?

Model choice comes *last* deliberately. The usual practice -- fix the data,
sweep the classifiers -- answers the least interesting question first and
conditions everything else on an arbitrary representation.

Two honest caveats, recorded in every plan
------------------------------------------
*Greedy search is not exhaustive.* Carrying one winner forward can miss an
interaction: a representation that only pays off under cross-attention will be
eliminated at stage 3 while fusion is still fixed at ``concat``. The planner
supports ``top_k`` to carry several candidates forward, which mitigates but does
not remove this.

*Sequential stages accumulate selection bias.* Each stage picks a maximum over
noisy estimates, so the carried-forward winner is optimistically biased. The
final reported number must come from a clean run of the selected methodology,
and any claim of superiority must rest on the paired tests in
:mod:`sdmbench.evaluation.compare`, not on stage-winner status.
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from sdmbench.core.methodology import Methodology, ModalitySpec
from sdmbench.exceptions import ConfigurationError

__all__ = ["Stage", "StagePlan", "BenchmarkPlan", "StagedPlanner"]


class Stage(str, enum.Enum):
    """The seven benchmark stages, in the order they must be run."""

    INFORMATION = "information"
    SCALE = "scale"
    REPRESENTATION = "representation"
    FUSION = "fusion"
    SPATIAL = "spatial"
    OBJECTIVE = "objective"
    MODEL = "model"

    @property
    def question(self) -> str:
        return {
            Stage.INFORMATION: "What information helps?",
            Stage.SCALE: "At what spatial scale?",
            Stage.REPRESENTATION: "How should information be represented?",
            Stage.FUSION: "How should modalities interact?",
            Stage.SPATIAL: "Do relationships between locations help?",
            Stage.OBJECTIVE: "What should the model learn?",
            Stage.MODEL: "Which algorithm, given all of the above?",
        }[self]

    @property
    def order(self) -> int:
        return list(Stage).index(self)


@dataclass
class StagePlan:
    """The methodologies to run for one stage."""

    stage: Stage
    methodologies: list[Methodology] = field(default_factory=list)
    #: The methodology this stage starts from (the previous stage's winner).
    baseline: Methodology | None = None
    #: What varies in this stage, for the report.
    varying: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def n_configurations(self) -> int:
        return len(self.methodologies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "question": self.stage.question,
            "varying": self.varying,
            "n_configurations": self.n_configurations,
            "baseline": self.baseline.name if self.baseline else None,
            "methodologies": [m.to_dict() for m in self.methodologies],
            "notes": self.notes,
        }

    def render(self) -> str:
        lines = [
            f"Stage {self.stage.order + 1}: {self.stage.value.upper()}",
            f"  Question: {self.stage.question}",
            f"  Varying:  {self.varying}",
            f"  Configurations: {self.n_configurations}",
        ]
        for methodology in self.methodologies:
            lines.append(f"    - {methodology.name}")
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


@dataclass
class BenchmarkPlan:
    """A full staged plan."""

    stages: list[StagePlan] = field(default_factory=list)
    reference_model: str = ""
    n_species: int | None = None
    n_scenarios: int = 1
    notes: list[str] = field(default_factory=list)

    @property
    def total_configurations(self) -> int:
        return sum(s.n_configurations for s in self.stages)

    def estimated_jobs(self) -> int:
        """Jobs = configurations x species x scenarios."""
        return self.total_configurations * (self.n_species or 1) * self.n_scenarios

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_model": self.reference_model,
            "n_species": self.n_species,
            "n_scenarios": self.n_scenarios,
            "total_configurations": self.total_configurations,
            "estimated_jobs": self.estimated_jobs(),
            "stages": [s.to_dict() for s in self.stages],
            "notes": self.notes,
        }

    def render(self) -> str:
        lines = ["STAGED BENCHMARK PLAN", "=" * 21, ""]
        if self.reference_model:
            lines.append(f"Reference model (stages 1-6): {self.reference_model}")
        if self.n_species:
            lines.append(f"Species: {self.n_species}   Scenarios: {self.n_scenarios}")
        lines.append(
            f"Total configurations: {self.total_configurations}   "
            f"Estimated jobs: {self.estimated_jobs():,}"
        )
        lines.append("")
        for stage in self.stages:
            lines.append(stage.render())
            lines.append("")
        if self.notes:
            lines.append("PLAN NOTES")
            lines.extend(f"  - {n}" for n in self.notes)
        return "\n".join(lines)


class StagedPlanner:
    """Builds a staged plan from the axes the user wants to explore.

    Parameters
    ----------
    base:
        The starting methodology. Stage 1 varies its modality set; each later
        stage varies one axis of the previous stage's winner.
    reference_model:
        The single model used for stages 1-6. It should be strong and fast --
        a gradient-boosted tree is the usual choice, so that methodology
        differences are not masked by a weak learner or drowned in GPU time.
    top_k:
        How many winners to carry forward from each stage. ``1`` is a pure
        greedy search; larger values reduce the risk of eliminating an option
        that only pays off in combination.
    """

    def __init__(
        self,
        base: Methodology,
        *,
        reference_model: str = "xgboost",
        top_k: int = 1,
        n_species: int | None = None,
        n_scenarios: int = 1,
    ) -> None:
        self.base = base
        self.reference_model = reference_model
        self.top_k = max(1, int(top_k))
        self.n_species = n_species
        self.n_scenarios = n_scenarios

    # ------------------------------------------------------------- stage 1 --
    def plan_information(
        self,
        modalities: Sequence[str],
        *,
        include_singles: bool = True,
        include_pairs: bool = False,
        include_leave_one_out: bool = True,
        include_all: bool = True,
    ) -> StagePlan:
        """Stage 1: which modalities carry information?

        Three complementary views, because they answer different questions:

        * **Singles** -- what can this modality do alone? (marginal value)
        * **Leave-one-out** -- what is lost by removing it from the full set?
          (unique contribution, after everything else)
        * **All** -- the ceiling.

        A modality can look strong alone and add nothing leave-one-out, which
        means it is redundant with the others. That distinction is the whole
        point of running both.
        """
        available = [m for m in modalities if self.base.modality(m) is not None]
        missing = [m for m in modalities if self.base.modality(m) is None]
        plan = StagePlan(
            stage=Stage.INFORMATION,
            baseline=self.base,
            varying="modality subset",
        )
        if missing:
            plan.notes.append(f"not present in the base methodology, skipped: {missing}")
        if not available:
            raise ConfigurationError("stage 1 needs at least one available modality")

        seen: set[str] = set()

        def add(methodology: Methodology) -> None:
            key = methodology.methodology_hash
            if key not in seen:
                seen.add(key)
                plan.methodologies.append(methodology)

        if include_all and len(available) > 1:
            add(
                self.base.with_modalities(available)._renamed("all-modalities")
            )
        if include_singles:
            for name in available:
                add(self.base.with_modalities([name])._renamed(f"only-{name}"))
        if include_pairs:
            for a, b in itertools.combinations(available, 2):
                add(self.base.with_modalities([a, b])._renamed(f"{a}+{b}"))
        if include_leave_one_out and len(available) > 2:
            for name in available:
                rest = [m for m in available if m != name]
                add(self.base.with_modalities(rest)._renamed(f"all-minus-{name}"))

        plan.methodologies = [
            m.with_model(self.reference_model)._renamed(m.name) for m in plan.methodologies
        ]
        plan.notes.append(
            "Singles measure marginal value; leave-one-out measures unique contribution. "
            "A modality strong alone but useless leave-one-out is redundant."
        )
        return plan

    # ------------------------------------------------------------- stage 2 --
    def plan_scale(
        self,
        winner: Methodology,
        *,
        scales: Sequence[str] = ("250m", "1km", "5km", "25km"),
        raster_modalities: Sequence[str] | None = None,
        include_multiscale: bool = True,
    ) -> StagePlan:
        """Stage 2: at what spatial scale does each raster modality help?"""
        from sdmbench.core.modality import ModalityKind, STANDARD_MODALITIES

        targets = list(raster_modalities) if raster_modalities else [
            spec.name
            for spec in winner.modalities
            if (definition := STANDARD_MODALITIES.get(spec.name))
            and definition.kind is ModalityKind.RASTER
        ]
        plan = StagePlan(stage=Stage.SCALE, baseline=winner, varying="spatial scale")
        if not targets:
            plan.notes.append(
                "no raster modalities in the winning methodology; scale does not apply "
                "and this stage is skipped"
            )
            return plan

        for modality in targets:
            for scale in scales:
                plan.methodologies.append(winner.with_scales(modality, [scale]))
            if include_multiscale:
                plan.methodologies.append(winner.with_scales(modality, scales))
        plan.notes.append(
            "Single scales isolate the response; the multi-scale variant tests whether "
            "combining them beats the best single scale."
        )
        return plan

    # ------------------------------------------------------------- stage 3 --
    def plan_representation(
        self,
        winner: Methodology,
        *,
        representations: dict[str, Sequence[str]] | None = None,
    ) -> StagePlan:
        """Stage 3: how should each modality be encoded?"""
        from sdmbench.core.modality import STANDARD_MODALITIES
        from sdmbench.representations import REPRESENTATIONS

        plan = StagePlan(
            stage=Stage.REPRESENTATION, baseline=winner, varying="representation encoder"
        )
        for spec in winner.modalities:
            if representations and spec.name in representations:
                candidates = list(representations[spec.name])
            else:
                definition = STANDARD_MODALITIES.get(spec.name)
                kind = spec.kind or (definition.kind.value if definition else "tabular")
                candidates = REPRESENTATIONS.accepting(kind)
            for encoder in candidates:
                if encoder == spec.representation:
                    continue
                plan.methodologies.append(winner.with_representation(spec.name, encoder))
        plan.notes.append(
            "The incumbent representation is not re-run here; its score carries over "
            "from the previous stage."
        )
        return plan

    # ------------------------------------------------------------- stage 4 --
    def plan_fusion(
        self,
        winner: Methodology,
        *,
        strategies: Sequence[str] = ("concat", "early", "late", "weighted", "gated", "cross-attention", "moe"),
    ) -> StagePlan:
        """Stage 4: how should the modalities interact?"""
        plan = StagePlan(stage=Stage.FUSION, baseline=winner, varying="fusion strategy")
        if len(winner.modalities) < 2:
            plan.notes.append(
                "the winning methodology has one modality; there is nothing to fuse "
                "and this stage is skipped"
            )
            return plan
        for strategy in strategies:
            if strategy == winner.fusion:
                continue
            plan.methodologies.append(winner.with_fusion(strategy))
        return plan

    # ------------------------------------------------------------- stage 5 --
    def plan_spatial(
        self,
        winner: Methodology,
        *,
        strategies: Sequence[str | None] = (None, "geographic-knn", "environmental-knn", "hybrid", "patch-adjacency"),
        graph_model: str = "graph-transformer",
    ) -> StagePlan:
        """Stage 5: does explicit spatial structure help?

        Graph strategies require a graph model, so this stage swaps the
        predictor as well -- and therefore confounds "graph helps" with
        "graph model helps". The no-graph arm uses the same graph model with an
        empty edge set where possible, so the comparison isolates topology; that
        caveat is recorded in the plan.
        """
        plan = StagePlan(
            stage=Stage.SPATIAL, baseline=winner, varying="graph construction strategy"
        )
        for strategy in strategies:
            if strategy is None:
                plan.methodologies.append(winner.with_graph(None)._renamed(f"{winner.name}-nograph"))
            else:
                plan.methodologies.append(
                    winner.with_graph(strategy).with_model(graph_model)
                )
        plan.notes.append(
            "CAVEAT: graph arms use a graph model and the no-graph arm does not, so this "
            "stage confounds topology with architecture. To isolate topology, compare the "
            "graph strategies against each other rather than against the no-graph arm."
        )
        return plan

    # ------------------------------------------------------------- stage 6 --
    def plan_objective(
        self,
        winner: Methodology,
        *,
        objectives: Sequence[str] = ("weighted-bce", "focal", "pairwise-ranking", "maxent", "poisson", "deepmaxent"),
    ) -> StagePlan:
        """Stage 6: what should the model be asked to learn?"""
        plan = StagePlan(stage=Stage.OBJECTIVE, baseline=winner, varying="learning objective")
        for objective in objectives:
            if objective == winner.objective:
                continue
            plan.methodologies.append(winner.with_objective(objective))
        plan.notes.append(
            "Objectives only apply to neural and graph models; classical adapters have "
            "fixed internal objectives and are excluded from this stage."
        )
        plan.notes.append(
            "Point-process objectives (maxent, poisson, deepmaxent) output an intensity, "
            "not a probability. Rank metrics remain comparable; calibration slope does not."
        )
        return plan

    # ------------------------------------------------------------- stage 7 --
    def plan_model(
        self,
        winner: Methodology,
        *,
        models: Sequence[str] = (
            "maxnet", "random-forest", "brt", "gam",
            "knn", "xgboost", "lightgbm", "catboost",
            "tabpfn-sdm", "graph-transformer", "autogluon",
        ),
    ) -> StagePlan:
        """Stage 7: which algorithm, holding the methodology fixed?

        The conventional benchmark -- run last, on the representation the
        earlier stages selected, so every algorithm competes on equal and
        well-chosen input.
        """
        plan = StagePlan(stage=Stage.MODEL, baseline=winner, varying="prediction model")
        for model in models:
            plan.methodologies.append(winner.with_model(model))
        return plan

    # ---------------------------------------------------------------- full --
    def plan_all(
        self,
        *,
        modalities: Sequence[str] | None = None,
        scales: Sequence[str] = ("250m", "1km", "5km", "25km"),
        models: Sequence[str] | None = None,
        stages: Iterable[Stage] | None = None,
    ) -> BenchmarkPlan:
        """Build the whole plan up front, using the base methodology throughout.

        This is a *plan*, not a run. Later stages are enumerated against the
        base methodology rather than a real winner, so the configuration counts
        are honest but the specific configurations will be regenerated as each
        stage completes. :meth:`next_stage` does that during execution.
        """
        wanted = set(stages) if stages else set(Stage)
        plan = BenchmarkPlan(
            reference_model=self.reference_model,
            n_species=self.n_species,
            n_scenarios=self.n_scenarios,
        )
        base = self.base

        if Stage.INFORMATION in wanted:
            plan.stages.append(
                self.plan_information(modalities or base.modality_names)
            )
        if Stage.SCALE in wanted:
            plan.stages.append(self.plan_scale(base, scales=scales))
        if Stage.REPRESENTATION in wanted:
            plan.stages.append(self.plan_representation(base))
        if Stage.FUSION in wanted:
            plan.stages.append(self.plan_fusion(base))
        if Stage.SPATIAL in wanted:
            plan.stages.append(self.plan_spatial(base))
        if Stage.OBJECTIVE in wanted:
            plan.stages.append(self.plan_objective(base))
        if Stage.MODEL in wanted:
            plan.stages.append(
                self.plan_model(base, models=models) if models else self.plan_model(base)
            )

        plan.notes = [
            "Stages are sequential: each is generated from the previous stage's winner, "
            "so the configurations listed for stages 2+ are estimates until those stages run.",
            "Greedy stage selection can miss interactions between axes. Increase top_k to "
            "carry several candidates forward.",
            "Stage winners are maxima over noisy estimates and are optimistically biased. "
            "Report the final methodology from a clean run, and support any superiority "
            "claim with the paired tests in sdmbench.evaluation.compare.",
        ]
        return plan

    def next_stage(
        self,
        stage: Stage,
        winner: Methodology,
        **kwargs: Any,
    ) -> StagePlan:
        """Generate one stage from the actual winner of the previous stage."""
        builders = {
            Stage.INFORMATION: lambda: self.plan_information(
                kwargs.pop("modalities", winner.modality_names), **kwargs
            ),
            Stage.SCALE: lambda: self.plan_scale(winner, **kwargs),
            Stage.REPRESENTATION: lambda: self.plan_representation(winner, **kwargs),
            Stage.FUSION: lambda: self.plan_fusion(winner, **kwargs),
            Stage.SPATIAL: lambda: self.plan_spatial(winner, **kwargs),
            Stage.OBJECTIVE: lambda: self.plan_objective(winner, **kwargs),
            Stage.MODEL: lambda: self.plan_model(winner, **kwargs),
        }
        return builders[stage]()


def _renamed(self: Methodology, name: str) -> Methodology:
    """Return a copy with a new display name (hash is unaffected)."""
    from dataclasses import replace as _replace

    return _replace(self, name=name)


# Attached rather than defined on the dataclass so that Methodology stays a
# plain data object with no planner-specific API.
Methodology._renamed = _renamed  # type: ignore[attr-defined]
