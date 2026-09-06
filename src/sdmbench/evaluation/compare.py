"""Paired model comparison.

Because every model is evaluated on the same species under the same protocol,
comparisons are **paired**. That is a substantially stronger design than
comparing two independent means: the species-to-species variation in difficulty
is enormous (a species with 12 presences and one with 900 are not the same
problem), and pairing removes it.

Practical consequence: a 0.03 difference in mean ROC-AUC across 226 species may
be highly consistent (a model that wins on 200 of them) or noise (a model that
wins on 115). The mean alone cannot tell you which. So every comparison here
reports the paired difference, a bootstrap CI, win/tie/loss counts, and a
paired test -- and :meth:`ComparisonResult.verdict` refuses to call a winner on
a numerically larger mean alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from sdmbench.evaluation.aggregate import per_species_table

__all__ = ["ComparisonResult", "compare_models", "compare_all_pairs", "bootstrap_ci"]


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int = 10000,
    confidence: float = 0.95,
    seed: int = 32639,
    statistic: str = "mean",
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a statistic."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    func = np.median if statistic == "median" else np.mean
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True)
    stats_ = func(draws, axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(stats_, alpha)),
        float(np.quantile(stats_, 1.0 - alpha)),
    )


@dataclass
class ComparisonResult:
    """The outcome of comparing two models across species."""

    model_a: str
    model_b: str
    metric: str
    scenario: str | None
    n_species: int
    mean_a: float
    mean_b: float
    mean_difference: float
    median_difference: float
    ci_low: float
    ci_high: float
    wins: int
    ties: int
    losses: int
    wilcoxon_statistic: float = np.nan
    wilcoxon_p: float = np.nan
    ttest_statistic: float = np.nan
    ttest_p: float = np.nan
    effect_size: float = np.nan
    tie_tolerance: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def ci_excludes_zero(self) -> bool:
        """Whether the bootstrap CI for the paired difference excludes zero."""
        if not np.isfinite(self.ci_low) or not np.isfinite(self.ci_high):
            return False
        return (self.ci_low > 0) or (self.ci_high < 0)

    def verdict(self, alpha: float = 0.05) -> str:
        """A conservative statement of what the comparison supports.

        Requires agreement between the bootstrap CI and the paired test before
        claiming a difference. A larger mean on its own is never sufficient.
        """
        if self.n_species < 3:
            return "INSUFFICIENT_DATA"
        significant = self.ci_excludes_zero and (
            np.isfinite(self.wilcoxon_p) and self.wilcoxon_p < alpha
        )
        if not significant:
            return "NO_RELIABLE_DIFFERENCE"
        return f"{self.model_a}_BETTER" if self.mean_difference > 0 else f"{self.model_b}_BETTER"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "metric": self.metric,
            "scenario": self.scenario,
            "n_species": self.n_species,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "mean_difference": self.mean_difference,
            "median_difference": self.median_difference,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "wins": self.wins,
            "ties": self.ties,
            "losses": self.losses,
            "wilcoxon_statistic": self.wilcoxon_statistic,
            "wilcoxon_p": self.wilcoxon_p,
            "ttest_statistic": self.ttest_statistic,
            "ttest_p": self.ttest_p,
            "effect_size_cohens_dz": self.effect_size,
            "verdict": self.verdict(),
            "notes": self.notes,
        }

    def render(self) -> str:
        lines = [
            f"{self.model_a}  vs  {self.model_b}   [{self.metric}"
            + (f", {self.scenario}" if self.scenario else "")
            + "]",
            f"  species compared     {self.n_species}",
            f"  mean {self.model_a:<18} {self.mean_a:.4f}",
            f"  mean {self.model_b:<18} {self.mean_b:.4f}",
            f"  paired difference    {self.mean_difference:+.4f} "
            f"(median {self.median_difference:+.4f})",
            f"  95% bootstrap CI     [{self.ci_low:+.4f}, {self.ci_high:+.4f}]",
            f"  win / tie / loss     {self.wins} / {self.ties} / {self.losses}",
            f"  Wilcoxon signed-rank p = {self.wilcoxon_p:.4g}",
            f"  paired t-test        p = {self.ttest_p:.4g}",
            f"  effect size (dz)     {self.effect_size:.3f}",
            f"  VERDICT              {self.verdict()}",
        ]
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


def compare_models(
    results: pd.DataFrame,
    model_a: str,
    model_b: str,
    *,
    metric: str = "roc_auc",
    scenario: str | None = None,
    tie_tolerance: float = 0.0,
    n_boot: int = 10000,
    confidence: float = 0.95,
    seed: int = 32639,
) -> ComparisonResult:
    """Compare two models on the species both completed.

    Parameters
    ----------
    tie_tolerance:
        Differences with magnitude at or below this count as ties rather than
        wins. ``0.0`` counts any non-zero difference.
    """
    table = per_species_table(results, metric=metric, scenario=scenario)
    if table.empty or model_a not in table.columns or model_b not in table.columns:
        available = list(table.columns) if not table.empty else []
        raise KeyError(
            f"cannot compare {model_a!r} and {model_b!r}: available models with "
            f"{metric!r} results are {available}"
        )

    # Restrict to species BOTH models completed -- the pairing is the point.
    paired = table.loc[:, [model_a, model_b]].dropna(how="any")
    a = paired[model_a].to_numpy(dtype=float)
    b = paired[model_b].to_numpy(dtype=float)
    differences = a - b
    n = len(differences)

    notes: list[str] = []
    dropped = len(table) - n
    if dropped:
        notes.append(
            f"{dropped} species dropped because at least one model has no {metric} value there"
        )
    if n < 3:
        notes.append("too few paired species for a meaningful test")

    wins = int(np.sum(differences > tie_tolerance))
    losses = int(np.sum(differences < -tie_tolerance))
    ties = n - wins - losses

    ci_low, ci_high = bootstrap_ci(
        differences, n_boot=n_boot, confidence=confidence, seed=seed
    )

    wilcoxon_stat = wilcoxon_p = ttest_stat = ttest_p = np.nan
    if n >= 3 and np.any(differences != 0):
        from scipy import stats

        try:
            # Wilcoxon is the primary test: paired metric differences across
            # species are not reliably normal.
            result = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            wilcoxon_stat, wilcoxon_p = float(result.statistic), float(result.pvalue)
        except ValueError as exc:
            notes.append(f"Wilcoxon test unavailable: {exc}")
        try:
            t_result = stats.ttest_rel(a, b)
            ttest_stat, ttest_p = float(t_result.statistic), float(t_result.pvalue)
        except ValueError as exc:
            notes.append(f"paired t-test unavailable: {exc}")

    sd = float(np.std(differences, ddof=1)) if n > 1 else np.nan
    effect = float(np.mean(differences) / sd) if sd and np.isfinite(sd) and sd > 0 else np.nan

    return ComparisonResult(
        model_a=model_a,
        model_b=model_b,
        metric=metric,
        scenario=scenario,
        n_species=n,
        mean_a=float(np.mean(a)) if n else np.nan,
        mean_b=float(np.mean(b)) if n else np.nan,
        mean_difference=float(np.mean(differences)) if n else np.nan,
        median_difference=float(np.median(differences)) if n else np.nan,
        ci_low=ci_low,
        ci_high=ci_high,
        wins=wins,
        ties=ties,
        losses=losses,
        wilcoxon_statistic=wilcoxon_stat,
        wilcoxon_p=wilcoxon_p,
        ttest_statistic=ttest_stat,
        ttest_p=ttest_p,
        effect_size=effect,
        tie_tolerance=tie_tolerance,
        notes=notes,
    )


def compare_all_pairs(
    results: pd.DataFrame,
    *,
    metric: str = "roc_auc",
    scenario: str | None = None,
    reference: str | None = None,
    correction: str = "holm",
    **kwargs: Any,
) -> pd.DataFrame:
    """Compare every model pair, or every model against a reference.

    Parameters
    ----------
    reference:
        When given, only ``reference vs X`` comparisons are made -- the usual
        design when asking whether anything beats the published state of the
        art.
    correction:
        Multiple-comparison correction for the Wilcoxon p-values: ``"holm"``
        (default), ``"bonferroni"``, or ``"none"``. Running every pair without
        correction manufactures significance.
    """
    table = per_species_table(results, metric=metric, scenario=scenario)
    if table.empty:
        return pd.DataFrame()
    models = list(table.columns)

    pairs: list[tuple[str, str]] = []
    if reference is not None:
        if reference not in models:
            raise KeyError(f"reference model {reference!r} has no {metric} results")
        pairs = [(reference, m) for m in models if m != reference]
    else:
        pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1 :]]

    rows = []
    for a, b in pairs:
        try:
            rows.append(
                compare_models(results, a, b, metric=metric, scenario=scenario, **kwargs).to_dict()
            )
        except KeyError:
            continue
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["wilcoxon_p_adjusted"] = _adjust_pvalues(
        frame["wilcoxon_p"].to_numpy(dtype=float), method=correction
    )
    frame["correction"] = correction
    return frame.sort_values("mean_difference", ascending=False).reset_index(drop=True)


def _adjust_pvalues(p: np.ndarray, *, method: str = "holm") -> np.ndarray:
    """Holm-Bonferroni (default) or Bonferroni correction."""
    p = np.asarray(p, dtype=float)
    finite = np.isfinite(p)
    out = np.full_like(p, np.nan)
    values = p[finite]
    n = len(values)
    if n == 0:
        return out
    if method == "none":
        out[finite] = values
        return out
    if method == "bonferroni":
        out[finite] = np.minimum(values * n, 1.0)
        return out
    if method != "holm":
        raise ValueError(f"unknown correction {method!r}; use holm, bonferroni or none")

    order = np.argsort(values)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        # Holm: step down, enforcing monotonicity.
        running = max(running, (n - rank) * values[idx])
        adjusted[idx] = min(running, 1.0)
    out[finite] = adjusted
    return out
