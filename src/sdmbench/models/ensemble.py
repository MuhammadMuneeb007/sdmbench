"""Presence/background ensemble class balancing.

The motivating asymmetry, from Dinnage & Warren (2026) sec. 2.3:

    "Presence records are valuable and relatively scarce, representing actual
     observations of species occurrence. Pseudo-absences, by contrast, are
     interchangeable samples from the background environmental space; any given
     pseudo-absence point could be replaced by another randomly drawn
     background point without loss of information."

So presences are kept whole in every ensemble member, and only the background
is resampled. The component is model-agnostic: any adapter benefits from it if
its learner is sensitive to class imbalance, so it is not confined to TabPFN.

Two schemes -- and they are not the same
----------------------------------------
The two public sources describe *different* procedures, and this matters
enough to implement both explicitly rather than average them into a guess.

``"balanced"`` (paper sec. 2.3, the default)
    1. Retain all presence records in every ensemble member.
    2. For each of K members, draw a balanced random sample of pseudo-absences
       **equal in number to the presences**.
    3. Members may overlap.
    4. Fit on each approximately balanced subset.
    5. **Average logits** across members before applying softmax.

    With K = 16, ``n_estimators = 16``, ``average_before_softmax = True`` and
    ``balance_probabilities = True``.

``"partition"`` (Hugging Face model card)
    Keeps all presences in every sub-batch but **partitions** the background
    across sub-batches so that 100% of the background is used exactly once,
    then averages predictions. The card describes this for both the training
    sub-batching and ``run_tabpfn_finetuned_ensemble()``.

These give different training sets: ``balanced`` draws ``n_presence``
background points per member with overlap between members; ``partition``
divides the whole background into disjoint blocks. The paper's own words --
"a different balanced subsample of absences, allowing overlap between members"
-- specify the former, so it is the default for the published ``TabPFN-SS``
variant. The upstream repository (``rdinnager/TabPFN-SDM``) was not public when
this was written, so the discrepancy could not be resolved against the authors'
code; :mod:`sdmbench.provenance` records this as UNVERIFIED and
``sdmbench upstream diff`` will re-check it once the repository appears.

Averaging in logit space
------------------------
Step 5 averages *logits*, not probabilities. The paper notes this "tends to
produce better-calibrated probability estimates than averaging probabilities
directly" -- which is the point, given that Miller's calibration slope is one
of the three reported metrics. :func:`average_logits` implements it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Literal, Sequence

import numpy as np

__all__ = [
    "PresenceBackgroundEnsemble",
    "EnsembleMember",
    "average_logits",
    "average_probabilities",
    "PUBLISHED_K",
]

#: Ensemble size used in the paper. Sec. 2.3: "The choice of K = 16 ensemble
#: members reflects a balance between computational cost and ensemble
#: diversity ... performance improved with increasing K up to approximately 16
#: members, with diminishing returns thereafter."
PUBLISHED_K = 16

_EPS = 1e-7


def _logit(p: np.ndarray, eps: float = _EPS) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -500, 500)))


def average_logits(predictions: Sequence[np.ndarray]) -> np.ndarray:
    """Average member predictions in logit space, then map back to probability.

    This is the ``average_before_softmax = True`` behaviour: combine on the
    unbounded scale where the members' evidence is additive, then squash once.
    """
    if not len(predictions):
        raise ValueError("no member predictions to average")
    stacked = np.vstack([_logit(np.asarray(p, dtype=float).ravel()) for p in predictions])
    return _expit(stacked.mean(axis=0))


def average_probabilities(predictions: Sequence[np.ndarray]) -> np.ndarray:
    """Plain arithmetic mean of member probabilities."""
    if not len(predictions):
        raise ValueError("no member predictions to average")
    return np.vstack([np.asarray(p, dtype=float).ravel() for p in predictions]).mean(axis=0)


@dataclass
class EnsembleMember:
    """The training indices used by one ensemble member."""

    member_id: int
    indices: np.ndarray
    n_presence: int
    n_background: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "n_train": int(len(self.indices)),
            "n_presence": self.n_presence,
            "n_background": self.n_background,
        }


@dataclass
class PresenceBackgroundEnsemble:
    """Build class-balanced ensemble members from presence-background data.

    Parameters
    ----------
    n_members:
        K, the number of ensemble members. Defaults to :data:`PUBLISHED_K`.
    scheme:
        ``"balanced"`` (paper sec. 2.3) or ``"partition"`` (model card). See the
        module docstring -- these are genuinely different procedures.
    background_ratio:
        Background points per presence in ``"balanced"`` mode. ``1.0`` gives
        the balanced sample the paper specifies.
    max_train_size:
        Cap on a member's training size. The paper capped this at 1,500 samples
        per split "for GPU memory efficiency" (sec. 2.4.2); the model card
        gives the same figure as ``max_train_size_per_subbatch``.
    combine:
        ``"logit"`` (paper) or ``"probability"``.
    seed:
        Base seed; member *i* uses ``seed + i`` so members differ but the whole
        ensemble is reproducible.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    >>> ens = PresenceBackgroundEnsemble(n_members=2, seed=0)
    >>> members = ens.build_members(y)
    >>> all(m.n_presence == 3 and m.n_background == 3 for m in members)
    True
    """

    n_members: int = PUBLISHED_K
    scheme: Literal["balanced", "partition"] = "balanced"
    background_ratio: float = 1.0
    max_train_size: int | None = 1500
    combine: Literal["logit", "probability"] = "logit"
    seed: int = 32639
    #: Recorded per run so the exact member composition is auditable.
    members_: list[EnsembleMember] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.n_members < 1:
            raise ValueError("n_members must be >= 1")
        if self.scheme not in {"balanced", "partition"}:
            raise ValueError(f"unknown scheme {self.scheme!r}; use 'balanced' or 'partition'")
        if self.combine not in {"logit", "probability"}:
            raise ValueError(f"unknown combine {self.combine!r}; use 'logit' or 'probability'")

    # ------------------------------------------------------------- membership --
    def build_members(self, y: np.ndarray) -> list[EnsembleMember]:
        """Compute the training indices for every ensemble member."""
        y = np.asarray(y, dtype=int).ravel()
        presence_idx = np.flatnonzero(y == 1)
        background_idx = np.flatnonzero(y == 0)

        if len(presence_idx) == 0 or len(background_idx) == 0:
            # Single-class data: one member using everything. The caller
            # decides whether that is a skippable condition.
            self.members_ = [
                EnsembleMember(0, np.arange(len(y)), len(presence_idx), len(background_idx))
            ]
            return self.members_

        members = (
            self._balanced_members(presence_idx, background_idx)
            if self.scheme == "balanced"
            else self._partition_members(presence_idx, background_idx)
        )
        self.members_ = members
        return members

    def _balanced_members(
        self, presence_idx: np.ndarray, background_idx: np.ndarray
    ) -> list[EnsembleMember]:
        """Paper sec. 2.3: balanced draw per member, overlap allowed."""
        n_pres = len(presence_idx)
        n_draw = int(round(n_pres * self.background_ratio))
        n_draw = max(1, min(n_draw, len(background_idx)))

        members: list[EnsembleMember] = []
        for i in range(self.n_members):
            rng = np.random.default_rng(self.seed + i)
            chosen_bg = rng.choice(background_idx, size=n_draw, replace=False)
            indices = np.sort(np.concatenate([presence_idx, chosen_bg]))
            indices = self._cap(indices, presence_idx, rng)
            members.append(
                EnsembleMember(
                    member_id=i,
                    indices=indices,
                    n_presence=int(np.isin(indices, presence_idx).sum()),
                    n_background=int(np.isin(indices, background_idx).sum()),
                )
            )
        return members

    def _partition_members(
        self, presence_idx: np.ndarray, background_idx: np.ndarray
    ) -> list[EnsembleMember]:
        """Model card: all presences everywhere, background split disjointly."""
        rng = np.random.default_rng(self.seed)
        shuffled = rng.permutation(background_idx)
        chunks = np.array_split(shuffled, self.n_members)

        members: list[EnsembleMember] = []
        for i, chunk in enumerate(chunks):
            if len(chunk) == 0:
                continue
            indices = np.sort(np.concatenate([presence_idx, chunk]))
            indices = self._cap(indices, presence_idx, np.random.default_rng(self.seed + i))
            members.append(
                EnsembleMember(
                    member_id=i,
                    indices=indices,
                    n_presence=int(np.isin(indices, presence_idx).sum()),
                    n_background=int(np.isin(indices, background_idx).sum()),
                )
            )
        return members

    def _cap(
        self, indices: np.ndarray, presence_idx: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Enforce ``max_train_size``, dropping background before presences.

        Presences are scarce and individually informative; the cap exists for
        GPU memory, so it takes from the interchangeable class first.
        """
        if self.max_train_size is None or len(indices) <= self.max_train_size:
            return indices
        is_presence = np.isin(indices, presence_idx)
        pres = indices[is_presence]
        bg = indices[~is_presence]
        room = self.max_train_size - len(pres)
        if room <= 0:
            # More presences than the cap: keep a random subset of presences.
            return np.sort(rng.choice(pres, size=self.max_train_size, replace=False))
        keep_bg = rng.choice(bg, size=min(room, len(bg)), replace=False)
        return np.sort(np.concatenate([pres, keep_bg]))

    def iter_member_data(
        self, X: Any, y: np.ndarray
    ) -> Iterator[tuple[EnsembleMember, Any, np.ndarray]]:
        """Yield ``(member, X_member, y_member)`` for each ensemble member."""
        y = np.asarray(y, dtype=int).ravel()
        for member in self.build_members(y):
            idx = member.indices
            X_member = X.iloc[idx] if hasattr(X, "iloc") else np.asarray(X)[idx]
            yield member, X_member, y[idx]

    # ----------------------------------------------------------------- fitting --
    def fit_predict(
        self,
        fit_predict_fn: Callable[[Any, np.ndarray, Any], np.ndarray],
        X_train: Any,
        y_train: np.ndarray,
        X_test: Any,
    ) -> np.ndarray:
        """Run the full ensemble and combine member predictions.

        Parameters
        ----------
        fit_predict_fn:
            ``(X_member, y_member, X_test) -> probabilities``. Called once per
            member with a fresh model.

        Returns
        -------
        numpy.ndarray
            Combined probability of presence per test row.
        """
        predictions: list[np.ndarray] = []
        for _member, X_member, y_member in self.iter_member_data(X_train, y_train):
            predictions.append(np.asarray(fit_predict_fn(X_member, y_member, X_test), dtype=float))
        if not predictions:
            raise ValueError("ensemble produced no member predictions")
        return (
            average_logits(predictions)
            if self.combine == "logit"
            else average_probabilities(predictions)
        )

    # ---------------------------------------------------------------- metadata --
    def describe(self) -> dict[str, Any]:
        return {
            "n_members": self.n_members,
            "scheme": self.scheme,
            "background_ratio": self.background_ratio,
            "max_train_size": self.max_train_size,
            "combine": self.combine,
            "seed": self.seed,
            "source": (
                "Dinnage & Warren 2026 sec. 2.3"
                if self.scheme == "balanced"
                else "rdinnager/tabpfn-sdm-finetuned model card"
            ),
            "members": [m.to_dict() for m in self.members_],
        }
