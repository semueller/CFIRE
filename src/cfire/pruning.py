from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Set, TypeAlias, Tuple

import numpy as np

from .pruning_metrics import RuleKey, RuleMetrics, compute_rule_metrics

Literal:    TypeAlias = Tuple[int, Tuple[float, float]]     # (dimension, (low, high)) interval test
Clause:     TypeAlias = list[Literal]                       # Conjunction (AND) of literals
ClassRules: TypeAlias = list[Clause]                        # Disjunction (OR) of clauses for one class label
Rules:      TypeAlias = list[ClassRules]                    # List of ClassRules, one entry per class in the data set

Box: TypeAlias = list[Tuple[float, float]]

@dataclass(frozen=True)
class PruningDecision:
    to_remove: Set[RuleKey]
    rationale: str

def decide_by_wins(metrics: RuleMetrics, win_threshold: int) -> PruningDecision:
    """
    Classic policy from the original script:
    remove rules whose absolute win count <= win_threshold.
    """
    to_remove: Set[RuleKey] = {
        metrics.clause_keys[i]
        for i, w in enumerate(metrics.wins)
        if int(w) <= int(win_threshold)
    }
    rationale = f"Removed {len(to_remove)}/{len(metrics.clause_keys)} rules with wins <= {win_threshold}"
    return PruningDecision(to_remove=to_remove, rationale=rationale)

def prune_rules(rule_tree: Sequence[Sequence], to_remove: Set[RuleKey]) -> Rules:
    """
    Pure function: return a new rules structure with (class, clause) keys filtered out.
    No side effects; user decides outside whether to set cfire.dnf.rules = new_rules.
    """
    return [
        [r for cid, r in enumerate(rules) if (cls, cid) not in to_remove]
        for cls, rules in enumerate(rule_tree)
    ]


# ---------------------------------------------------------------------------------------------------------------
# pruning of a rule model (CFIRE pruning paper, arXiv:2601.03776)
@dataclass
class PruneResult:
    """Outcome of `prune`. The original rule model is not modified.

    dnf: the pruned rule model; win_threshold: rules with at most this many wins on the fit data were removed;
    removed: one entry per removed rule (class, rule, n_covered = fit samples it matches, share_covered_by_kept /
    share_covered_by_kept_same_class = share of those samples that a kept rule (of the same class) also matches);
    before / after: accuracy (rejects count as wrong), precision (agreement on covered samples), coverage and rule
    count on the fit data."""
    dnf: object
    mode: str
    tolerance: float | None
    win_threshold: int
    removed: list
    before: dict
    after: dict

    def to_dict(self) -> dict:
        return {"mode": self.mode, "tolerance": self.tolerance, "win_threshold": self.win_threshold,
                "before": self.before, "after": self.after,
                "removed": [{**r, "rule": [[int(f), _num(lo), _num(hi)] for f, (lo, hi) in r["rule"]]} for r in self.removed]}


def _num(v):
    v = float(v)
    return None if not np.isfinite(v) else v


class _AsCFIRE:
    """the interface compute_rule_metrics expects (dnf attribute, call with explain=True)"""
    def __init__(self, dnf):
        self.dnf = dnf

    def __call__(self, X, explain=False):
        return self.dnf(X, explain=explain)


def _summary(dnf, X, Y) -> dict:
    pred = np.asarray(dnf.predict(X))
    covered = pred != -1
    return {"accuracy": float(np.mean(pred == Y)),
            "precision": float(np.mean(pred[covered] == Y[covered])) if covered.any() else 0.0,
            "coverage": float(np.mean(covered)), "n_rules": int(sum(len(c) for c in dnf.rules))}


def prune(dnf, X, Y, mode: str = "safe", tolerance: float = 0.05, max_threshold: int = 99) -> PruneResult:
    """Prune a rule model by how often its rules win on the fit data (the rule with the best fit-data accuracy among
    the rules matching a sample wins it).

    mode "safe": remove entirely redundant rules based on the fit data (rules that never win; win_threshold 0).
    mode "best": raise the threshold 0, 1, ..., max_threshold and keep the last rule model whose accuracy on the fit
    data stays at or above (1 - tolerance) times the original accuracy (the search of the pruning paper; tolerance
    0.05 there). Wins are counted once, on the original rule model.
    X, Y: fit data and the black box's predictions (the data the rule model was fitted on)."""
    from copy import deepcopy
    from .rules import DNFClassifier
    if mode not in ("safe", "best"):
        raise ValueError(f"mode must be 'safe' or 'best', got {mode!r}")
    X, Y = np.asarray(X), np.asarray(Y)
    original = deepcopy(dnf)
    original.compute_rule_performance(X, Y)
    metrics = compute_rule_metrics(_AsCFIRE(original), X)
    before = _summary(original, X, Y)

    def candidate(threshold):
        to_remove = decide_by_wins(metrics, win_threshold=threshold).to_remove
        model = DNFClassifier(prune_rules(original.rules, to_remove), "accuracy")
        model.compute_rule_performance(X, Y)
        return model, to_remove

    best, best_removed, best_t = (*candidate(0), 0)
    if mode == "best":
        for t in range(1, max_threshold + 1):
            model, to_remove = candidate(t)
            if before["accuracy"] > 0 and np.mean(np.asarray(model.predict(X)) == Y) < (1 - tolerance) * before["accuracy"]:
                break
            best, best_removed, best_t = model, to_remove, t

    # what the removed rules carried: samples they match, and how many of those a kept rule also matches
    matches = metrics.match_per_sample
    kept = {(c, i) for c, rules in enumerate(original.rules) for i in range(len(rules))} - set(best_removed)
    removed = []
    for (c, i) in sorted(best_removed):
        rows = [s for s, keys in enumerate(matches) if (c, i) in keys]
        by_kept = [s for s in rows if any(k in kept for k in matches[s])]
        same = [s for s in rows if any(k in kept and k[0] == c for k in matches[s])]
        n = len(rows)
        removed.append({"class": int(c), "rule": original.rules[c][i], "n_covered": n,
                        "share_covered_by_kept": len(by_kept) / n if n else 1.0,
                        "share_covered_by_kept_same_class": len(same) / n if n else 1.0})
    return PruneResult(dnf=best, mode=mode, tolerance=tolerance if mode == "best" else None, win_threshold=best_t,
                       removed=removed, before=before, after=_summary(best, X, Y))
