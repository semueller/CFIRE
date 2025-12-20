from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Set, Tuple, TypedDict

import numpy as np

RuleKey = Tuple[int, int]  # (class_id, clause_id)

class PerfDict(TypedDict, total=False):
    accuracy: float  # extend later if needed (e.g., support, f1)

@dataclass(frozen=True)
class RuleMetrics:
    # structure
    clause_keys: List[RuleKey]                     # stable order of (class, clause)
    match_per_sample: List[List[RuleKey]]          # for each sample: matched rule keys

    # rule quality / winners
    perf_by_key: Dict[RuleKey, PerfDict]           # (class,clause) -> {'accuracy': ...}
    wins: np.ndarray                               # shape [n_clauses]
    loss: np.ndarray                               # shape [n_clauses]
    winner_key_per_sample: List[Optional[RuleKey]] # length = n_samples

    # extra descriptive stats (no printing)
    coverage_per_rule: np.ndarray                  # how often each rule matched
    match_hist: Dict[int, int]                     # #matches -> count of samples
    share_multi: float                             # % of samples with ≥2 matches
    collision_ratio: Dict[str, float]              # {'intra': %, 'inter': %}

class   CFIRELike(Protocol):
    @property
    def dnf(self) -> object: ...
    def __call__(self, X, explain: bool = False):
        """When explain=True, should return either:
           - iterable of (pred, matches) with matches: List[Tuple[RuleKey, any_payload]]
           - OR directly List[List[RuleKey]]
        """

def canon(rule: Sequence) -> Tuple:
    """Canonicalize rule so it matches keys in rule_performances."""
    return tuple(rule) if isinstance(rule, list) else rule

def _normalize_matches(explain_out) -> List[List[RuleKey]]:
    """Normalize CFIRE explain output to List[List[RuleKey]].

    Supported:
      A) (preds, matches_per_sample)
      B) iterable of (pred, matches) per sample
      C) already List[List[RuleKey]]
    Where each `matches` can be:
      - None  -> treated as []
      - List[Tuple[RuleKey, any_payload]]
      - List[RuleKey]
    """
    # Case A: top-level tuple
    if isinstance(explain_out, tuple) and len(explain_out) == 2:
        _, matches_global = explain_out
        return _normalize_matches(matches_global)

    # Ensure indexable sequence
    try:
        first = explain_out[0]  # type: ignore[index]
    except Exception:
        explain_out = list(explain_out)
        first = explain_out[0] if explain_out else None

    # Case B: list of (pred, matches) pairs per-sample
    if isinstance(first, tuple) and len(first) == 2:
        out: List[List[RuleKey]] = []
        for _, matches in explain_out:  # type: ignore[assignment]
            if matches is None:
                out.append([])
                continue
            # matches might be List[(RuleKey, payload)] or List[RuleKey]
            if isinstance(matches, list) and matches:
                m0 = matches[0]
                # List[(RuleKey, payload)]
                if isinstance(m0, tuple) and len(m0) >= 2 and isinstance(m0[0], tuple):
                    out.append([k for (k, _) in matches])
                # List[RuleKey]
                elif isinstance(m0, tuple) and len(m0) == 2 and all(isinstance(x, (int, np.integer)) for x in m0):
                    out.append(matches)
                else:
                    out.append([])
            else:
                out.append([])
        return out

    # Case C: already List[List[RuleKey]]
    return explain_out

def _build_perf_by_key(cf: CFIRELike) -> Dict[RuleKey, PerfDict]:
    """Extract per‑rule accuracy from cf.dnf.rule_performances mapping."""
    out: Dict[RuleKey, PerfDict] = {}
    rules = getattr(cf.dnf, "rules")
    perf = getattr(cf.dnf, "rule_performances")
    for cls_id, class_rules in enumerate(rules):
        for cid, rule in enumerate(class_rules):
            pd = perf[cls_id][canon(rule)]
            out[(cls_id, cid)] = {"accuracy": float(pd.get("accuracy", 0.0))}
    return out

def _compute_winners(
        match_per_sample: List[List[RuleKey]],
        clause_keys: List[RuleKey],
        perf_by_key: Dict[RuleKey, PerfDict],
) -> tuple[np.ndarray, np.ndarray, List[Optional[RuleKey]]]:
    """Winner = matched rule with max per‑rule accuracy."""
    key2col = {k: i for i, k in enumerate(clause_keys)}

    def best_key(keys: List[RuleKey]) -> RuleKey:
        return max(keys, key=lambda k: float(perf_by_key.get(k, {}).get("accuracy", 0.0)))

    n_samples, n_clauses = len(match_per_sample), len(clause_keys)
    winner_idx = np.full(n_samples, -1, dtype=int)
    wins = np.zeros(n_clauses, dtype=int)
    loss = np.zeros(n_clauses, dtype=int)

    for s, keys in enumerate(match_per_sample):
        if not keys:
            continue
        w = best_key(keys)
        w_col = key2col[w]
        winner_idx[s] = w_col
        for k in keys:
            (wins if k == w else loss)[key2col[k]] += 1

    winner_key_per_sample: List[Optional[RuleKey]] = [
        clause_keys[idx] if idx != -1 else None
        for idx in winner_idx
    ]
    return wins, loss, winner_key_per_sample

def _coverage_hist_collision(
        clause_keys: List[RuleKey],
        match_per_sample: List[List[RuleKey]],
        wins: np.ndarray,
) -> tuple[np.ndarray, Dict[int, int], float, Dict[str, float]]:
    """
    coverage_per_rule, match_hist, share_multi, collision_ratio
    """
    n_samples, n_clauses = len(match_per_sample), len(clause_keys)
    key2col = {k: i for i, k in enumerate(clause_keys)}

    M = np.zeros((n_samples, n_clauses), dtype=bool)
    coverage = np.zeros(n_clauses, dtype=int)
    match_counts = np.zeros(n_samples, dtype=int)

    for s, keys in enumerate(match_per_sample):
        match_counts[s] = len(keys)
        for k in keys:
            j = key2col[k]
            M[s, j] = True
            coverage[j] += 1

    # histogram of #matches per sample
    uniq, cnts = np.unique(match_counts, return_counts=True)
    match_hist = {int(k): int(v) for k, v in zip(uniq.tolist(), cnts.tolist())}
    share_multi = float((match_counts >= 2).mean() * 100.0)

    # collision ratio based on winner's class
    # (we don't need 'wins' here strictly; included for symmetry with original)
    intra_res = 0
    inter_res = 0
    # For collision we need the winner per sample; recompute quickly with "most frequent winner class":
    # to avoid recomputing, we’ll compute using majority class among matched vs any other class present.
    #  here we just return placeholders; real computation happens in compute_rule_metrics with true winners.
    # We'll just return zeros here; compute_rule_metrics will overwrite with correct ratio.
    collision_ratio = {"intra": 0.0, "inter": 0.0}

    return coverage, match_hist, share_multi, collision_ratio

def _collision_ratio_from_winners(
        match_per_sample: List[List[RuleKey]],
        winner_key_per_sample: List[Optional[RuleKey]],
) -> Dict[str, float]:
    intra_res = 0
    inter_res = 0
    for keys, w in zip(match_per_sample, winner_key_per_sample):
        if len(keys) <= 1 or w is None:
            continue
        other_cls = {k[0] for k in keys} - {w[0]}
        if other_cls:
            inter_res += 1
        else:
            intra_res += 1
    total = intra_res + inter_res
    if total == 0:
        return {"intra": 0.0, "inter": 0.0}
    return {
        "intra": intra_res / total * 100.0,
        "inter": inter_res / total * 100.0,
    }


def compute_rule_metrics(cfire: CFIRELike, X_val) -> RuleMetrics:
    """
    Single call that the experiment orchestrator can use.
    Returns a RuleMetrics object with everything needed for pruning decisions.
    """
    # matches
    explain_out = cfire(X_val, explain=True)
    match_per_sample = _normalize_matches(explain_out)

    # stable key list by first appearance
    clause_keys: List[RuleKey] = []
    seen: Set[RuleKey] = set()
    for keys in match_per_sample:
        for k in keys:
            if k not in seen:
                seen.add(k)
                clause_keys.append(k)

    # per-rule perf and winners
    perf_by_key = _build_perf_by_key(cfire)
    wins, loss, winner_key_per_sample = _compute_winners(match_per_sample, clause_keys, perf_by_key)

    # descriptive stats
    coverage_per_rule, match_hist, share_multi, _ = _coverage_hist_collision(
        clause_keys, match_per_sample, wins
    )
    collision_ratio = _collision_ratio_from_winners(match_per_sample, winner_key_per_sample)

    return RuleMetrics(
        clause_keys=clause_keys,
        match_per_sample=match_per_sample,
        perf_by_key=perf_by_key,
        wins=wins,
        loss=loss,
        winner_key_per_sample=winner_key_per_sample,
        coverage_per_rule=coverage_per_rule,
        match_hist=match_hist,
        share_multi=share_multi,
        collision_ratio=collision_ratio,
    )


def _stats_summary(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return dict(count=0, mean=0.0, std=0.0, min=0.0, p25=0.0, median=0.0, p75=0.0, max=0.0)
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "max": float(np.max(x)),
    }


def loggable_rule_metrics(rm: RuleMetrics, win_threshold) -> dict:
    """
    Flatten RuleMetrics into a dict of scalars with explicit names.
    - Drops winner_coverage_pct (as requested).
    - Renames match-hist keys to explicit sample_* names.
    - Uses clause_* prefixes for per-clause summaries.
    """
    out: dict = {}

    out["win_threshold"] = win_threshold

    # Totals
    clauses_total = int(len(rm.clause_keys))
    samples_total = int(len(rm.match_per_sample))
    out["clauses_total"] = clauses_total
    out["samples_total"] = samples_total

    # ---- Per-clause coverage distribution ----
    coverage_per_clause = np.asarray(rm.coverage_per_rule, dtype=float) if clauses_total else np.array([])
    cov_stats = _stats_summary(coverage_per_clause)
    for k, v in cov_stats.items():
        out[f"per_clause_coverage_{k}"] = v  # mean/std/min/… of clause coverages
    out["clauses_active_pct"] = 100.0 * float((coverage_per_clause > 0).mean()) if coverage_per_clause.size else 0.0

    # ---- Sample-level determinism / ambiguity (match histogram) ----
    match_hist = rm.match_hist  # {num_rules_that_matched: sample_count}
    samples_uncovered = int(match_hist.get(0, 0))                             # 0 matches
    samples_single_match = int(match_hist.get(1, 0))                          # exactly 1 match
    samples_multi_match = int(sum(v for k, v in match_hist.items() if k >= 2))# 2+ matches

    denom = max(samples_total, 1)
    out["samples_uncovered"] = samples_uncovered
    out["samples_uncovered_pct"] = 100.0 * samples_uncovered / denom
    out["samples_single_match"] = samples_single_match
    out["samples_single_match_pct"] = 100.0 * samples_single_match / denom
    out["samples_multi_match"] = samples_multi_match
    out["samples_multi_match_pct"] = 100.0 * samples_multi_match / denom

    out["avg_rules_matched_per_sample"] = float(sum(k * v for k, v in match_hist.items()) / denom) if match_hist else 0.0
    out["max_rules_matched_per_sample"] = int(max(match_hist.keys())) if match_hist else 0

    # ---- Collision ratios (already % upstream) ----
    out["collision_intra_pct"] = float(rm.collision_ratio.get("intra", 0.0))
    out["collision_inter_pct"] = float(rm.collision_ratio.get("inter", 0.0))

    # ---- Wins/loss + per-clause win-rate stats ----
    wins = np.asarray(rm.wins, dtype=float)
    loss = np.asarray(rm.loss, dtype=float)
    total_duels = wins + loss
    with np.errstate(invalid="ignore", divide="ignore"):
        clause_winrate = np.where(total_duels > 0, wins / total_duels, 0.0)
    wr_stats = _stats_summary(clause_winrate)
    out["wins_total"] = float(wins.sum())
    out["loss_total"] = float(loss.sum())
    for k, v in wr_stats.items():
        out[f"clause_winrate_{k}"] = v


    distinct_winner_clauses = int(len({w for w in rm.winner_key_per_sample if w is not None}))
    out["distinct_winner_clauses"] = distinct_winner_clauses

    # ---- Per-clause accuracies ----
    clause_accs = np.array([float(d.get("accuracy", 0.0)) for d in rm.perf_by_key.values()], dtype=float)
    acc_stats = _stats_summary(clause_accs)
    for k, v in acc_stats.items():
        out[f"clause_accuracy_{k}"] = v

    if coverage_per_clause.size and clause_accs.size == coverage_per_clause.size and coverage_per_clause.sum() > 0:
        out["clause_accuracy_weighted_mean"] = float((clause_accs * coverage_per_clause).sum() / coverage_per_clause.sum())
    else:
        out["clause_accuracy_weighted_mean"] = float(acc_stats["mean"])

    return out
