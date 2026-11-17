"""Readable rules (display only; the rule model is not changed).

Literals on the indicators of one categorical input column are printed as one condition on that column, as the rule
computes it: "colour = red" (indicator required to be 1), "colour not in {red, blue}" (indicators required to be 0),
"colour in {...all fit-data values}" (literals spanning [0, 1]).
Numeric literals print as intervals, closed on both sides as they are evaluated. Values are printed in the units of
the rule model's input (e.g. standardised); names for features and category values can be given.
"""
from __future__ import annotations

import numpy as np


def _fmt(v: float) -> str:
    return f"{float(v):.4g}"


def _interval(name: str, lo: float, hi: float) -> str:
    if np.isfinite(lo) and np.isfinite(hi):
        return f"{name} = {_fmt(lo)}" if lo == hi else f"{_fmt(lo)} <= {name} <= {_fmt(hi)}"
    if np.isfinite(lo):
        return f"{name} >= {_fmt(lo)}"
    if np.isfinite(hi):
        return f"{name} <= {_fmt(hi)}"
    return f"{name} any"


def _categorical(name, encoding, column, literals, names) -> str:
    """the literals on one column's indicators, as computed: an indicator required to be 1 -> "= v"; indicators
    required to be 0 -> "not in {...}"; only literals spanning [0, 1] -> "in {all fit-data values}" """
    values = encoding.values[column]
    k_of = {j: k for k, j in enumerate(np.flatnonzero(encoding.source == column))}
    required = sorted({k_of[d] for d, (lo, hi) in literals if lo > 0})
    excluded = sorted({k_of[d] for d, (lo, hi) in literals if hi < 1})
    label = lambda k: str(names.get(values[k], _fmt(values[k])))
    if required:  # at most one indicator is 1, so the other literals add nothing (two required: never true)
        return " and ".join(f"{name} = {label(k)}" for k in required)
    if excluded:
        return f"{name} not in {{{', '.join(label(k) for k in excluded)}}}"
    return f"{name} in {{{', '.join(label(k) for k in range(len(values)))}}}"


def format_rule(rule, encoding=None, feature_names=None, value_names=None) -> str:
    """one rule (conjunction of (column, (lo, hi)) literals) as text.
    feature_names: per input column; value_names: {input column: {value: name}} for categorical columns."""
    def name(i):
        return f"x{i}" if feature_names is None else str(feature_names[i])

    if encoding is None:
        return " and ".join(_interval(name(int(d)), lo, hi) for d, (lo, hi) in rule) or "true"
    by_col: dict[int, list] = {}
    for d, b in rule:
        by_col.setdefault(int(encoding.source[int(d)]), []).append((int(d), b))
    parts = []
    for col in sorted(by_col):
        lits = by_col[col]
        if encoding.feature_types[col] == "numeric":
            parts.extend(_interval(name(col), lo, hi) for _, (lo, hi) in lits)
            continue
        parts.append(_categorical(name(col), encoding, col, lits, (value_names or {}).get(col, {})))
    return " and ".join(parts) or "true"


def format_rules(dnf, feature_names=None, value_names=None) -> list[list[str]]:
    """all rules of a DNFClassifier as text, per class"""
    enc = getattr(dnf, "encoding", None)
    return [[format_rule(r, enc, feature_names, value_names) for r in class_rules] for class_rules in dnf.rules]
