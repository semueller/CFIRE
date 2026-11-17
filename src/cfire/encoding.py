"""Internal one-hot encoding of categorical input columns.

CFIRE mines the input as the black box sees it. A column of category codes (binary columns included) cannot be
described by intervals, so CFIRE encodes it internally: one indicator column per value observed in the fit data.
Numeric columns pass through unchanged. The rules stay interval rules over the encoded columns; the encoding is part
of the rule model (`DNFClassifier.encoding`), so the rule model takes the black box's input.

A value not observed in the fit data maps to all-zero indicators: it satisfies literals of the form
"indicator = 0" (the value is not in a set), never "indicator = 1".
"""
from __future__ import annotations

import numpy as np

FEATURE_TYPES = ("numeric", "categorical")


class Encoding:
    """feature_types: one of FEATURE_TYPES per input column; values: per input column the sorted list of its values
    (categorical) or None (numeric).

    Encoded columns, in input-column order: one per numeric column, one per value of a categorical column.
    `source[j]` is the input column of encoded column j, `value[j]` its category value (None for numeric)."""

    def __init__(self, feature_types, values):
        feature_types = list(feature_types)
        for t in feature_types:
            if t not in FEATURE_TYPES:
                raise ValueError(f"feature type must be one of {FEATURE_TYPES}, got {t!r}")
        if len(values) != len(feature_types):
            raise ValueError("one entry of values per feature type expected")
        self.feature_types = feature_types
        self.values = [None if t == "numeric" else [float(v) for v in vals]
                       for t, vals in zip(feature_types, values)]
        self.source, self.value = [], []
        for i, vals in enumerate(self.values):
            for v in ([None] if vals is None else vals):
                self.source.append(i)
                self.value.append(v)
        self.source = np.asarray(self.source, dtype=int)

    @classmethod
    def fit(cls, X, feature_types) -> "Encoding":
        """values of the categorical columns as observed in X"""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(feature_types):
            raise ValueError(f"feature_types has {len(feature_types)} entries, X has {X.shape[-1]} columns")
        values = [None if t == "numeric" else sorted(np.unique(X[:, i]).tolist())
                  for i, t in enumerate(feature_types)]
        return cls(feature_types, values)

    @property
    def n_input(self) -> int:
        return len(self.feature_types)

    @property
    def n_encoded(self) -> int:
        return len(self.source)

    @property
    def is_identity(self) -> bool:
        return all(t == "numeric" for t in self.feature_types)

    def transform(self, X) -> np.ndarray:
        """input columns -> encoded columns (float)"""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return self.transform(X[None])[0]
        if X.shape[1] != self.n_input:
            raise ValueError(f"expected {self.n_input} input columns, got {X.shape[1]}")
        out = np.empty((X.shape[0], self.n_encoded), dtype=float)
        for j, (i, v) in enumerate(zip(self.source, self.value)):
            out[:, j] = X[:, i] if v is None else (X[:, i] == v)
        return out

    def expand(self, M) -> np.ndarray:
        """per-input-column matrix (relevance bits or scores) -> encoded columns: the entry of a categorical column is
        copied to all of its indicators"""
        M = np.asarray(M)
        if M.shape[-1] != self.n_input:
            raise ValueError(f"expected {self.n_input} columns, got {M.shape[-1]}")
        return M[..., self.source]

    def groups(self) -> list[list[int]]:
        """encoded column indices per input column"""
        return [np.flatnonzero(self.source == i).tolist() for i in range(self.n_input)]

    def to_dict(self) -> dict:
        return {"feature_types": list(self.feature_types), "values": [None if v is None else list(v) for v in self.values]}

    @classmethod
    def from_dict(cls, d: dict) -> "Encoding":
        return cls(d["feature_types"], d["values"])

    def __eq__(self, other):
        return isinstance(other, Encoding) and self.to_dict() == other.to_dict()

    def __repr__(self):
        return f"Encoding({self.n_input} input columns -> {self.n_encoded} encoded)"
