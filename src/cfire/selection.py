"""Explainer selection: fit CFIRE once per local explainer and select a rule model per black box.

The CFIRE paper (2025) treats the local explainer as a variable: for each black box, rule models are computed from the
explanations of several explainers and the one that agrees best with the black box on the input data is selected.
CFIRESelect keeps every rule model and its score, so the selection can be redone with another criterion without
refitting.
"""
from __future__ import annotations

from typing import Callable, Mapping, Union

import numpy as np

from .core import CFIRE
from .rules import DNFClassifier

# fitness: (rule model, X, black-box labels Y) -> score, higher is better
FitnessFn = Callable[[DNFClassifier, np.ndarray, np.ndarray], float]
Fitness = Union[str, FitnessFn]  # name in FITNESS or a function


def covered_accuracy(dnf: DNFClassifier, X: np.ndarray, Y: np.ndarray) -> float:
    """precision; agreement with the black box on the samples the rule model covers (rejects excluded); 0 if none is covered.
    Criterion of the CFIRE paper (2025)."""
    pred = np.asarray(dnf.predict(X))
    covered = pred != -1
    return float(np.mean(pred[covered] == Y[covered])) if covered.any() else 0.0


def accuracy(dnf: DNFClassifier, X: np.ndarray, Y: np.ndarray) -> float:
    """agreement with the black box on all samples (rejects count as disagreement)"""
    return float(np.mean(np.asarray(dnf.predict(X)) == Y))


def coverage(dnf: DNFClassifier, X: np.ndarray, Y: np.ndarray | None = None) -> float:
    return float(np.mean(np.asarray(dnf.predict(X)) != -1))


FITNESS: dict[str, FitnessFn] = {"covered_accuracy": covered_accuracy, "accuracy": accuracy}


class CFIRESelect:
    """CFIRE over several explanation sets for the same black box and data.

    explanations: {name: (n_samples, n_features) attribution matrix} for the samples passed to fit
    fitness: name in FITNESS or callable (dnf, X, Y) -> float (higher is better); ties: first in `explanations` order
    cfire_kwargs: passed to every CFIRE (e.g. frequency_threshold, max_dt_depth, random_state)
    """

    def __init__(self, explanations: Mapping[str, np.ndarray], fitness: Fitness = "covered_accuracy",
                 **cfire_kwargs):
        if not explanations:
            raise ValueError("no explanations given")
        self.explanations = dict(explanations)
        self.fitness = fitness
        self.cfire_kwargs = cfire_kwargs
        self.models_: dict[str, CFIRE] = {}
        self.scores_: dict[str, float] = {}
        self.selected_: str | None = None

    def _fitness_fn(self, fitness: Fitness) -> FitnessFn:
        return FITNESS[fitness] if isinstance(fitness, str) else fitness

    def fit(self, X: np.ndarray, Y: np.ndarray) -> CFIRESelect:
        X, Y = np.asarray(X), np.asarray(Y)
        self._X, self._Y = X, Y
        for name, E in self.explanations.items():
            cf = CFIRE(None, np.asarray(E), None, **self.cfire_kwargs)
            cf._verbose = False
            cf.fit(X, Y)
            self.models_[name] = cf
        return self.reselect(self.fitness)

    def reselect(self, fitness: Fitness) -> CFIRESelect:
        """score all stored rule models with `fitness` on the fit data and select the best (no refit)"""
        f = self._fitness_fn(fitness)
        self.fitness = fitness
        self.scores_ = {n: f(cf.dnf, self._X, self._Y) for n, cf in self.models_.items()}
        self.selected_ = max(self.scores_, key=self.scores_.get)  # first maximum in insertion order
        return self

    @property
    def dnf(self) -> DNFClassifier:
        return self.models_[self.selected_].dnf

    def __call__(self, X: np.ndarray, explain: bool = False):
        return self.dnf(X, explain=explain)
