import warnings

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from cfire import CFIRE, DNFClassifier, binarise_threshold
from cfire import pruning, pruning_metrics


@pytest.fixture(scope="module")
def iris():
    X, y = load_iris(return_X_y=True)
    X = (X - X.mean(0)) / X.std(0)
    model = LogisticRegression(max_iter=500).fit(X, y)
    y_pred = model.predict(X)
    # simple local attributions: coefficient x feature value for the predicted class
    E = model.coef_[y_pred] * X
    return X, y_pred, E


def fit(X, y, E, **kw):
    cf = CFIRE(None, E, None, **kw)
    cf._verbose = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cf.fit(X, y)
    return cf


def test_binarise_threshold():
    E = np.array([[1.0, 0.5, 0.005, -2.0], [0.0, 0.0, 0.0, 0.0]])
    B = binarise_threshold(E, 0.01)
    assert B.dtype == bool
    # row 0: max |e| = 2 -> normalised 0.5, 0.25, 0.0025, -1 -> only the first two are relevant
    assert B.tolist() == [[True, True, False, False], [False, False, False, False]]


def test_fit_returns_rule_model(iris):
    X, y, E = iris
    cf = fit(X, y, E, random_state=0)
    assert isinstance(cf.dnf, DNFClassifier)
    assert cf.dnf.n_classes == 3
    assert np.mean(cf(X) == y) > 0.5


def test_random_state_makes_fit_deterministic(iris):
    X, y, E = iris
    outs = []
    for s in range(3):
        np.random.seed(s)  # global state must not matter
        outs.append(str(fit(X, y, E, random_state=0).dnf.rules))
    assert len(set(outs)) == 1


def test_fixed_and_adaptive_depth_both_run(iris):
    X, y, E = iris
    assert fit(X, y, E, random_state=0).dnf is not None
    assert fit(X, y, E, random_state=0, max_dt_depth=7).dnf is not None


def test_accepts_tensor_like_explanations(iris):
    X, y, E = iris

    class FakeTensor:  # mimics torch.Tensor without importing torch
        def __init__(self, a): self.a = a
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.a

    cf = CFIRE(lambda f, X, Y: FakeTensor(E), None, None, random_state=0)
    cf._verbose = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cf.fit(X, y)
    assert isinstance(cf._explanations, np.ndarray)


def test_class_without_itemsets_warns(iris):
    X, y, E = iris
    E = E.copy()
    E[y == 2] = 0.0  # no relevant features for class 2 -> no closed itemsets
    cf = CFIRE(None, E, None, random_state=0)
    cf._verbose = False
    with pytest.warns(UserWarning, match="class 2"):
        cf.fit(X, y)


def test_safe_pruning_keeps_predictions(iris):
    X, y, E = iris
    cf = fit(X, y, E, random_state=0)
    before = cf.dnf.predict(X)
    metrics = pruning_metrics.compute_rule_metrics(cf, X)
    rules = pruning.prune_rules(cf.dnf.rules, pruning.decide_by_wins(metrics, win_threshold=0).to_remove)
    pruned = DNFClassifier(rules, "accuracy")
    pruned.compute_rule_performance(X, y)
    assert np.mean(pruned.predict(X) == y) == np.mean(before == y)


def test_dnf_json_roundtrip(iris):
    import json
    X, y, E = iris
    cf = fit(X, y, E, random_state=0)
    d = json.loads(json.dumps(cf.dnf.to_dict(feature_names=["a", "b", "c", "d"])))
    back = DNFClassifier.from_dict(d)
    assert np.array_equal(back.predict(X), cf.dnf.predict(X))
    assert d["feature_names"] == ["a", "b", "c", "d"]


def test_select_per_explainer(iris):
    from cfire import CFIRESelect, covered_accuracy
    X, y, E = iris
    rng = np.random.default_rng(0)
    sel = CFIRESelect({"good": E, "noise": rng.normal(size=E.shape)}, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel.fit(X, y)
    assert set(sel.models_) == {"good", "noise"} and sel.selected_ in sel.models_
    assert sel.scores_[sel.selected_] == max(sel.scores_.values())
    assert sel.scores_["good"] == covered_accuracy(sel.models_["good"].dnf, X, y)
    sel.reselect("accuracy")  # no refit
    assert sel.fitness == "accuracy" and np.array_equal(sel(X), sel.dnf.predict(X))
