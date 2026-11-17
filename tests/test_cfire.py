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


def test_binarisers():
    """threshold on row-normalised scores; top k by value or by magnitude, only positive scores relevant"""
    from cfire import binarise_topk, binarise_topk_abs
    E = np.array([[4.0, -5.0, 2.0, 0.02, 1.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0]])
    assert binarise_threshold(E, 0.01).tolist() == [[True, False, True, False, True], [False] * 5]  # 0.02/5 < 0.01
    assert binarise_topk(E, 2).tolist() == [[True, False, True, False, False], [False] * 5]
    # by magnitude the top 2 are -5 and 4; -5 is negative, so only 4 is relevant
    assert binarise_topk_abs(E, 2).tolist() == [[True, False, False, False, False], [False] * 5]
    assert binarise_topk(E, 0.4).tolist() == binarise_topk(E, 2).tolist()  # fraction: ceil(0.4 * 5) = 2
    with pytest.raises(ValueError):
        binarise_topk(E, 1.5)


def test_prune(iris):
    """safe keeps fit-data predictions; best stays within the tolerance; original untouched; removed + kept = original"""
    import copy
    from cfire import prune
    X, y, E = iris
    cf = fit(X, y, E, random_state=0)
    original = copy.deepcopy(cf.dnf.rules)
    safe = prune(cf.dnf, X, y, mode="safe")
    assert np.array_equal(safe.dnf.predict(X), cf.dnf.predict(X)) and safe.win_threshold == 0
    best = prune(cf.dnf, X, y, mode="best", tolerance=0.05)
    assert best.after["accuracy"] >= 0.95 * best.before["accuracy"] - 1e-12
    assert best.after["n_rules"] <= safe.after["n_rules"] <= safe.before["n_rules"]
    assert cf.dnf.rules == original  # not modified
    for res in (safe, best):
        kept = [r for c in res.dnf.rules for r in c]
        assert len(kept) + len(res.removed) == sum(len(c) for c in original)
        assert all(r["rule"] in original[r["class"]] and 0 <= r["share_covered_by_kept"] <= 1 for r in res.removed)
        json_safe = res.to_dict(); assert set(json_safe) >= {"mode", "win_threshold", "removed", "before", "after"}
    with pytest.raises(ValueError):
        prune(cf.dnf, X, y, mode="aggressive")


# ---- categorical input columns (internal one-hot encoding) ---------------------------------------------------------
@pytest.fixture(scope="module")
def categorical():
    """x0 numeric, colour with 4 values, b binary, x3 numeric; class 1 iff colour in {1, 2} and x0 > 0;
    the explanations mark x0 and colour as relevant"""
    rng = np.random.default_rng(0)
    n = 600
    x0, colour = rng.normal(size=n), rng.integers(0, 4, n).astype(float)
    X = np.c_[x0, colour, rng.integers(0, 2, n).astype(float), rng.normal(size=n)]
    y = (np.isin(colour, [1, 2]) & (x0 > 0)).astype(int)
    E = np.zeros_like(X)
    E[:, :2] = 1.0
    types = ["numeric", "categorical", "categorical", "numeric"]
    return X, y, E, types


def test_encoding():
    from cfire import Encoding
    X = np.array([[0.5, 2.0, 1.0], [1.5, 0.0, 0.0], [2.5, 2.0, 0.0]])
    enc = Encoding.fit(X, ["numeric", "categorical", "categorical"])
    assert enc.values == [None, [0.0, 2.0], [0.0, 1.0]] and enc.n_encoded == 5
    assert enc.transform(X).tolist() == [[0.5, 0, 1, 0, 1], [1.5, 1, 0, 1, 0], [2.5, 0, 1, 1, 0]]
    assert enc.transform(np.array([[0.0, 7.0, 1.0]])).tolist() == [[0.0, 0, 0, 0, 1]]  # unseen value: all zero
    B = np.array([[True, False, True]])
    assert enc.expand(B).tolist() == [[True, False, False, True, True]]
    assert enc.groups() == [[0], [1, 2], [3, 4]]
    assert Encoding.from_dict(enc.to_dict()) == enc
    with pytest.raises(ValueError):
        Encoding.fit(X, ["numeric", "categorical"])
    with pytest.raises(ValueError):
        Encoding.fit(X, ["numeric", "nominal", "numeric"])


def test_all_numeric_types_change_nothing(iris):
    X, y, E = iris
    plain = fit(X, y, E, random_state=0)
    typed = fit(X, y, E, random_state=0, feature_types=["numeric"] * 4)
    assert typed.encoding is None and typed.dnf.encoding is None
    assert typed.dnf.rules == plain.dnf.rules and typed.dnf.n_literals == plain.dnf.n_literals


def test_categorical_rules(categorical):
    from cfire import format_rules
    X, y, E, types = categorical
    cf = fit(X, y, E, random_state=0, max_dt_depth=5, feature_types=types)
    groups = cf.encoding.groups()
    # itemsets hold the indicators of a column all or none
    for coll in cf.frequent_nodes:
        for node in coll.nodes:
            items = set(int(i) for i in node.itemset)
            assert all(set(g) <= items or not set(g) & items for g in groups)
    assert np.array_equal(cf.dnf.predict(X), y)  # model-space input; the known rule is recovered
    text = format_rules(cf.dnf, feature_names=["x0", "colour", "b", "x3"])
    assert any("colour not in {0, 3}" in r for r in text[1])
    # literals on the indicators of colour count once per rule
    assert cf.dnf.n_literals == sum(len({int(cf.encoding.source[d]) for d, _ in r}) for c in cf.dnf.rules for r in c)


def test_categorical_unseen_value(categorical):
    X, y, E, types = categorical
    cf = fit(X, y, E, random_state=0, max_dt_depth=5, feature_types=types)
    # value 7 was not seen: all indicators 0; it satisfies only "not in S" literals, so no crash, a defined answer
    pred = cf.dnf.predict(np.array([[1.0, 7.0, 0.0, 0.0]]))
    enc_row = cf.encoding.transform(np.array([[1.0, 7.0, 0.0, 0.0]]))
    assert pred.shape == (1,) and enc_row[0, 1:5].sum() == 0


def test_categorical_serialisation_prune_select(categorical):
    import json
    from cfire import prune, CFIRESelect
    X, y, E, types = categorical
    cf = fit(X, y, E, random_state=0, max_dt_depth=5, feature_types=types)
    d = json.loads(json.dumps(cf.dnf.to_dict()))
    assert d["format"] == "cfire.DNFClassifier/2" and d["encoding"]["feature_types"] == types
    back = DNFClassifier.from_dict(d)
    assert back.encoding == cf.encoding and np.array_equal(back.predict(X), cf.dnf.predict(X))
    pruned = prune(cf.dnf, X, y, mode="safe")
    assert pruned.dnf.encoding == cf.encoding and np.array_equal(pruned.dnf.predict(X), cf.dnf.predict(X))
    sel = CFIRESelect({"e": E}, random_state=0, max_dt_depth=5, feature_types=types)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel.fit(X, y)
    assert np.array_equal(sel(X), cf.dnf.predict(X))


def test_explanations_must_match_data(iris):
    X, y, E = iris
    with pytest.raises(ValueError, match="shape"):
        fit(X, y, E[:, :3])
