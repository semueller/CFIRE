import enum

import numpy as np
from copy import deepcopy

import lxg
from lxg.util import load_pkl
from lxg.models import DNFClassifier

class DNFConsistencyReturnCode(enum.Enum):
    # correct, single rule
    CORRECT_SINGLE = 1
    # correct, multi rule consistent
    CORRECT_MULTI = 2
    # correct, multi rule inconsistent
    CORRECT_INCONSISTENT = 3
    # wrong, single rule
    WRONG_SINGLE = 4
    # wrong multi rule consistent
    WRONG_MULTI = 5
    # wrong, multi rule inconsistent
    WRONG_INCONSISTENT = 6
    # multiple classes returned
    AMBIGUOUS = 7
    # no rules applied
    NOT_COVERED = 8


def greedy_set_cover(X: set[int], F: list[set[int]]) -> list[int]:
    # X -> items that need to covered by transactions -> in F
    U = X.copy()
    C = []
    n_elements = len(U)
    n_iter = 0
    while len(U) > 0:
        _intersects = [len(U.intersection(f)) for f in F]
        # if max(_intersects) == 0:
        #     break
        _f_idx = np.argmax(_intersects)
        f = F[_f_idx]
        U = U - f
        C.append(_f_idx)
        n_iter += 1
        if n_iter > 2*n_elements:
            print(f"\n\n\n SETCOVERING TAKES LONG")
            print(X)
            print(U)
            print(C)

    return C #, coverage


def _rule_matches_batch(rule, X) -> list[bool]:
    return [_rule_matches(rule, x) for x in X]

def _rule_matching_fraction(rule, X) -> float:
    return np.mean(_rule_matches_batch(rule, X))

def novelty(rule, X, idxs_target) -> float:
    ''' novelty(r) = p(hb) - p(h)p(b)
    # p(hb) is accuracy; p(h) is fraction of target class; p(b) is support of rule in full dataset
    # -> rules that describe a single class very well and other samples very badly receive high scores
    :param r: rule
    :param X: full dataset
    :param idxs_target: idxs of targeted samples (eg. a certain class)
    :return: score between -0.25 <= novelty(r) <= 0.25
    '''
    support_target: list[bool] = _rule_matches_batch(rule, X[idxs_target])
    support_all: list[bool] = _rule_matches_batch(rule, X)
    return _novelty(n_samples=len(X), matching_idxs=np.array(support_all), target_idxs=np.array(support_target))

def _novelty(n_samples, matching_idxs, target_idxs) -> float:
    coverage_target = len(np.intersect1d(matching_idxs, target_idxs)) / n_samples
    coverage_all = len(matching_idxs) / n_samples
    weight = len(target_idxs) / n_samples
    return coverage_target - weight * coverage_all


def greedy_novelty_set_cover(n_samples, transactions, y_target) -> list[int]:

    uncovered = set(y_target)
    covered = set([])
    _transactions = [set(t) for t in transactions]
    result = []
    unused_transactions = np.arange(len(_transactions)).tolist()
    while len(uncovered) > 0:
        novelties = [_novelty(n_samples, _transactions[t]-covered, uncovered) for t in unused_transactions]
        _idx = unused_transactions[np.argmax(novelties)]
        unused_transactions.remove(_idx)
        uncovered -= _transactions[_idx]
        covered |= _transactions[_idx]#covered.union(_transactions[_idx])
        result.append(_idx)
    assert set(y_target) <= covered
    return result

def item_to_model_bin(itemsets, bins, n_dims, n_models) -> list[tuple[int, int, tuple]]:
    item_model_bin = []
    for m in range(n_models):
        for dim in range(n_dims):
            bins_idx = m * n_dims + dim
            if len(bins[bins_idx]) == 0:
                item_model_bin.append((m, dim, (np.nan, np.nan)))
            else:
                for bin in bins[bins_idx]:
                    item_model_bin.append((m, dim, tuple(bin)))
    _converted = []
    for items, support in itemsets:
        _converted.append([])
        for item in items:
            _converted[-1].append((item_model_bin[item]))
        # sort items by key=(model, dim, bin)
        _converted[-1] = sorted(_converted[-1], key=lambda x: (x[0], x[1], -x[2][0]))
        _converted[-1].insert(0, support)
    # sort itemsets by support
    _converted = sorted(_converted, key=lambda x: -x[0])
    return _converted

def _dnf_matches(r, x) -> bool:    # rule is dnf with potentially multiple terms [term([literal1 and literal2 ..]) or term2(..) ..]
    for term in r:
        if _rule_matches(term, x):
            return True
    return False

def _rule_matches(t, x) -> bool:
    # rule is dnf with potentially multiple terms [term([literal1 and literal2 ..]) or term2(..) ..]
    term_matches = True
    for literal in t:
        dim, (start, end) = literal
        if not start <= x[dim] <= end:
            term_matches = False
            break
    if term_matches:
        return True
    return False

def get_make_classification_exp_rules_params(fname):
    fname = fname.split('_')
    _make_classification_params = lxg.datasets._info_make_classification(fname[0][2:])  # cut off 't-' prefix
    fname = fname[1:]
    fname = [f.split('-') for f in fname]
    rules_params = {k: v for (k, v) in fname}
    rules_params['task'] = _make_classification_params
    rules_params['s'] = int(rules_params['s'].replace('[', '').replace(']', ''))
    rules_params['sc'] = bool(rules_params['sc'])
    return rules_params

def get_hypercube_exp_rules_params(fname):
    fname = fname.split('_')
    _make_classification_params = lxg.datasets._info_hypercube(fname[0][2:])  # cut off 't-' prefix
    fname = fname[1:]
    fname = [f.split('-') for f in fname]
    rules_params = {k: v for (k, v) in fname}
    rules_params['task'] = _make_classification_params
    rules_params['s'] = int(rules_params['s'].replace('[', '').replace(']', ''))
    rules_params['sc'] = bool(rules_params['sc'])
    return rules_params
def get_task_params(fname):
    fname = fname.split('_')
    task = fname[0][2:]
    fname = fname[1:]
    fname = [f.split('-') for f in fname]
    rules_params = {k: v for (k, v) in fname}
    rules_params['task'] = {
        'n_dims': lxg.datasets.__info_dim_classes[task][0],
        'n_classes': lxg.datasets.__info_dim_classes[task][1]
    }
    rules_params['s'] = int(rules_params['s'].replace('[', '').replace(']', ''))
    rules_params['sc'] = bool(rules_params['sc'])
    return rules_params


def _filter_fnames_params(paths, params):
    filtered_paths = deepcopy(paths)
    for k, v in params.items():
        if v is not None:
            if k == 'ex':  # epxl method
                if type(v) is not list:
                    v = [v]
                _f_ex = []
                _str_ex = lambda p: str(p.stem).split('_')[1]
                for ex in v:
                    _str_p = f'{k}-{ex}'
                    _f_ex.extend([p for p in filtered_paths if _str_p == _str_ex(p)])
                filtered_paths = [f for f in filtered_paths if f in _f_ex]

            else:
                _str_p = f's-[{v}]' if k == 's' else f'{k}-{v}'
                filtered_paths = [p for p in filtered_paths if _str_p in str(p)]
            if len(filtered_paths) == 0:
                print(f"all paths were removed when this arg was applied: {k}-{v}")
                return []
    return filtered_paths
def load_nn_dnfs(task, selection_params):
    from cfire import _variables_cfire as _variables


    nn_dnf_dir = _variables.get_nn_dnf_dir(task)
    pths = _filter_fnames_params([f for f in nn_dnf_dir.iterdir()], selection_params)
    nn_dnfs = []
    for pth in pths:
        nn_dnf = load_pkl(pth)
        if 'budgeted' in str(nn_dnf_dir):
            nc = len(nn_dnf)
            print(len(nn_dnf))
            _rules = []
            _meta = []
            for _class_dnfs in nn_dnf:
                if len(_class_dnfs) == 0: continue
                _class_dnfs_s = sorted(_class_dnfs, key=lambda x: len(x[1]))
                _rules.append(_class_dnfs_s[0][1])
                _meta.append(_class_dnfs_s[0][0])
            if len(_rules) < nc:
                continue
            dnf = DNFClassifier(_rules)
            dnf._meta_information = _meta
        else:
            dnf = DNFClassifier(nn_dnf)
        nn_dnfs.append(dnf)
    return nn_dnfs

def load_expl_rules(task, selection_params, modelclass=None):
    from cfire import _variables_cfire as _variables


    expl_rules_dir = _variables.get_expl_rule_dir(task, modelclass)
    fnames = [f for f in expl_rules_dir.iterdir()]
    pths = _filter_fnames_params(fnames, selection_params)
    pths = list(filter(lambda f: not 'itemset' in str(f), pths))
    expl_rules = []
    for pth in pths:
        _er = load_pkl(pth)
        expl_rules.append(_er)
    return expl_rules


def __preprocess_explanations(explanations, filtering):
    # normalize magnitude
    e = deepcopy(explanations)
    _max = np.expand_dims(np.max(np.abs(e), -1), -1)
    _max[np.argwhere(_max == 0)[:, 0], 0] = 1  # to avoid divide by zero
    e /= _max

    if type(filtering) == str:  # top_k, k either being an integer or expressed via fraction
        _k = eval(filtering.split('topkabs')[1] if 'abs' in filtering
                       else filtering.split('topk')[1])
        if _k < 1:
            _k = int(np.ceil(_k * e.shape[-1]))
        if 'abs' in filtering:
            _arg_sorted = np.argsort(np.abs(e), axis=-1)[:, :-_k]
        else:
            _arg_sorted = np.argsort(e, axis=-1)[:, :-_k]

        for i, _args in enumerate(_arg_sorted):
            e[i, _args] = 0.

        e[e < 0.] = 0.
    else:
        # e[np.abs(e) < significance_threshold] = 0. USING NEGATIVE VALUES IN EXPLS AS A PATTERN TO COMPUTE A DNF MAKES NO SENSE
        e[e < filtering] = 0.
    return e

def __preprocess_explanations_ext(
    explanations: np.ndarray,
    *,
    top_k: int | None = None,
    fraction: float | None = None,
    threshold: float | None = None,
    use_abs: bool = False
):
    """
    Scale each row to [-1, 1] then zero-out less relevant features using a filter, choose ONE:
    - `top_k`: keep k most relevant features per row
    - `fraction`: keep fraction of most relevant values per row
    - `threshold`: keep features with values above threshold

    Optional Flag:
    - `use_abs`: Consider absolute values, instead of priotizing positive ones
    """

    # make sure only one filter is applied
    if sum(arg is not None for arg in (top_k, fraction, threshold)) != 1:
        raise ValueError("Specify exactly one of top_k, fraction, or threshold")

    # normalize magnitude
    e = deepcopy(explanations)
    _max = np.expand_dims(np.max(np.abs(e), -1), -1)
    _max[np.argwhere(_max == 0)[:, 0], 0] = 1  # to avoid divide by zero
    e /= _max

    # TODO: we keep negatives for the normalization, good idea?!

    # apply value threshold, early return
    if threshold is not None:
        e[e < threshold] = 0. # TODO: should we also clamp here?
        return e

    # determine k value (via top_k or fraction)
    k = int(np.ceil(fraction * e.shape[-1]) if fraction is not None else top_k)

    # zero out all values expect top k
    order = np.argsort(np.abs(e) if use_abs else e, axis=-1)[:, :-k]
    for i, _args in enumerate(order):
        e[i, _args] = 0.

    # clamp negatives TODO: can this actually happen?
    e[e < 0.] = 0.

    return e

