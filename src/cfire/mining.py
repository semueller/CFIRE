from __future__ import annotations # python 3.10 +
import numpy as np
from functools import reduce
from itertools import chain
from copy import deepcopy



import sys, warnings

from lxg.models import DNFClassifier

from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.model_selection import GridSearchCV, ParameterGrid

from collections import OrderedDict

from collections import OrderedDict


class OrderedSet:
    """
    OrderedSet: A drop-in replacement for Python's built-in set class that maintains the order of elements.
    This class provides all the functionalities of the native set class, including typical binary operations
    (union, intersection, difference, symmetric difference) and set comparison methods (subset, superset).
    The elements are stored in the order they are added.

    Author: [python] @ ChatGPT4 developed by OpenAI, tweaked with more safety checks by Claude Sonnett
    """
    def __init__(self, iterable=None):
        self._dict = OrderedDict()
        if iterable is not None:
            self.update(iterable)

    def add(self, element):
        self._dict[element] = None

    def update(self, iterable):
        for element in iterable:
            self.add(element)

    def discard(self, element):
        self._dict.pop(element, None)

    def remove(self, element):
        if element in self._dict:
            del self._dict[element]
        else:
            raise KeyError(element)

    def pop(self):
        if not self._dict:
            raise KeyError("pop from an empty set")
        return self._dict.popitem(last=False)[0]

    def clear(self):
        self._dict.clear()

    def __contains__(self, element):
        return element in self._dict

    def __iter__(self):
        return iter(self._dict.keys())

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return f"{self.__class__.__name__}({list(self._dict.keys())})"

    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        return all(element in other for element in self)

    def __lt__(self, other):
        return self <= other and self != other

    def __ge__(self, other):
        return all(element in self for element in other)

    def __gt__(self, other):
        return self >= other and self != other

    def __add__(self, other):
        if isinstance(other, (OrderedSet, set)):
            new_set = OrderedSet(self)
            new_set.update(other)
            return new_set
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (OrderedSet, set)):
            return OrderedSet(item for item in self if item not in other)
        return NotImplemented

    def __and__(self, other):
        if isinstance(other, (OrderedSet, set)):
            return OrderedSet(item for item in self if item in other)
        return NotImplemented

    def __xor__(self, other):
        if isinstance(other, (OrderedSet, set)):
            return OrderedSet(item for item in self if item not in other) | \
                   OrderedSet(item for item in other if item not in self)
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return list(self) == list(other)
        if isinstance(other, set):
            return set(self) == other
        return False

    def __or__(self, other):
        return self.__add__(other)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return OrderedSet(list(self._dict.keys())[index])
        return list(self._dict.keys())[index]

    def intersection(self, *others):
        result = OrderedSet(self)
        for other in others:
            result &= other
        return result

    def union(self, *others):
        result = OrderedSet(self)
        for other in others:
            result |= other
        return result

    def difference(self, *others):
        result = OrderedSet(self)
        for other in others:
            result -= other
        return result

    def symmetric_difference(self, other):
        return self ^ other

    def issubset(self, other):
        return self <= other

    def issuperset(self, other):
        return self >= other

    def copy(self):
        return OrderedSet(self)

    def isdisjoint(self, other):
        return not any(element in self for element in other)

    # Add an optional parameter for order-aware equality
    def equals(self, other, order_matters: bool = False):
        """
        Compare this OrderedSet with another set-like object.

        :param other: The object to compare with
        :param order_matters: If True, the order of elements must match
        :return: True if the sets are equal, False otherwise
        """
        if order_matters and isinstance(other, OrderedSet):
            return list(self) == list(other)
        return self == other

class ItemsetNode():
    def __init__(self, parent : ItemsetNode | None =None, itemset=None, compute_rules=None, rel_items_global: OrderedSet=None, n_samples_global: int =None):
        '''
        :param parent:
        :param itemset:
        :param compute_rules: if anything other than None rules will be computed
        :param rel_items_global:
        :param n_samples_global:
        '''
        self.parent = parent
        self.depth = parent.depth + 1 if parent is not None else 0
        self.left: ItemsetNode | None = None
        self.right: ItemsetNode | None = None

        self.complexity_limit = 8
        self.rel_items_global: OrderedSet = rel_items_global
        self.n_samples_global: int = n_samples_global

        self.gely_state = None

        self.itemset = itemset
        self.is_frequent = None

        self.dnf: DNFClassifier = None
        self.simple_dnf = None
        self.dt = None
        self._dnfs = []
        self._dts = []
        self.data_target = None
        self.data_other = None
        self._compute_rules_fn = self._compute_rules_min_max if compute_rules is None else compute_rules
        self.support = np.nan  # on data_target; absolute number
        self.accuracy = np.nan
        self.precision = np.nan
        self.recall = np.nan
        self.f1 = np.nan
        self.coverage_ratio = np.nan
        #max possible number of rules in binary tree of size complexity_limit and complexity_limit many rules
        self.complexity_factor = np.nan # 1 - np.log2(self.dnf.n_literals) / np.log2(2**self.complexity_limit * self.complexity_limit)
        self.completeness_factor = np.nan

    def get_sibling(self, child):
        # siblings = [c for c in self.children if c != child]
        # assert len(self.children) == len(siblings)+1
        # return siblings
        if child == self.right:
            return self.left
        elif child == self.left:
            return self.right
        else:
            print("You're lost, child!")

    def add_child(self, child, direction=None):
        # self.children.append(child)
        # return
        if direction == 'left' and self.left is None:
            self.left = child
        elif direction == 'right' and self.right is None:
            self.right = child

    def get_frequent_children(self):
        r = self.get_children()
        fr = [n for n in r if n.is_frequent]
        return fr

    def get_children(self, r=None):
        if r is None:
            r = []
        if self.left is not None:
            r.append(self.left)
            self.left.get_children(r)
        if self.right is not None:
            r.append(self.right)
            self.right.get_children(r)
        return r

    def compute_purity(self, x_target, x_other):
        n_samples = len(x_target) + len(x_other)
        tp: list = self.dnf(x_target)
        fp: list = self.dnf(x_other)
        ntp, nfp = np.sum(tp), np.sum(fp)
        self.accuracy =(ntp+(len(x_other)-nfp))/n_samples
        self.support = ntp
        self.precision = ntp / (ntp + nfp) if ntp+nfp > 0 else 0
        self.recall = ntp/len(x_target)

    def compute_coverage_ratio(self, ):
        if self.accuracy is np.nan:
            self.compute_purity(self.data_target, self.data_other)
        return self.accuracy


    def _get_upper_complexity_bound(self, _n=None):
        # maximum number of literals
        if _n is None:
            # _n = len(self.data_target)
            _n = self.support

        return 2**self.complexity_limit * self.complexity_limit

        # if self.dt is not None:
        #
        #     def max_depth_for_class(tree, class_of_interest):
        #         """
        #         Find the maximum depth of a node leading to a specific class in a sklearn decision tree.
        #
        #         :param tree: A fitted sklearn DecisionTreeClassifier
        #         :param class_of_interest: The class label we're interested in
        #         :return: The maximum depth of a node predicting the class of interest
        #         """
        #
        #         def traverse_tree(node, depth):
        #             if tree.tree_.feature[node] == -2:  # Leaf node
        #                 if np.argmax(tree.tree_.value[node]) == class_of_interest:
        #                     return depth
        #                 return -1
        #
        #             left_child = tree.tree_.children_left[node]
        #             right_child = tree.tree_.children_right[node]
        #
        #             left_depth = traverse_tree(left_child, depth + 1)
        #             right_depth = traverse_tree(right_child, depth + 1)
        #
        #             return max(left_depth, right_depth)
        #
        #         return traverse_tree(0, 0)
        #     mdc = max_depth_for_class(self.dt, 1)
        #     if mdc < 0:
        #         print(f"CLASS NOT IN TREE: {mdc}")
        #     _max_depth = max(1, mdc)
        #     # ORIGINALLY:
        #     # return (self.dt.get_depth()+1) * _n
        #
        #     # lose upper bound: each sample is covered by its own path -> _max_depth * _n_samples
        #     # SLIGHTLY TIGHTER:
        #     # _n_literals_max = _max_depth * _n
        #
        #     # but this is a binary tree, so a given depth can only produce so many leafs/ paths
        #     # up to 2**depth paths of up to length depth appended to conjunction of all items
        #     _n_terms_max = (2**_max_depth) * _max_depth
        #
        #     return _n_terms_max
        #
        # else:
        #     return len(self.rel_items_global)
        #     # return len(self.itemset) * _n

    def compute_complexity_factor(self, ):
        cb = self._get_upper_complexity_bound()
        # self.complexity_factor = (cb-self.dnf.n_literals)/cb
        if self.dnf.n_literals > 0:
            self.complexity_factor = 1 - np.log2(self.dnf.n_literals) / np.log2(cb)
        else:
            self.complexity_factor = np.inf
        # self.complexity_factor = self.complexity_factor ** self.complexity_limit  # MAGIC
        # coverage_weighting = self.support / self.n_samples_global
        # self.complexity_factor = complexity * coverage_weighting
        # assert 0 <= self.complexity_factor <=1, f"complexity factor was {self.complexity_factor}"
        return self.complexity_factor


    def __get_unique_dims(self):
        _dims = []
        rules = self.dnf[0]
        for clause in rules:
            for term in clause:
                _dims.append(term[0])
        return np.unique(_dims)

    def compute_completeness_factor(self, ):
        n_dims = len(self.__get_unique_dims())
        self.completeness_factor = n_dims / len(self.rel_items_global)
        assert 0 <= self.completeness_factor <= 1, f"completeness factor was {self.completeness_factor}"
        return self.completeness_factor

    def score(self, acc_weight=0.5, cx_weight=0.5, cs_weight=0.):
        assert np.nan not in [self.accuracy, self.complexity_factor, self.completeness_factor]
        return (acc_weight * self.accuracy +
                cx_weight * self.complexity_factor +
                cs_weight * self.completeness_factor)


    def compute_initial_rules(self, X):
        # compute rules only based on itemset, ie not "discriminatory"
        if self.data_target is None:
            self.data_target = deepcopy(X)
        rules = self._compute_rules_fn(X, self.itemset)
        rules = [rules]  # wrap
        dnf = DNFClassifier(rules)
        self.simple_dnf = dnf
        self.dnf = dnf
        self.support = np.sum(dnf(X))
        self.compute_complexity_factor()
        return self.simple_dnf

    def compute_statistics(self, x_target, x_other):

        if not self.is_frequent:
            return
        if self.data_target is None:
            self.data_target = deepcopy(x_target)
        if self.dnf is None:
            self.compute_initial_rules(x_target)
        self.data_other = x_other.copy()
        self.compute_purity(x_target, x_other)
        self.compute_coverage_ratio()
        self.compute_complexity_factor()
        self.compute_completeness_factor()
        return self.accuracy

    @staticmethod
    def _compute_rules_min_max(X, itemset):
        dims = sorted(list(itemset))
        _picked_dims = X[:, dims]

        conjunction_on_data = [(dim, (_min, _max)) for dim, _min, _max in
                   zip(dims, np.min(_picked_dims, axis=0), np.max(_picked_dims, axis=0))]

        return [conjunction_on_data]

    def compute_rules_dt(self, dt_kwargs, data_target=None, data_other=None):
        self._compute_rules_dt(dt_kwargs=dt_kwargs, data_target=data_target, data_other=data_other)
        # for d in range(1, self.complexity_limit+1):
        #     self._compute_rules_dt(dt_kwargs=dict(max_depth=d))

    def _compute_rules_dt(self, dt_kwargs=None, data_target=None, data_other=None):
        # raise NotImplementedError
        # print("TODO, fit multiple trees of various depths for easier comparison later")

        if data_target is None:
            data_target = self.data_target
        if data_other is None:
            data_other = self.data_other

        if self.simple_dnf is None:
            self.compute_initial_rules(data_target)
    
        def get_dims_conjunction(conjunction):
            return np.unique([d for (d, (_, _)) in conjunction])

        def __conjunction_matches_sample(sample, conjunction):
            matches = True
            for term in conjunction:
                dim, (start, end) = term
                if not start <= sample[dim] <= end:
                    # clause cannot be fulfilled anymore
                    matches = False
                    break
            return matches

        def simplify_conjunction(literals):
            # for all terms that share a dimension, take the tightest bounds
            if len(literals) == 1:
                return literals
            simplified_conjunction = []
            unique_dims = np.unique([t[0] for t in literals])
            for d in unique_dims:
                d_literals = [ll for ll in literals if ll[0] == d]
                if len(d_literals) == 1:
                    simplified_conjunction.append(d_literals[0])
                    continue
                # sanity check
                _mi, _ma = min([ll[1][0] for ll in d_literals]), max([ll[1][1] for ll in d_literals])
                if _mi > _ma: raise ValueError
                # use the highest minimum and the lowest maximum for same dim
                _lb, _rb = max([ll[1][0] for ll in d_literals]), min([ll[1][1] for ll in d_literals])
                simplified_conjunction.append((d, (_lb, _rb)))
                assert np.abs(_lb) != np.inf and np.abs(_rb) != np.inf
            return simplified_conjunction
        
        _new_dnfs = [[]]
        c = 0
        assert len(self.simple_dnf.rules) == 1  # TODO, this ensures we only call this function once after initialising rules
        for conjunction in self.simple_dnf.rules[c]:  # [0] to access class rules for the one class
            # add conjunctive term to class dnf
            rule_applicable = []
            X = np.vstack([data_target, data_other])
            Y = np.array([c]*len(data_target) + [1]*len(data_other))
            for x in X:
                rule_applicable.append(__conjunction_matches_sample(x, conjunction))

            # term_applicable = np.array(term_applicable)  # support
            if not np.any(rule_applicable):
                ''' 
                TERM DOESNT MATCH ANYTHING, we discard it
                this should never happen because the explanation rules were computed on this very dataset
                '''
                raise ValueError

            rule_applicable = np.argwhere(rule_applicable).squeeze()

            # split X, Y into c vs. rest set
            _Y_app = Y[rule_applicable]
            if np.all(_Y_app == c):  # RULE IS CORRECT
                _new_dnfs[-1].append(conjunction); continue
            _Y_app_neg = np.argwhere(_Y_app != 0).reshape(-1)  # false positives

            if len(_Y_app_neg) == 0:  # if no errors, keep old conjunction/literal as is

                _new_dnfs[-1].append(conjunction); continue

            else:  # if the conjunction led to false positives, refine with DT

                _Y_app_pos = np.argwhere(_Y_app == c).reshape(-1)  # true positives
                _Y_new = np.zeros(len(rule_applicable))
                _Y_new[_Y_app_pos] = 1  # new c vs. rest labels
                _X_app = X[rule_applicable]  # select applicable data points
                # select dims covered by conjunction
                d = get_dims_conjunction(conjunction)
                _X_app = _X_app[:, d]

                assert len(_X_app) == len(_Y_new)

                if dt_kwargs is not None and 'max_depth' in dt_kwargs.keys():
                    _max_depth = dt_kwargs['max_depth']
                else:
                    # _max_depth = min([max(self.complexity_limit - len(self.itemset), 2), self.complexity_limit])
                    _max_depth = max(self.complexity_limit - len(self.itemset), 2)

                __n_ones = sum(_Y_new)
                __n_zeros = len(_Y_new) - __n_ones

                if __n_zeros <= 2 or __n_ones <= 2:  # don't bother
                    _new_dnfs[-1].append(conjunction); continue


                def _fit_tree(pg, cv, X, Y):

                    grid_search = GridSearchCV(DT(), pg, cv=int(cv), scoring='accuracy')
                    # Fit the GridSearchCV object to the data
                    grid_search.fit(X, Y)
                    dt = grid_search.best_estimator_
                    return dt

                _cv = 5
                if __n_ones < 5 or __n_zeros < 5:  # too few samples per class
                    _cv = min(__n_ones, __n_zeros)-1
                    if _cv < 2:
                        _cv = 2
                    _max_depth = min(_max_depth, 3)
                if len(_X_app) <= 10:  # too few samples to really bother
                    _cv = min(_cv, 2)
                    _max_depth = 2
                param_grid = {'max_depth': range(1, _max_depth)}
                                  # 'class_weight': [{0: 1, 1:2}]}
                if dt_kwargs is not None and 'class_weight' in dt_kwargs.keys():
                    param_grid['class_weight'] = dt_kwargs['class_weight']
                dt = _fit_tree(param_grid, _cv, _X_app, _Y_new)
                _r = DNFClassifier.from_DT(dt)
                if len(_r.rules[1]) == 0:  # no rules were found for target class
                    param_grid.update({'class_weight': ['balanced']})
                    dt = _fit_tree(param_grid, _cv, _X_app, _Y_new)
                    _r = DNFClassifier.from_DT(dt)
                acc_dt = dt.score(_X_app, _Y_app)
                self.dt = dt
                self._dts.append(dt)

                acc_dnf = np.mean(_r(_X_app) == _Y_app)
                if not acc_dnf == acc_dt:
                    print("dnf deviates from tree")
                if not np.isclose(acc_dnf, acc_dt):
                    print(" ... and it's not even close!")
                assert acc_dnf == acc_dt  # check that our rule classifier behaves like the tree empirically
                new_c_dnf = _r[1]  # 1 because _Y[_term_pos] was set to 1

                # if new_c_dnf == [[]]:  # we weren't able to produce any rules, keep old ones (as bad as they may be?)
                #     # this happened eg where we only looked at a single dimension, all samples had the same value ([0.32698354])
                #     # but we had multiple labels -> so this would've been a garbage rule anyways. <-> there are cases
                #     # where it is okay not to have a rule
                #     _new_dnfs[-1].append(conjunction)
                #     continue
                ## FOR EACH PATH THAT IS TAKEN IN THE TREE, WE NEED TO PREPEND THE CONJUNCTION AND APPEND THE PATH
                # map rules back to original dims d selected in conjunction
                _extended_conjunctions_mapped = []
                for new_conjunction in new_c_dnf:
                    new_conjunction_mapped = []
                    if new_conjunction == []:
                        continue  # idk, sometimes there is an empty list todo
                    new_conjunction_mapped.extend(conjunction)  # prefix decision path with conjunction
                    for term in new_conjunction:
                        (_d, (mi, ma)) = term
                        new_conjunction_mapped.append((d[_d], (mi, ma)))

                    assert np.all([d[1][0]<=d[1][1] for d in new_conjunction_mapped])
                    _simplfied_conjunction = simplify_conjunction(new_conjunction_mapped)
                    if not np.all([d[1][0]<=d[1][1] for d in _simplfied_conjunction]):
                        print(_simplfied_conjunction)
                    assert np.all([d[1][0]<=d[1][1] for d in _simplfied_conjunction])
                    #_new_dnfs[-1 = current_class][current term = -1]
                    _new_dnfs[-1].append(_simplfied_conjunction)

        if len(_new_dnfs[-1]) == 0:  # this can happen when we fit a tree but it didn't have a leaf for positive class
            _new_dnfs[-1].append([(-1, (np.nan, np.nan))])

        assert len(_new_dnfs) == 1
        new_dnf = DNFClassifier(_new_dnfs, self.dnf.tie_break) # TODO: why do we 'keep' tie break here?
        self.dnf = new_dnf
        self._dnfs.append(new_dnf)
        self.compute_complexity_factor()

class recursionlimit:
    def __init__(self, limit):
        self.limit = limit

    def __enter__(self):
        self.old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(self.limit)

    def __exit__(self, type, value, tb):
        sys.setrecursionlimit(self.old_limit)

def _is_frequent(pattern, D, threshold):
    support_idxs = []
    if len(pattern) == 0:
        return False, 0, support_idxs
    for i, d in enumerate(D):
        if pattern.issubset(d):
            support_idxs.append(i)
    is_frequent = len(support_idxs) >= threshold
    return is_frequent, len(support_idxs), support_idxs


def _it(X, D) -> OrderedSet:
    '''return: all transactions that contain all items in X'''

    tids = []

    for x in X:  # set of items
        _tids = []
        for tid, y in enumerate(D):
            if x in y:
                _tids.append(tid)
        if len(_tids) > 0:
            tids.append(OrderedSet(_tids))

    if len(tids) == 0:
        return OrderedSet()
    _tids_all_x_appear_in = OrderedSet.intersection(*tids)
    return _tids_all_x_appear_in


def _ti(Y, D) -> OrderedSet:
    '''return: all items common to all transactions in Y'''
    '''_ti(_it(X)) -> can never be empty (if |Y| > 0), because at least all X have to be returned'''
    if len(Y) == 0:
        return OrderedSet()
    iids = [D[tid] for tid in Y]
    return OrderedSet.intersection(*iids)

def get_default_closure(Database):
    return lambda X: _ti(_it(X, Database), Database)

## frequency check and closure operator using binary matrix instead of sets and list

def _ti_binary_matrix(Y, B) -> set:
    iids = B[list(Y), :]
    intersected = reduce(lambda a,b: np.bitwise_and(a, b), list(iids))
    intersected = np.argwhere(intersected).squeeze()
    if len(intersected.shape) == 0:
        intersected = np.expand_dims(intersected, 0)
    intersected = list(intersected)
    return set(intersected)

def _it_binary_matrix(X, B) -> set:
    tids = [set(list(np.argwhere(B[:, x]).squeeze(1))) for x in X]
    return set.intersection(*tids)


def _is_frequent_binary(itemset, B, threshold):
    if len(itemset) == 0:
        return False, 0
    _itemset_binary = np.zeros(B.shape[1], dtype=int)
    _itemset_binary[list(itemset)] = 1
    match = B @ _itemset_binary
    match = match == len(itemset)
    n_occurences = np.sum(match)
    is_frequent = n_occurences >= threshold
    return is_frequent, n_occurences


def _check_applicable_any(rule: list[int, tuple[float, float]], data: np.ndarray) -> bool:
    for x in data:
        applicable = True
        for (d, (mi, ma)) in rule:  # for term in DNF rule
            if not (mi < x[d] <= ma):
                applicable = False
        if applicable:
            return True
    return False  # rule applies to no datapoint


def is_clean(dims, X, Y, target_label) -> bool:
    _dims = None
    n_dims = X.shape[1]
    if np.any(dims > n_dims): # multiple models/ explanations used, hence dims can appear multiple times
        _dims = [d%n_dims for d in dims]
    else:
        _dims = dims
    _dims = np.array(_dims)
    selected_data = X[Y==target_label, _dims]

    _data_rule = [(d, (mi-1e-15, ma)) for d, mi, ma in zip( _dims,
        np.min(selected_data, 0), np.max(selected_data, 0)
    )]
    # return True if rule applies to no datapoint from another class
    return not _check_applicable_any(_data_rule, X[Y != target_label])


def find_max_pos_and_min_neg(t, o, d):
    raise NotImplementedError
    # TODO: if t or o contains no positive/ negative values in any of d then min/max will be None -> that's not right,
    #   we need to filter differently
    """
    For each dimension in d, find the maximal positive value in x_synth_target[:, dim]
    that is still lower than the minimal positive value in x_synth_other[:, dim].
    Similarly, repeat for negative values.

    Parameters:
    x_synth_target (np.ndarray): The subset of perturbed samples where labels are True.
    x_synth_other (np.ndarray): The subset of perturbed samples where labels are False.
    d (list or np.ndarray): A set of dimensions to check.

    Returns:
    lists[tuples]: two lists of tuples containing the maximal positive and minimal negative values for each dimension, list 1 for taget t, list 2 for other o
    """
    results = {}
    # _target_samples, _other_samples
    _ts, _os = [], []
    for dim in d:
        # Get positive and negative values from target and other
        target_positive = t[:, dim][t[:, dim] > 0]
        target_negative = t[:, dim][t[:, dim] <= 0]
        other_positive = o[:, dim][o[:, dim] > 0]
        other_negative = o[:, dim][o[:, dim] <= 0]

        # Find max positive in target that's lower than min positive in other
        if len(target_positive) > 0 and len(other_positive) > 0:
            max_pos_target = np.max(target_positive[target_positive < np.min(other_positive)])
            min_pos_other = np.min(other_positive)
        else:
            max_pos_target, min_pos_other = None, None  # No valid comparison

        # Find min negative in target that's greater than max negative in other
        if len(target_negative) > 0 and len(other_negative) > 0:
            min_neg_target = np.max(target_negative[target_negative > np.max(other_negative)])
            max_neg_other = np.max(other_negative)
        else:
            min_neg_target, max_neg_other = None, None  # No valid comparison

        _ts.append((min_neg_target, max_pos_target))
        _os.append((max_neg_other, min_pos_other))

    return _ts, _os


def generate_perturbations(x, d, n, low_val, high_val):
    perturbation = np.random.uniform(low_val, high_val, size=(n, len(d)))
    signs = np.random.choice([-1, 1], size=(n, len(d)))
    perturbation *= signs
    perturbed = np.tile(x, (n, 1))
    perturbed[:, d] += perturbation
    return perturbed


def generate_synthetic_data(_target, _other, callable, dims):
    '''

    :param _target: data where callable returns True
    :param _other:  data where callable returns False
    :param callable: takes input data and returns True or False
    :param dims: dimensions of input data that should be perturbed
    :return:
            _synth_target: synthetitc data that is close to _target and callable return True
            _synth_other: synthetic data that is close to _target and callable returns False
    '''
    _synth_target, _synth_other = [], []
    # n_samples = min(256, max(64, 2 ** (len(dims) + 1)))
    n_samples = min(128, max(16, 2 ** (len(dims) + 1)))
    n_max_iter = 10
    for x in _target:
        labels = np.array([True])
        _perturbed_target = []
        _perturbed_other = []
        hi = 0.5
        lo = 0
        inc = hi - lo
        _i = 0
        _true_i = 0
        while np.all(labels) and _i <= n_max_iter:  # as long as we only find positive examples, we keep expanding
            perturbed = generate_perturbations(x, dims, n_samples, low_val=lo, high_val=hi)
            labels: np.array[bool] = callable(perturbed)
            if np.any(labels) and not (len(_perturbed_other) > 0 and len(_perturbed_other[-1]) > 0):
                # if we already had negative samples in the last iteration, don't introduce more positive samples
                perturbed_target = perturbed[labels]
            else:
                perturbed_target = []
            _perturbed_target.append(perturbed_target)
            if not np.all(labels):
                perturbed_other = perturbed[np.logical_not(labels)]  # ~labels
            else:
                perturbed_other = []
            _perturbed_other.append(perturbed_other)
            if len(perturbed_other) == 0:  # we haven't found any negative samples
                hi += inc
                lo += inc
            else:
                # we found negative samples, will increase hi/lo less
                hi += inc*0.5
                lo += inc*0.5
                if len(_perturbed_other) > 1 and len(_perturbed_other[-2]) > 0:
                    break  # we have added perturbed samples twice, quit
                else:
                    # we have added perturbed samples only once, make sure to make another iteration by setting _i low enough
                    _i = n_max_iter-1
            _i += 1
            _true_i += 1
            print(_true_i)

        _perturbed_target = [p for p in _perturbed_target if len(p) > 0]
        if len(_perturbed_target) > 0:
            _perturbed_target = np.vstack(_perturbed_target)
            _synth_target.append(_perturbed_target)
        _perturbed_other = [p for p in _perturbed_other if len(p) > 0]
        if len(_perturbed_other) > 0:
            _perturbed_other = np.vstack(_perturbed_other)
            _synth_other.append(_perturbed_other)

        # if len(_perturbed_target) > 0 and len(_perturbed_other) > 0:
        #     _pt_shifted = _perturbed_target - x
        #     _po_shifted = _perturbed_other - x
        #     # search for min max _shift_ wrt original x
        #     dims_bounds_target, dims_bounds_other = find_max_pos_and_min_neg(_pt_shifted, _po_shifted, dims)
        #     dbt_pos = [d[1] for d in dims_bounds_target]
        #     dbt_neg = [d[0] for d in dims_bounds_target]
        #     x_synth_target = np.tile(x, (2, 1))
        #     x_synth_target[0, dims] += dbt_pos
        #     x_synth_target[1, dims] += dbt_neg
        #     _synth_target.append(x_synth_target)
        #
        #     dbo_pos = [d[1] for d in dims_bounds_other]
        #     dbo_neg = [d[0] for d in dims_bounds_other]
        #     x_synth_other = np.tile(x, (2, 1))
        #     x_synth_other[0, dims] += dbo_pos
        #     x_synth_other[1, dims] += dbo_neg
        #     _synth_other.append(x_synth_other)
        #
        #
        # elif len(_perturbed_target) > 0:
        #     # only positive samples, make box as big as possible
        #     dbt_pos = np.max(_perturbed_target[:, dims], axis=0)
        #     dbt_neg = np.min(_perturbed_target[:, dims], axis=0)
        #     x_synth_target = np.tile(x, (2, 1))
        #     x_synth_target[0, dims] += dbt_pos
        #     x_synth_target[1, dims] += dbt_neg
        #     _synth_target.append(x_synth_target)
        #
        # else:  # len(_perturbed_other) > 0
        #     # only negative samples
        #     dbo_pos = np.max(_perturbed_target[:, dims], axis=0)
        #     dbo_neg = np.min(_perturbed_target[:, dims], axis=0)
        #     x_synth_other = np.tile(x, (2, 1))
        #     x_synth_other[0, dims] += dbo_pos
        #     x_synth_other[1, dims] += dbo_neg
        #     _synth_other.append(x_synth_other)

    if len(_synth_target) > 0:
        _synth_target = np.vstack(_synth_target)
    if len(_synth_other) > 0:
        _synth_other = np.vstack(_synth_other)
    return _synth_target, _synth_other

def list_closed_no_rules(C, N, i, t, D, I, T, results, closure=None, parent: ItemsetNode=None,
                         subset_test=False, direction=None, model_callable=None):
    if parent is None:
        raise ValueError

    if closure is None:
        closure = lambda X: _ti(_it(X, D), D)

    idx = np.argwhere(I == i).squeeze()  # {k in I\C : k >= i}
    X = I[idx:]
    X = OrderedSet(X) - C

    if len(X) > 0:
        _i_prime = X[0]  # min(X) # get nextitem
        _C_prime: OrderedSet = closure(C.union({_i_prime}))

        # we could probably optimize the size of D and _data_target because support of subsequent itemsets is
        # always a subset of the support of previous itemsets
        is_frequent, count, support_idxs = _is_frequent(_C_prime, D, t)

        newNode = ItemsetNode(parent=parent, itemset=_C_prime,
                              rel_items_global=parent.rel_items_global, n_samples_global=parent.n_samples_global)
        newNode.is_frequent = is_frequent
        newNode.idxs = support_idxs

        if is_frequent:

            if len(_C_prime.intersection(N)) == 0:  # and check_class_variable(_C_prime, class_item_ids):
                # print(sorted(_C_prime), count)
                parent.add_child(newNode, 'left')
                results.append((_C_prime, count, newNode))


                idx = np.argwhere(I == _i_prime).squeeze()
                if idx >= len(I) - 1:  # we reached item with highest ordinal
                    return
                _next_i = I[idx + 1]  # _i_prime + 1  # leads to Index Out Of Range Error on I
                # next_gely_args = {"C":_C_prime, "N":N, "i":_next_i}
                # newNode.gely_state = next_gely_args
                list_closed_no_rules(_C_prime, N, _next_i,  # What if min(X) == max(I)?
                                     t, D, I, T, results,
                                     closure, parent=newNode, direction='left',
                                     subset_test=subset_test)

        idx = np.argwhere(I == _i_prime).squeeze() + 1  # {k in I\C: k > _i_prime}
        Y = I[idx:]  # not out of bounds, just empty
        Y = OrderedSet(Y) - C
        if len(Y) > 0:
            _i_prime_prime = min(Y)
            newNode_pp = ItemsetNode(parent=parent, itemset=None,
                                     rel_items_global=parent.rel_items_global, n_samples_global=parent.n_samples_global)
            parent.add_child(newNode_pp, 'right')

            list_closed_no_rules(C, N.union({_i_prime}), _i_prime_prime, t,
                                 D, I, T, results,
                                 closure, parent=newNode_pp, direction='right',
                                 subset_test=subset_test)

    return

def list_closed_dscriminatory(C, N, i, t, D, I, T, results,
                              data_target, data_other, closure=None, parent: ItemsetNode=None,
                              subset_test=False, direction=None, model_callable=None, dt_kwargs=None):
    # TODO
    #  subset_test ->
    #  Q: should we filter data according to support and even put data from other nodes into other?
    #  A: fitler data according to support yes to not deliberately fit to data that
    #     does not fit the itemset (explanations), but don't necessarily put in other because that will make
    #     the rules even more complicated. instead we should propose to compute a precision/recall
    #     to see how many rules apply to samples where "they shouldn't" (rules coming from a different itemset that
    #     does not include that sample in their support)
    #     then our system would require a local attribution method and we can
    #       1. filter our rules for rules that use the dimensions indicated
    #       2. check their applicability
    #       -> faithful rules.

    if parent is None:
        raise ValueError

    if closure is None:
        closure = lambda X: _ti(_it(X, T, D), I, D)

    idx = np.argwhere(I == i).squeeze()  # {k in I\C : k >= i}
    X = I[idx:]
    X = OrderedSet(X) - C

    if len(X) > 0:
        _i_prime = X[0] #min(X) # get nextitem
        _C_prime: OrderedSet = closure(C.union({_i_prime}))

        # we could probably optimize the size of D and _data_target because support of subsequent itemsets is
        # always a subset of the support of previous itemsets
        is_frequent, count, support_idxs = _is_frequent(_C_prime, D, t)

        # _D = D[support_idxs]
        if subset_test:
            if len(support_idxs) > 0:
                _D = [D[si] for si in support_idxs]
                _data_target = data_target[support_idxs]
            else:
                _D = []
                _data_target = np.empty(0)
        else:
            _D = D
            _data_target = data_target

        newNode = ItemsetNode(parent=parent, itemset=_C_prime,
                              rel_items_global=parent.rel_items_global, n_samples_global=parent.n_samples_global)
        newNode.is_frequent = is_frequent
        newNode.data_target = _data_target.copy()

        if is_frequent:
            # model callable takes data and returns True if sample is in target class, False otherwise
            if model_callable is not None:
                _synthetic_data_target, _synthetic_data_other = generate_synthetic_data(
                    _data_target, data_other, model_callable, list(_C_prime)
                )
                # print(f"generated {len(_synthetic_data_other)+len(_synthetic_data_target)} samples")
                if len(_synthetic_data_target) > 0:
                    _enriched_data_target = np.vstack([_data_target, _synthetic_data_target])
                    if dt_kwargs is None:
                        dt_kwargs = {}
                    dt_kwargs['class_weight'] = ['balanced']
                else:
                    _enriched_data_target = _data_target
                if len(_synthetic_data_other) > 0:
                    if dt_kwargs is None:
                        dt_kwargs = {}
                    dt_kwargs['class_weight'] = ['balanced']
                    _enriched_data_other = np.vstack([data_other, _synthetic_data_other])
                else:
                    _enriched_data_other = data_other
            else:
                _enriched_data_target = data_target
                _enriched_data_other = data_other

            newNode.compute_initial_rules(_enriched_data_target)
            newNode.compute_statistics(_data_target, data_other)
            # print(f"ini-> newNode acc: {newNode.accuracy}")
            # if newNode.accuracy == 1. and (newNode.parent is not None and newNode.parent.accuracy == 1.):
            #     if len(_C_prime.intersection(N)) == 0:
            #         parent.add_child(newNode, 'left')
            #         results.append((_C_prime, count, newNode))
            #     # by whatever miracle we found a pure node so we can terminate search here
            #     return
            # rule support is high and rule yields no conflict with other class/ is monochromatic
            # elif (is_frequent and
            if len(_C_prime.intersection(N)) == 0:  # and check_class_variable(_C_prime, class_item_ids):
                # print(sorted(_C_prime), count)
                parent.add_child(newNode, 'left')
                results.append((_C_prime, count, newNode))

                newNode.compute_rules_dt(dt_kwargs=dt_kwargs, data_target=_enriched_data_target, data_other=_enriched_data_other)
                newNode.compute_statistics(_data_target, data_other)
                    # print(f"dt -> newNode acc: {newNode.accuracy}")

                idx = np.argwhere(I == _i_prime).squeeze()
                if idx >= len(I)-1:  # we reached item with highest ordinal
                    return
                _next_i = I[idx + 1]  # _i_prime + 1  # leads to Index Out Of Range Error on I
                # next_gely_args = {"C":_C_prime, "N":N, "i":_next_i}
                # newNode.gely_state = next_gely_args
                list_closed_dscriminatory(_C_prime, N, _next_i,  # What if min(X) == max(I)?
                            t, _D, I, T, results, _data_target, data_other,
                                          closure, parent=newNode, direction='left',
                                          subset_test=subset_test)

        idx = np.argwhere(I == _i_prime).squeeze()+1  # {k in I\C: k > _i_prime}
        Y = I[idx:]  # not out of bounds, just empty
        Y = OrderedSet(Y) - C
        if len(Y) > 0:
            _i_prime_prime = min(Y)
            newNode_pp = ItemsetNode(parent=parent, itemset=None,
                                     rel_items_global=parent.rel_items_global, n_samples_global=parent.n_samples_global)
            parent.add_child(newNode_pp, 'right')
            # parent.add_child(newNode, 'right')
            list_closed_dscriminatory(C, N.union({_i_prime}), _i_prime_prime, t,
                        _D, I, T, results, _data_target, data_other,
                                      closure, parent=newNode_pp, direction='right',
                                      subset_test=subset_test)

    return


def gely_discriminatory(B, threshold, X_target, X_other, item_order, remove_copmlete_transactions=True,
                        subset_test=False, model_callable=None, compute_dt=True, compute_rules=True, dt_kwargs=None):
    '''
    :param D: Is a binary matrix;
              Columns = Items, index used as ordering
              Tows = Transactions, index used as tid
    :param threshold: frequency threshold used to determine if an itemset is considered frequent
    :return:
    '''

    if 0 < threshold < 1:
        threshold = int(B.shape[0] * threshold)
        # print(f'threshold set to {threshold}')

    n_full_transactions = None
    if remove_copmlete_transactions and not np.all(B==1):  # remove rows that are all 1s
        n_I = B.shape[1]
        _T_sizes = np.sum(B, 1)
        unfull_transactions = _T_sizes < n_I
        D = B[unfull_transactions]
        if np.sum(D) > 0:
            n_full_transactions = np.sum(-1*(unfull_transactions-1))
            # print(f"removed {n_full_transactions} transactions that contained all items")
            if threshold < 1:
                threshold = max(int((B.shape[1] - n_full_transactions) * threshold), 1)
            else:
                threshold = threshold - n_full_transactions
                if threshold <= 0:
                    threshold = 1
                # print(f"adapted threshold to {threshold+n_full_transactions} - {n_full_transactions} = {threshold}")
        else:
            D = B
    else:
        D = B

    if np.sum(D) == 0:
        return ([], [])

    _size_T, _size_I = D.shape
    # T, I = np.arange(_size_T), np.arange(_size_I)
    # I = item_order
    T = np.arange(_size_T)
    #  remove items that never occur in any transaction in D (ie, filter zero columns)
    _non_zero_cols = np.argwhere(np.sum(D, 0) > 0).squeeze()
    # print(f"nonzero cols: {_non_zero_cols}")

    #
    I = np.array([i for i in item_order if i in _non_zero_cols])
    _t = threshold


    #  transform binary rows into transactions represented by lists of items (as indices)
    _D = []
    for _i, t in enumerate(D):
        t_iids = np.argwhere(t).squeeze()
        if len(t_iids.shape) == 0:
            t_iids = np.expand_dims(t_iids, 0)
        _D.append(list(t_iids))
    _D = [set(d) for d in _D]

    closure = lambda X: _ti(_it(X, _D), _D)
    _fcis = []
    import numbers
    if isinstance(I, numbers.Number):
        I = [I]
    if len(I) == 0:
        # this happens if filtering for full tansactions leaves only transactions that contain no items
        print("I is empty")
        print(B)
        print(D)
        print("panic")
    try:
        if not compute_rules:
            root_node = ItemsetNode(None, I, rel_items_global=OrderedSet(I), n_samples_global=len(B))
            list_closed_no_rules(
            C=OrderedSet(), N=OrderedSet(), i=I[0], t=_t, D=_D, I=I, T=T,
            results=_fcis, parent=root_node,
            closure=closure, subset_test=subset_test, model_callable=model_callable)
        else: # default
            root_node = ItemsetNode(None, I, rel_items_global=OrderedSet(I), n_samples_global=len(X_target))
            list_closed_dscriminatory(
                C=OrderedSet(), N=OrderedSet(), i=I[0], t=_t, D=_D, I=I, T=T,
                results=_fcis, data_target=X_target.copy(), data_other=X_other, parent=root_node,
                closure=closure, subset_test=subset_test, model_callable=model_callable,
                dt_kwargs=dt_kwargs # passed the dt depth
            )
    except RecursionError as e:
        if hasattr(e, 'message'):
            if "recursion depth exceeded" in e.message:
                old_rlimit = sys.getrecursionlimit()
                new_rlimit = int(old_rlimit * 1.5)
                print(f"raising recursion limit from {old_rlimit} to {new_rlimit}")
                del _fcis
                _fcis = []
                with recursionlimit(new_rlimit):
                    if not compute_rules:
                        root_node = ItemsetNode(None, I,
                                                rel_items_global=OrderedSet(I), n_samples_global=len(B))
                        list_closed_no_rules(
                            C=OrderedSet(), N=OrderedSet(), i=I[0], t=_t, D=_D, I=I, T=T,
                            results=_fcis, parent=root_node,
                            closure=closure, subset_test=subset_test, model_callable=model_callable)
                    else:  # default
                        root_node = ItemsetNode(None, I,
                                                rel_items_global=OrderedSet(I), n_samples_global=len(X_target))
                        list_closed_dscriminatory(
                            C=OrderedSet(), N=OrderedSet(), i=I[0], t=_t, D=_D, I=I, T=T,
                            results=_fcis, data_target=X_target.copy(), data_other=X_other, parent=root_node,
                            closure=closure, subset_test=subset_test, model_callable=model_callable
                        )

    # remove data from nodes for saving; put data in root_node
    _children = root_node.get_children()
    _data_target_nodes = [n.data_target for n in _children if n.data_target is not None]
    # print()
    for node in _children:
        node.data_target = None
        node.data_other = None
    root_node.data_target = X_target
    root_node.data_other = X_other
    gely_nodes = [root_node] + _children
    # if len(_children) == 0:
    #     raise ValueError("NO CHILDREN")

    if not compute_rules:
        support_idxs_frequent_nodes = [deepcopy(r.idxs) for r in root_node.get_frequent_children()]
        for r in root_node.get_frequent_children():
            del r.idxs

    else:
        _rule_preds = [r.dnf(X_target) for r in root_node.get_frequent_children()]
        support_idxs_frequent_nodes = []
        for r in _rule_preds:
            _r = np.argwhere(r).squeeze()
            # if only a single indes is True then argwhere returns array without shape and set() raises error because
            # it expects an iterable and not an integer/ shapeless array
            if _r.shape == ():
                _r = np.expand_dims(np.array(_r), 0)
            support_idxs_frequent_nodes.append(set(_r))
        # _items_covered = set(chain.from_iterable(support_idxs_frequent_nodes))
    return (support_idxs_frequent_nodes, gely_nodes)


def list_closed(C, N, i, t,
                  D, I, T, results, closure=None, D_binary=None, class_item_ids=None):
    raise NotImplementedError
    if closure is None:
        closure = lambda X: _ti(_it(X, T, D), I, D)

    idx = np.argwhere(I == i).squeeze()  # {k in I\C : k >= i}
    X = I[idx:]
    X = set(X) - C

    if len(X) > 0:
        _i_prime = min(X)
        _C_prime: set = closure(C.union({_i_prime}))

        if D_binary is not None:
            is_frequent, count = _is_frequent_binary(_C_prime, D_binary, t)
        else:
            is_frequent, count, _ = _is_frequent(_C_prime, D, t)

        if is_frequent and len(_C_prime.intersection(N)) == 0:  # and check_class_variable(_C_prime, class_item_ids):
            # print(sorted(_C_prime), count)
            results.append((_C_prime, count))
            idx = np.argwhere(I == _i_prime).squeeze()
            if idx >= len(I)-1:  # we reached item with highest ordinal
                return
            _next_i = I[idx + 1]  # _i_prime + 1  # leads to Index Out Of Range Error on I
            list_closed(_C_prime, N, _next_i,  # What if min(X) == max(I)?
                        t, D, I, T, results, closure, D_binary, class_item_ids)

        idx = np.argwhere(I == _i_prime).squeeze()+1  # {k in I\C: k > _i_prime}
        Y = I[idx:]  # no out of bounds, just empty
        Y = set(Y) - C
        if len(Y) > 0:
            _i_prime_prime = min(Y)
            list_closed(C, N.union({_i_prime}), _i_prime_prime, t,
                        D, I, T, results, closure, D_binary, class_item_ids)

    return


def gely(B, threshold, use_binary=False, remove_copmlete_transactions=True, targets=None, verbose=True):
    '''

    :param D: Is a binary matrix;
              Columns = Items, index used as ordering
              Tows = Transactions, index used as tid
    :param threshold: frequency threshold used to determine if an itemset is considered frequent
    :return:
    '''

    if 0 < threshold < 1:
        threshold = int(B.shape[0] * threshold)
        # print(f'threshold set to {threshold}')

    n_full_transactions = None
    if remove_copmlete_transactions:
        n_I = B.shape[1]
        _T_sizes = np.sum(B, 1)
        unfull_transactions = _T_sizes < n_I
        D = B[unfull_transactions]
        n_full_transactions = np.sum(-1*(unfull_transactions-1))
        # print(f"removed {n_full_transactions} transactions that contained all items")
        if threshold < 1:
            threshold = max(int((B.shape[1] - n_full_transactions) * threshold), 1)
        else:
            threshold = threshold - n_full_transactions
            if threshold <= 0:
                threshold = 1
            # print(f"adapted threshold to {threshold+n_full_transactions} - {n_full_transactions} = {threshold}")
    else:
        D = B

    _size_T, _size_I = D.shape
    T, I = np.arange(_size_T), np.arange(_size_I)
    #  remove items that never occur in any transaction in D (ie, filter zero columns)
    _non_zero_cols = np.argwhere(np.sum(D, 0) > 0).squeeze()
    I = I[_non_zero_cols]
    _t = threshold

    class_item_ids = None
    if targets is not None:
        n_classes = len(np.unique(targets))
        _targets_binarized = np.zeros([B.shape[0], n_classes])
        _targets_binarized[:, targets] = 1
        D = np.hstack([D, _targets_binarized])
        class_item_ids = set(np.arange(D.shape[1] - n_classes, D.shape[1]))
    # transform binary rows into transactions represented by lists of items (as indices)
    _D = []
    for _i, t in enumerate(D):
        t_iids = np.argwhere(t).squeeze()
        if len(t_iids.shape) == 0:
            t_iids = np.expand_dims(t_iids, 0)
        _D.append(list(t_iids))
    _D = [set(d) for d in _D]

    if use_binary:
        D = D.astype(int)
        closure = lambda X: _ti_binary_matrix(_it_binary_matrix(X, D), D)  # D not _D !
    else:
        closure = lambda X: _ti(_it(X, _D), _D)

    _fcis = []
    import numbers
    if isinstance(I, numbers.Number):
        I = [I]
    try:
        list_closed(
            C=set(), N=set(), i=min(I), t=_t, D=_D, I=I, T=T,
            results=_fcis, closure=closure, D_binary=D if use_binary else None,
            class_item_ids=class_item_ids
        )
    except RecursionError as e:
        if hasattr(e, 'message'):
            if "recursion depth exceeded" in e.message:
                old_rlimit = sys.getrecursionlimit()
                new_rlimit = int(old_rlimit * 1.5)
                print(f"raising recursion limit from {old_rlimit} to {new_rlimit}")
                del _fcis
                _fcis = []
                with recursionlimit(new_rlimit):
                    list_closed(
                        C=set(), N=set(), i=min(I), t=_t, D=_D, I=I, T=T,
                        results=_fcis, closure=closure, D_binary=D if use_binary else None,
                        class_item_ids=class_item_ids
                    )


    # return list sorted by (largest itemsets, larger support)
    _fcis = sorted(_fcis, key=lambda x: (-x[1], len(x[0])), reverse=True)
    if n_full_transactions is not None:
        _fcis = [(f[0], f[1]+n_full_transactions) for f in _fcis]
    return _fcis

if __name__ == '__main__':
    D = ['abde', 'bce', 'abde', 'abce', 'abcde', 'bcd']
    support_thresh = 3
    I = ['a', 'b', 'c', 'd', 'e']
    _I = {k: v for v, k in enumerate(I)}

    T = np.arange(len(D))
    B = np.zeros((len(T), len(I)))
    for tid, t in zip(T, D):
        for _iid, i in enumerate(I):
            B[tid, _iid] = 1 if i in t else 0

    assert np.all(B ==
                  np.array([
                            [1, 1, 0, 1, 1],
                            [0, 1, 1, 0, 1],
                            [1, 1, 0, 1, 1],
                            [1, 1, 1, 0, 1],
                            [1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 0]
                        ])
                  )

    fcis = gely(B, support_thresh)
    for f, c in fcis:
        print(f"{[I[i] for i in f]} x {c}")