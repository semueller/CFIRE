from typing import Optional, Union

import sklearn.tree
import torch
from torch import nn, device

from itertools import chain, combinations

import numpy as np

from time import time

from sklearn.tree import DecisionTreeClassifier
from torch.nn.modules.module import T

import enum

from copy import deepcopy

class SimpleTorchWrapper():
    '''
    wrapper class that maps basic functions of an sklearn api based model to torch.

    '''

    def __init__(self, sklearn_model):
        self.model = sklearn_model
        self.inference_fn = self.model.predict_proba
        self.training = False
        self.device = 'cpu'


    # helper
    def __cast_input(self, X):
        try:
            X_np = X.detach().cpu().numpy()
        except AttributeError:
            X_np = np.array(X)
        return X_np

    # prediction wrapper
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        # take torch tensor/ numpy array and return
        X_np = self.__cast_input(X)
        y = self.inference_fn(X_np)
        y_tensor = torch.tensor(y)
        return y_tensor

    def predict_batch(self, x):
        _X = self.__cast_input(x)
        pred = self.model.predict(x)
        pred = torch.tensor(pred)
        return pred

    # dummy functions
    def eval(self):
        return

    def to(self, device):
        return self

    def parameters(self):
        return iter([self.model])

    def train(self, b):
        return None


class RuleClassifier:
    def __init__(self, rules: list[list[list[tuple]]]):
        # rule format: dimension, interval -> tuple(dimension, tuple(lower_limit, upper_limit))
        self.n_classes = len(rules)
        self.rules = rules

    def __call__(self, samples):
        return self.predict(samples)

    def __rule_match_sample(self, sample, rule):
        matches = True
        for clause in rule:
            dim, (start, end) = clause
            if not start < sample[dim] <= end:
                matches = False
        return matches

    def __predict_sample(self, sample):
        count_applicable = []
        for _class_rules in self.rules:
            count_applicable.append(0)
            for rule in _class_rules:
                if self.__rule_match_sample(sample, rule):
                    count_applicable[-1] += 1
            count_applicable[-1] /= len(_class_rules)
        return np.argmax(count_applicable)

    def predict(self, samples):
        predictions = []
        for sample in samples:
            predictions.append(self.__predict_sample(sample))
        return np.array(predictions)

    def get_num_rules(self):
        n_rules = []
        for c in range(len(self.rules)):
            n_rules.append(len(self.rules[c]))
        return n_rules

# class Literal():
#     def __init__(self, dim, left, right):
#         self.dim = dim
#         self.left = left
#         self.right = right
#
#     def __str__(self):
#         return str((self.dim, (self.left, self.right)))
#
# class Rule():
#     def __init__(self, terms: list):
#         self.terms = terms

def merge_rules(a, b):
    # if not mergeable return (a, b)
    # if mergeabl then return (merged_rule, None)

    def get_dims(r):
        return set([t[0] for t in r])

    def overlap(i1, i2):
        i1, i2 = sorted([i1, i2], key=lambda x: (x[0], x[1]))
        if i1[1] >= i2[0]:
            return True
        return False

    def overlap_not_contain(i1, i2):
        i1, i2 = sorted([i1, i2], key=lambda x: (x[0], x[1]))
        if i1[1] < i2[0]: return False
        if i1[0] <= i2[0] and i1[1] >= i2[0] and i1[1] < i2[1]: return True
        return False

    def contains(i1, i2):
        return i1[0] <= i2[0] and i2[1] <= i1[1]

    a = sorted(a, key=lambda t: t[0])
    b = sorted(b, key=lambda t: t[0])
    da = get_dims(a)
    db = get_dims(b)

    if not (da <= db or db <= da): # if rules use different dims, return
        return a, b

    if da == db:
        # if a, b look at the same dims: if all dims overlap, the rules can be merged
        new_rule = []
        _n_overlap = 0
        _containment = None

        if np.all([ta == tb for ta, tb in zip(a, b)]):  # rules are identical
            return a, None

        _a_contains_b = np.array([contains(ta[1], tb[1]) for ta, tb in zip(a, b)])
        _b_contains_a = np.array([contains(tb[1], ta[1]) for ta, tb in zip(a, b)])

        if np.all(_a_contains_b):
            return a, None
        if np.all(_b_contains_a):
            return b, None

        _eq = np.logical_and(_a_contains_b, _b_contains_a)

        if sum(_eq) == len(_eq) - 1:
            for i, (ta, tb) in enumerate(zip(a, b)):
                if _a_contains_b[i]:
                    new_rule.append(ta)
                elif _b_contains_a[i]:
                    new_rule.append(tb)
                elif overlap_not_contain(ta[1], tb[1]):
                    _start = min(ta[1][0], tb[1][0])
                    _end = max(ta[1][1], tb[1][1])
                    new_rule.append((ta[0], (_start, _end)))
                else:  # if they don't overlap, don't merge
                    return a, b
            # all contained or overlapped
            return new_rule, None

        #
        return a, b

    # else: da != db

    if len(db) < len(da):
        a, b = b, a
        da, db = db, da

    # a is the shorter, potentially more general, rule
    # check that all intervals of terms in a fully contain the terms in b

    def __get_term_for_dim(terms, dim):
        for t in terms:
            if t[0] == dim: return t
        return None
    # assert that all terms in shorter rule (->a) contain all respective terms in longer rule (->b)
    _a_contains_b = np.array([contains(ta[1], __get_term_for_dim(b, ta[0])[1]) for ta, tb in zip(a, b)])
    if not all(_a_contains_b):
        return a, b
    # b partially marks other intervals in shared dimensions, hence keep both rules
    return a, None



class DNFClassifier:
    def __init__(self, rules: list[list[list[tuple[int, tuple[float, float]]]]], tie_break="first"):
        # rule format: dimension, interval -> tuple(dimension, tuple(lower_limit, upper_limit))
        self.n_classes = len(rules)
        self.rules = rules
        #self.rules: List[List[List[Tuple[int, Tuple[float, float]]]]]
                      #^    ^    ^    Tupble : literals
                      #|    |    └─ AND for the literals
                      #|    └───── OR conjunctions
                      #└────────── Class DNF: list of clauses, one per class
        self.purge_dummy_rules()
        self.rule_performances = {c: {} for c in range(self.n_classes)}  # dict to collect performance statistics of rules
        self.tie_break = tie_break
        assert (self.tie_break in
                ["first", "shortest", "longest", "random", "accuracy", "f1"])
        self.n_literals = self.__comp_n_literals()
        self.n_rules = self.__comp_n_rules()
        self._meta_information = None
        self.score = None


    def __call__(self, samples, explain=False):
        return self.predict(samples, explain=explain)

    def __iter__(self):
        self.current = -1
        return self

    def __next__(self):
        self.current += 1
        if self.current < self.n_classes:
            return self.rules[self.current]
        raise StopIteration

    def __getitem__(self, idx):
        # make class subscriptable
        return self.rules[idx]

    def purge_dummy_rules(self):
        purged_rules = []
        for class_rules in self.rules:
            purged_class = []
            for term in class_rules:
                term = list(filter(lambda x: x != (-1, (np.nan, np.nan)), term))
                purged_class.append(term)
            purged_rules.append(purged_class)
        self.rules = purged_rules

    def assert_no_infty(self):
        for class_rules in self.rules:
            for terms in class_rules:
                for literal in terms:
                    assert np.abs(literal[1][0]) != np.inf and np.abs(literal[1][1]) != np.inf

    def __set_rules(self, new_rules):
        self.rules = deepcopy(new_rules)
        self.n_classes = len(self.rules)
        self.n_literals = self.__comp_n_literals()
        self.n_rules = self.__comp_n_rules()
        return

    def __comp_n_literals(self) -> int:
        n_literals = 0
        for c in range(self.n_classes):
            c_dnf = self.rules[c]
            n_literals += sum([len(term) for term in c_dnf])
        return n_literals

    def unique_literals(self) -> set[tuple[int, tuple[float, float]]]:
        """
        Return a *set* containing every literal that appears in *any* clause
        of *any* class.
        """
        uniq = set()
        for class_dnf in self.rules:          # ⋯ iterate classes
            for clause in class_dnf:          # ⋯ iterate OR-clauses
                for lit in clause:            # ⋯ iterate AND-literals
                    uniq.add(tuple(lit))      # lit is already a 2-tuple
        return uniq

    def unique_literals_per_class(self) -> list[set[tuple[int, tuple[float, float]]]]:
        """
        Return a *list* with length == ``self.n_classes``.
        Each element is the *set* of unique literals that appear in that
        particular class’ DNF.

        usage ex:
         dnf.unique_literals_per_class()[0]   # all literals used for class-0
        { (2, (-inf,  0.5)), (4, (1.1, inf)), ... }
        """
        per_class = []
        for class_dnf in self.rules:
            lit_set = set()
            for clause in class_dnf:          # OR-clauses
                lit_set.update(clause)        # add *all* literals in clause
            per_class.append(lit_set)
        return per_class

    def n_unique_literals45(self) -> int:
        """Shorthand for len(self.unique_literals())."""

        length = len(self.unique_literals())
        print(length)
        return length

    def n_unique_literals_per_class(self) -> list[int]:
        """
        Shorthand that just returns ``[len(s) for s in unique_literals_per_class()]``.
        """
        return [len(s) for s in self.unique_literals_per_class()]

    def __comp_n_unique_literals(self) -> int:
        """
        Return the number of unique literals that occur in the whole DNF.

        A literal is identified by its complete 2-tuple ``(dim, (lower, upper))``.
        If the exact same literal shows up multiple times (even across different
        clauses or classes) it is only counted once.
        """
        print("we are couting unique literals")
        unique_literals: set[tuple[int, tuple[float, float]]] = set()

        for class_rules in self.rules:
            for clause in class_rules:
                unique_literals.update(clause)

        return len(unique_literals)

    def get_n_terms(self):
        return self.n_literals

    def __comp_n_rules(self):
        n_rules = 0
        for class_rules in self.rules:
            n_rules += len(class_rules)
        return n_rules

    def get_n_rules(self):
        return self.n_rules

    def get_relevant_dims(self):
        d = []
        for c in range(self.n_classes):
            c_dnf = self.rules[c]
            for term in c_dnf:
                for literal in term:
                    if literal[0] not in d:
                        d.append(literal[0])
        d = sorted(d)
        return d

    def compute_rule_performance(self, X, Y):
        # compute performance statistics [accuracy, prec, recall, f1] for every single term of each class
        # if then our DNF predicts multiple classes, predict class of rules with "best" performance
        for c in range(self.n_classes):
            # Xc = X[Y == c]
            # Xother = X[Y != c]
            # Yother = Y != c
            _class_rules = self.rules[c]
            Yc = Y == c

            _c_stats = {}
            for clause in _class_rules:
                applicability = []
                for sample in X:
                    applicability.append(self.__clause_match_sample(sample=sample, clause=clause))
                applicability = np.array(applicability)
                precision = sklearn.metrics.precision_score(Yc, applicability, zero_division=0)
                recall = sklearn.metrics.recall_score(Yc, applicability, zero_division=0)
                f1 = sklearn.metrics.f1_score(Yc, applicability, zero_division=0)

                _c_stats[tuple(clause)] = {'accuracy':precision, 'f1':f1, 'recall': recall}
            self.rule_performances[c] = _c_stats
        return

    def simplify_merge_rules(self):
        new_rules = []
        for cdnf in self.rules:
            merged_rules = True
            new_cdnf = cdnf
            while merged_rules:
                merged_rules = False
                for i in range(len(new_cdnf)):
                    r1 = new_cdnf[i]
                    for j in range(i+1, len(new_cdnf)):
                        r2 = new_cdnf[j]
                        m1, m2 = merge_rules(a=r1, b=r2)
                        if m2 is None:
                            new_cdnf.remove(r1)
                            new_cdnf.remove(r2)
                            new_cdnf.append(m1)
                            merged_rules = True
                            break
                    if merged_rules:
                        break
            new_rules.append(new_cdnf)

        self.__set_rules(new_rules)

    ## todo, not sure why remove here
    def remove_empirically_redundant_rules(self, X, min_complexity=True):
        assert self.tie_break != "random"
        _tie_break = self.tie_break
        self.tie_break = "first"

        _pred_before = self(X)
        rules_before = deepcopy(self.rules)

        reduced_rules = []

        _list_removed_clauses = []
        for _, class_dnf in enumerate(self.rules):
            removed_supports = []
            supports = []
            for clause in class_dnf:
                # class dnf consists of multiple conjuctive clauses
                applicability = [self.__clause_match_sample(x, clause) for x in X]
                supports.append(set(np.argwhere(applicability).reshape(-1)))

            reduced_class_rules = deepcopy(class_dnf)
            # if support SA is subset of support SB, remove rule corresponding to SA
            # if support SA == SB, remove longer rule if min_complexity else remove less specific rule/ keep both
            removed_clause = True
            while removed_clause:
                removed_clause = False
                for i in range(len(reduced_class_rules)-1):
                    clause1, s1 = reduced_class_rules[i], supports[i]

                    if len(s1) == 0:  # remove rule that doesn't have support
                        removed_supports.append(deepcopy(s1))
                        _list_removed_clauses.append(deepcopy(clause1))
                        del reduced_class_rules[i]
                        del supports[i]
                        removed_clause = True; break

                    for j in range(i+1, len(reduced_class_rules)):
                        clause2, s2 = reduced_class_rules[j], supports[j]
                        if s1 == s2:  # if support sets are equal

                            if min_complexity:  # remove longer rule if we prioritize complexity > completeness
                                if len(clause2) < len(clause1):
                                    removed_supports.append(deepcopy(s1))
                                    _list_removed_clauses.append(deepcopy(clause1))
                                    del reduced_class_rules[i]
                                    del supports[i]
                                else:
                                    removed_supports.append(deepcopy(s2))
                                    _list_removed_clauses.append(deepcopy(clause2))
                                    applicable = []
                                    applicable_by_clause = None
                                    if not b_need_applicability_by_clause:
                                        # use faster version
                                        for _class_rules in self.rules:
                                            #
                                            applicable.append(
                                                self.__rule_match_sample(sample, _class_rules)
                                            )
                                    else:
                                        applicable_by_clause = []
                                        for _class_rules in self.rules:
                                            applicable_by_clause.append([])
                                            for clause in _class_rules:
                                                applicable_by_clause[-1].append(
                                                    self.__clause_match_sample(sample=sample, clause=clause))
                                            if np.any(applicable_by_clause[-1]):
                                                applicable.append(1)
                                            else:
                                                applicable.append(0)
                                    del supports[j]
                                    del reduced_class_rules[j]
                                removed_clause = True; break
                            else:  # keep more specific rule (copmleteness) or keep both if they use different dims
                                _clause_dims = lambda _clause: set(np.unique([t[0] for t in _clause]))
                                cd1, cd2 = _clause_dims(clause1), _clause_dims(clause2)
                                if cd1 <= cd2:  # clause2 more specific than clause 1
                                    removed_supports.append(deepcopy(s1))
                                    _list_removed_clauses.append(deepcopy(clause1))
                                    del reduced_class_rules[i]
                                    del supports[i]
                                    removed_clause = True; break
                                elif cd2 <= cd1:  # clause1 more specific than clause2
                                    removed_supports.append(deepcopy(s2))
                                    _list_removed_clauses.append(deepcopy(clause2))
                                    del supports[j]
                                    del reduced_class_rules[j]
                                    removed_clause = True; break
                                else:
                                    pass  # keep both because they map to different dims

                        elif s1 < s2:  # if s1 is contained in s2, remove clause1
                            removed_supports.append(deepcopy(s1))
                            _list_removed_clauses.append(deepcopy(clause1))
                            del reduced_class_rules[i]
                            del supports[i]
                            removed_clause = True; break
                        elif s2 < s1:  # if s2 is contained in s1, remove s2
                            removed_supports.append(deepcopy(s2))
                            _list_removed_clauses.append(deepcopy(clause2))
                            del supports[j]
                            del reduced_class_rules[j]
                            removed_clause = True; break

                        if removed_clause:
                            break
                    #
                    if removed_clause:
                        # start iteration over reduced_class_rules again after list changed
                        break

            _supports_left = set().union(*supports)
            _supports_removed = set().union(*removed_supports)
            assert len(_supports_removed - _supports_left) == 0  # assert we don't "lose" a sample
            reduced_rules.append(reduced_class_rules)

        n_literals_before = self.n_literals
        n_rules_before = self.n_rules
        self.__set_rules(deepcopy(reduced_rules))
        # print(f"removed redundant rules")
        # print(f"n_literals {n_literals_before} -> {self.n_literals}")
        # print(f"n_rules {n_rules_before} -> {self.n_rules}")
        _pred_after = self(X)
        _after_eq_before = np.all(_pred_before == _pred_after)
        # if not np.all(_pred_before == _pred_after):
        #     print(f"what happened")
        assert np.all(_pred_before == _pred_after)
        n_literals_before_simplification = self.n_literals
        n_rules_before_simplification = self.n_rules
        self.simplify_merge_rules()
        _pred_after_simplified = self(X)
        _simplified_eq_before = np.all(_pred_after_simplified == _pred_before)
        _simplified_eq_after = np.all(_pred_after_simplified == _pred_after)
        # if n_literals_before_simplification != self.n_literals:
        #     print(f"simplified rules")
        #     print(f"n_literals {n_literals_before_simplification} -> {self.n_literals}")
        #     print(f"n_rules {n_rules_before_simplification} -> {self.n_rules}")
        # print()
        # if not np.all(_pred_before == _pred_after_simplified):
        #     print(f"simplifying changed behavior")
        assert np.all(_pred_before == _pred_after_simplified)

        self.tie_break = _tie_break 
        return None

    def __rule_match_sample(self, sample, rule):
        # DNF
        for clause in rule:
            matches = True
            for literal in clause:
                dim, (start, end) = literal
                # TODO, checking <= on both sides may 'connect' intervals from different terms but hasn't happened so
                #  far. what _has_ happened is that the data is such that min == max ..
                if not start <= sample[dim] <= end:
                    # clause cannot be fulfilled anymore
                    matches = False
                    break
            # if all literals are fulfilled, return True
            if matches:
                return True
        # no clause returned True, hence return False
        return False

    def __clause_match_sample(self, sample, clause):
        matches = True
        for literal in clause:
            dim, (start, end) = literal
            # TODO, checking <= on both sides may 'connect' intervals from different terms but hasn't happened so
            #  far. what _has_ happened is that the data is such that min == max ..
            if not start <= sample[dim] <= end:
                # clause cannot be fulfilled anymore
                matches = False
                break
        # if all literals are fulfilled, return True
        return matches

    def __predict_sample(self, sample, explain=False):

        # if explain also return all applicable rules from predicted class

        if self.tie_break not in ["first", "random"] or explain:
            b_need_applicability_by_clause = True
        else:
            b_need_applicability_by_clause = False

        applicable = []
        applicable_by_clause = None
        if not b_need_applicability_by_clause:
            # use faster version
            for _class_rules in self.rules:
                #
                applicable.append(
                    self.__rule_match_sample(sample, _class_rules)
                )
        else:
            applicable_by_clause = []
            for _class_rules in self.rules:
                applicable_by_clause.append([])
                for clause in _class_rules:
                    applicable_by_clause[-1].append(self.__clause_match_sample(sample=sample, clause=clause))
                if np.any(applicable_by_clause[-1]):
                    applicable.append(1)
                else:
                    applicable.append(0)
            pass

        if b_need_applicability_by_clause:
            assert applicable_by_clause is not None

        if self.n_classes == 1:
            if not explain:
                return applicable[0]
            elif applicable[0]:
                c = self.rules[0]
                raise NotImplementedError("did not rework this yet for n_classes=1")
                # return (applicable[0], [c[i] for i, a in enumerate(applicable_by_clause[0]) if a])
            else:
                raise NotImplementedError("did not rework this yet for n_classes=1")
                # return (False, None)

        if not np.any(applicable):  # reject
            prediction = -1
        else:
            # if multiple, choose first class that matches
            prediction = np.argwhere(applicable).squeeze()
            if prediction.size == 1:  # only one class predicted
                prediction = prediction.item()

            elif self.tie_break == "first":
                prediction = prediction[0]

            elif self.tie_break == "accuracy":  # or f1
                _best_class, _best_acc = prediction[0], -1
                for p in prediction:
                    p_applicable = applicable_by_clause[p]
                    for i, a in enumerate(p_applicable):
                        if not a:
                            continue
                        r = self.rules[p][i]
                        acc = self.rule_performances[p][tuple(r)]['accuracy']
                        if acc > _best_acc:
                            _best_class, _best_acc = p, acc
                prediction = _best_class

            elif self.tie_break == "f1":  # or f1
                _best_class, _best_acc = prediction[0], -1
                for p in prediction:
                    p_applicable = applicable_by_clause[p]
                    for i, a in enumerate(p_applicable):
                        if not a:
                            continue
                        r = self.rules[p][i]
                        acc = self.rule_performances[p][set(r)]['f1']
                        if acc > _best_acc:
                            _best_class, _best_acc = p, acc
                prediction = _best_class

            elif self.tie_break == "shortest":
                # TODO: write __rule_match_sample function that returns
                #  indicating for each term in class DNF if it matches
                # shortest: choose class with fewest rules matching "shortest explanation"
                raise NotImplementedError
            elif self.tie_break == "longest":
                # longest: choose class with the most number of rules matching "most specific/ certain(?) expl"
                raise NotImplementedError
            elif self.tie_break == "random": # self.tie_break = "random":
                prediction = np.random.choice(prediction, 1).item()#prediction[0]
            else:
                raise ValueError

        if explain:
            if isinstance(prediction, int) and prediction == -1:   # reject
                return (-1, None)

            # collect ALL matching clauses, not just from the predicted class
            matching_clauses = [((class_idx, clause_idx), self.rules[class_idx][clause_idx])
                                for class_idx, clauses_active in enumerate(applicable_by_clause)
                                for clause_idx, active in enumerate(clauses_active) if active]

            return (prediction, matching_clauses)

        return prediction

    def predict(self, samples, explain=False):
        predictions = []
        for sample in samples:
            predictions.append(self.__predict_sample(sample, explain))
        if not explain:
            return np.array(predictions)
        else:
            return predictions

    def get_num_rules(self):
        n_rules = []
        for c in range(len(self.rules)):
            n_rules.append(len(self.rules[c]))
        return n_rules

    def score_recall(self, X, Y):
        raise NotImplementedError

    def __describe_sample(self, sample):
        # returns a list of lists of rules
        # index corresponds to class
        # each list holds all rules that were applicable from that class
        applicable_by_clause = []
        for c, _class_rules in enumerate(self.rules):
            applicable_by_clause.append([])
            for clause in _class_rules:
                if self.__clause_match_sample(sample=sample, clause=clause):
                    applicable_by_clause[-1].append(clause)
        return applicable_by_clause

    def describe(self, samples):
        dscriptions = []
        for sample in samples:
            dscriptions.append(self.__describe_sample(sample))
        return dscriptions


    def __literal_in_clause(self, literal, clause):
        for lc in clause:
            if literal[0] == lc[0]:
                if literal[1][0] >= lc[1][0] and literal[1][1] <= lc[1][1]:
                    return True
        return False
    def __clause_contained(self, clause1, clause2):
        for literal in clause1:
            if not self.__literal_in_clause(literal, clause2):
                return False
        return True

    def __dnf_contained(self, dnf1, dnf2):
        contained, diff = [], []
        for clause in dnf1:
            if np.any([self.__clause_contained(clause, clause2) for clause2 in dnf2]):
                contained.append(clause)
            else:
                diff.append(clause)
        return (contained, diff)

    def __contrast_classes(self, class1, class2):
        dnf1 = self.rules[class1]
        dnf2 = self.rules[class2]
        raise NotImplementedError


    @staticmethod
    def from_DT(dt: sklearn.tree.DecisionTreeClassifier, verbose=False):

        def __intervals_plausible(i1: tuple[int, tuple[float, float]],
                                  i2: tuple[int, [float, float]]) -> bool:
            if i1[0] != i2[0]: return True # if they apply to different dimensions
            interval1, interval2 = sorted([i1[1], i2[1]], key=lambda i: i[0])
            return interval1[1] > interval2[0]


        def __term_is_plausible(term: list[tuple[int, tuple[float, float]]]):
            '''
            checks if dnf (for one class) is plausible,
            ie that every term comprises of non-contradictory literals
            '''
            for literal in term:  # assert all literals have valid intervals where min < max
                if not literal[1][0] <= literal[1][1]: return False
            if len(term) == 1: return True
            # # all terms apply to the same dimension
            # assert len(np.unique([term[0] for term in terms])) == 1
            # assert that all terms intersect/ have partial overlap
            for ta, tb in combinations(term, 2):
                if not __intervals_plausible(ta, tb):
                    return False
            return True

        def get_path(leaf, l, r, remove_leaf=False):
            # given leaf node, walk up the binary tree until root, return path (optional: remove the leaf)
            def get_parent(child, _l, _r):
                if child in _l:
                    return np.argwhere(_l == child).squeeze().item()
                else:
                    return np.argwhere(_r == child).squeeze().item()

            current_node = leaf
            pth = [current_node]
            while current_node != 0:  # 0 is root_id
                current_node = get_parent(current_node, l, r)
                pth.append(current_node)
            pth = pth[::-1]
            if remove_leaf:
                return pth[:-1]
            else:
                return pth

        def path_to_dnf(path, dims, thresh, _lr):
            # given nodes on a path, their thresholds and information about direction left (lt) or right (gr)
            # convert path to dnf with (dim, (interval start, interval end))
            # where start==-np.inf if node is left from parent and end==np.inf if node is right of parent
            _dnf = []
            for idx, node in enumerate(path[:-1]):
                lr = _lr[path[idx + 1]]  # look up if child is left or right -> leq or gr
                term = None
                if lr == 'l':
                    term = (dims[node], (-np.inf, thresh[node]))
                else:
                    term = (dims[node], (thresh[node], np.inf))
                _dnf.append(term)
            _dnf = sorted(_dnf, key=lambda x: x[0])
            assert __term_is_plausible(_dnf)
            return _dnf

        def prune_dnf(term, _lr):
            # given a single dnf as [rule_1, rule_2, rule_3] w/ rule = (dim, (inverval start, interval end))
            # merge two rules if they operate on the same dim.
            # take the larger number if interval is open to the left (-np.inf),
            # take smaller number if interval is open to the right (np.inf)
            # merge rules [(d, (-np.inf, a)), (d, (b, np.inf))] -> (d (b, a))
            # because dnf stems from one path, the case where b > a should not occur, we assert nonetheless
            pruned_dnf = []
            dims = np.unique([t[0] for t in term])
            for dim in dims:
                literals = [t for t in term if t[0] == dim]
                if len(literals) <= 1:
                    pruned_dnf.extend(literals)
                    continue
                else:
                    lmax = max([l[1][0] for l in literals])
                    rmin = min([l[1][1] for l in literals])
                    # lmin = min([t[1][0] if t[1][0] != -np.inf else np.inf for t in literals])  # gr
                    # if lmin == np.inf:  # np.isnan(lmin): # lmin == np.nan is always False
                    #     lmin = -np.inf
                    # # rmax = max([t[1][1] if t[1][1] != np.inf else np.nan for t in _terms])
                    # rmax = max([t[1][1] if t[1][1] != np.inf else -np.inf for t in literals])
                    # if rmax == -np.inf:
                    #     rmax = np.inf
                    assert rmin >= lmax
                    assert rmin != -np.inf or lmax != np.inf
                    lit = (dim, (lmax, rmin))
                    pruned_dnf.append(lit)

            return pruned_dnf


        t = dt.tree_
        left = t.children_left
        right = t.children_right
        n_nodes = t.capacity
        _lr = ['l' if i in left else 'r' for i in range(n_nodes)]
        assert n_nodes == len(left)
        dims = [np.nan if left[i] == -1 or right[i] == -1 else t.feature[i] for i in range(n_nodes)]
        thresh = [np.nan if left[i] == -1 or right[i] == -1 else t.threshold[i] for i in range(n_nodes)]
        is_leaf = [True if left[i] == right[i] == -1 else False for i in range(n_nodes)]

        # https://github.com/scikit-learn/scikit-learn/blob/3f89022fa/sklearn/tree/_classes.py#L507
        _leaf_class = np.array([np.argmax(t.value[i]) if is_leaf[i] else -1 for i in range(n_nodes)])

        # assert len(np.unique(_leaf_class)) == len(dt.classes_) + 1  # at least one leaf per class + 1? make no sense

        paths_by_class = [[get_path(leaf, left, right) for leaf in np.argwhere(_leaf_class == j).reshape(-1)]
                          for j in dt.classes_]
        if verbose:
            print("TREE INFO")
            print(f"leafs:{[i for i, j in enumerate(is_leaf) if j]}")
            print("node, dim, thesh")
            for n, th, d, il in zip(range(n_nodes), thresh, dims, is_leaf):
                if not il:
                    print(f"{n}, {d}, {th:.4f}")

            print("\n\nPATHS")
            for i, p in enumerate(paths_by_class):
                print(f"class {i}")
                for pp in p:
                    print(pp)
                print()
        paths_by_class = [[p for p in pc] for pc in paths_by_class] # [:-1] to remove leaf node because we know the class already
        dnf = [[path_to_dnf(p, dims, thresh, _lr) for p in path_class] for path_class in paths_by_class]

        # pruned_dnf = [[prune_dnf(__dnf, _lr) for __dnf in cdnf if __dnf != []] for cdnf in dnf]
        # if verbose:
        #     print("DNFs")
        #     print(f"before \n ->  pruned")
        #     for i in range(len(dnf)):
        #         d = dnf[i]
        #         pd = pruned_dnf[i]
        #         print(f"class {i}")
        #         for dd, pd in zip(d, pd):
        #             print(f"{dd} \n -> {pd}")
        #         print()

        # _DNF = DNFClassifier(pruned_dnf)
        _DNF = DNFClassifier(dnf)
        # _DNF.simplify()
        return _DNF

class BvFExitCode(enum.Enum):
    SUCCESS = 0
    ATTRIBUTION_ERROR = 1
    NO_BOXES = 2
    AMBIGUOUS = 3
    WRONG_LABEL = 4
    CLOSURE_MISSING = 5


class BvFExplainer:

    def __init__(self,
                 model_fn,
                 local_explainer_fn,
                 boxsystems: tuple[set, list[list[list[tuple]]]],
                 closure_fn, binarization_fn=None,
                 relevance_threshold = 0.01):
        # rule format: dimension, interval -> tuple(dimension, tuple(lower_limit, upper_limit))

        self.model_fn = model_fn
        self.local_explainer = local_explainer_fn
        self.boxsystems = boxsystems
        self.closure_fn = closure_fn
        self.binarization_fn = self.__default_binarization_fn if binarization_fn is None else binarization_fn
        self.relevance_threshold = relevance_threshold


    def __call__(self, samples):
        self.explain_samples(samples)

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def __default_binarization_fn(self, attribution):
        attribution = attribution/max(abs(attribution))
        attribution = np.array(attribution > 0.01, dtype=int)
        return attribution


    def explain_samples(self, samples, targets=None, attributions=None, output_logic_fn=None):
        e = []
        if targets is None or attributions is None:
            for x in samples:
                e.append(self.explain_sample(x, output_logic_fn=output_logic_fn))
        else:
            assert len(samples) == len(targets) == len(attributions)
            for x, t, a in zip(samples, targets, attributions):
                e.append(
                    self.explain_sample(x, output_logic_fn, t, a)
                )
        return e


    def explain_sample(self, sample, output_logic_fn, target=None, attribution=None):


        # compute target class
        if target is None:
            target = self.model_fn(sample)
            target = int(target)

        # compute local explanation
        if attribution is None:
            attribution = self.local_explainer(sample)

        # compute closure
        binarized = self.binarization_fn(attribution).reshape(-1)
        items = set(list(np.argwhere(binarized).reshape(-1)))
        closed_itemset = self.closure_fn(items)

        return output_logic_fn(sample, target, attribution, items, closed_itemset)

        pass  # END OF METHOD

    def output_antichain_covering_predecessor(self, sample, target, attribution, items, closed_itemset):

        # instead of going "up" with the closure, we go "down"
        # if we do not find a closed itemset,
        # 1) consider all parents (existing, closed, frequent subsets)
        # 2) select those that contain the sample
        # 3) select the one where the dimensions have the highest attributions

        # how do we obtain all "parents"?
        # 1) get all subsets
        # 2)
        selected_box_system = None
        for boxes in self.boxsystems:
            if boxes[0] == items: # TODO CHECK closed_itemset?
                selected_box_system = boxes
                break

        if selected_box_system is None:
            # anti-chain logic goes here

            # consider all parents
            # select only parents that contain sample
            # select parent best matching attributions
            # hope.

            candidates = []
            # pre-select as candidates only itemsets that use some subset of (closed_)items
            for b in self.boxsystems:
                if set(b[0]).issubset(items): # TODO CHECK closed_itemset?
                    candidates.append(b)

            # check which ones cover our sample
            covering_candidates = []
            for cand in candidates:
                if np.any([len(d) for d in [cand[1].describe([sample])]]):
                    covering_candidates.append(cand)

            # if none of the systems contains the sample we return
            if len(covering_candidates) == 0:
                return (BvFExitCode.NO_BOXES,
                        f"No boxes that contained sample in boxsystem corresponding to itemset = {items}", [])

            # remove all itemsets that have a superset within the list of candidates
            antichain = []
            for c1 in covering_candidates:
                has_superset = False
                for c2 in covering_candidates:
                    if c1 != c2 and c1[0].issubset(c2[0]):
                        has_superset = True
                        break
                if not has_superset:
                    antichain.append(c1)


            # score elements of antichain
            if len(antichain) > 1:
                _antichain_itemsets = [a[0] for a in antichain]
                _antichain_itemsets_sums = [sum(attribution[list(a)]) for a in _antichain_itemsets]
                selected_box_system = antichain[np.argmax(_antichain_itemsets_sums)]
            else:
                selected_box_system = antichain[0]

        expl = selected_box_system[1].describe([sample])
        # check if label correct
        labels = [c for c in range(len(expl))  if len(expl[c]) > 0 ]
        if len(labels) == 0:
            return (BvFExitCode.NO_BOXES,
                    f"No boxes that contained sample in boxsystem corresponding to closed itemset = {closed_itemset}", [])

        if len(labels) > 1:
            return (BvFExitCode.AMBIGUOUS,
                    f"ambiguous prediction, target {target} but got predicted classes {labels}", expl)

        # labels != 0 and labels <= 2 -> len(labels) == 1
        label = labels[0]

        if label == target:
            return (BvFExitCode.SUCCESS,
                    f"rules match predicted class {target}", expl[label])

        if label != target:
            return (BvFExitCode.WRONG_LABEL,
                    f"rules match wrong class. predited {label} != target {target}", expl[label])

        pass

    def output_logic_v1(self, sample, target, attribution, items, closed_itemset):

        only_non_negative_features = True
        if set(closed_itemset) > set(items):
            diff = set(closed_itemset) - set(items)
            for d in diff:
                if attribution[d] < 0 and abs(attribution[d]) > self.relevance_threshold:
                    only_non_negative_features = False

        # select itemset, apply sample and return (cases i-iv)
        expl = None
        ## case (i)
        if not only_non_negative_features:
            return (BvFExitCode.ATTRIBUTION_ERROR,
                    f"closure added features with negative attribution scores, items={items}, closure={closed_itemset}", [])

        bs = None
        for boxes in self.boxsystems:
            if boxes[0] == closed_itemset:
                bs = boxes
                break

        if bs is None:
            return (BvFExitCode.CLOSURE_MISSING,
                    f"No boxsystem based on {closed_itemset}", [])

        # returns all applicable rules and labels
        # expl contains one list per class showing all applicable rules
        # if no rule of a class is applicable, the list is empty.
        # the index of a list indicates class label
        expl = bs[1].describe([sample])

        # check if label correct
        labels = [c for c in range(len(expl))  if len(expl[c]) > 0 ]
        if len(labels) == 0:
            return (BvFExitCode.NO_BOXES,
                    f"No boxes that contained sample in boxsystem corresponding to closed itemset = {closed_itemset}", [])

        if len(labels) > 1:
            return (BvFExitCode.AMBIGUOUS,
                    f"ambiguous prediction, target {target} but got predicted classes {labels}", expl)

        # labels != 0 and labels <= 2 -> len(labels) == 1
        label = labels[0]

        if label == target:
            return (BvFExitCode.SUCCESS,
                    f"rules match predicted class {target}", expl[label])

        if label != target:
            return (BvFExitCode.WRONG_LABEL,
                    f"rules match wrong class. predited {label} != target {target}", expl[label])

        # END OF METHOD; should never be reached
        pass

def make_fmnist_large() -> nn.Module:
    sequential = nn.Sequential(
        nn.Conv2d(1, 6, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(6, 12, 3),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(300, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    return SimpleNet(sequential=sequential)


def make_fmnist_small(n_classes=10):
    sequential = nn.Sequential(
        nn.Conv2d(1, 12, 5),
        nn.ReLU(),
        nn.MaxPool2d(4, 4),
        nn.Flatten(),
        nn.Linear(432, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, n_classes)
    )
    return SimpleNet(sequential=sequential)

class FCNN(nn.Module):
    def __init__(self, in_features, feature_multiplier=1, classes=2):
        super(FCNN, self).__init__()
        torch.manual_seed(1234)
        self.name = "fcnn"
        self.fc1 = nn.Linear(in_features=in_features, out_features=int(in_features * feature_multiplier))
        self.fc2 = nn.Linear(in_features=int(in_features * feature_multiplier),
                             out_features=int(in_features * feature_multiplier))
        self.fc3 = nn.Linear(in_features=int(in_features * feature_multiplier), out_features=classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

    def predict_batch_softmax(self, x):
        x = self(x)
        return self.softmax(x)

def make_ff(shapes_layers, actfun_out=None):
    # shapes layers: output size layer is input size of next ..
    layers = [nn.Linear(s_in, s_out) for (s_in, s_out) in zip(shapes_layers[:-1], shapes_layers[1:])]
    actfun = nn.ReLU
    architecture = []
    for layer in layers:
        architecture.append(layer)
        architecture.append(actfun())
    architecture = architecture[:-1] # delete last actfun
    if actfun_out is not None:
        architecture.append(actfun())
    # architecture2 = [
    #     nn.Linear(16,16),
    #     nn.ReLU(),
    #     nn.Linear(16,16),
    #     nn.ReLU(),
    #     nn.Linear(16,2),
    # ]
    sequential = nn.Sequential(*architecture)
    return SimpleNet(sequential=sequential)
    # return FCNN(16)

class SimpleNet(nn.Module):
    def __init__(self, sequential):
        super(SimpleNet, self).__init__()
        self.net = sequential
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.net(x)
        return x

    def predict_batch(self, x):
        pred = self.forward(x)
        return torch.argmax(pred, dim=1)

    def predict_batch_softmax(self, x):
        pred = self.forward(x)
        sm_pred = self.softmax(pred)
        return sm_pred

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters())
    #     return optimizer
    #
    # def training_step(self, train_batch, batch_idx):
    #     x, y = train_batch
    #     y_pred = self.net(x)
    #     # y_pred = torch.argmax(y_pred, dim=1)
    #     loss = self.LossFun(y_pred, y)
    #     self.log('train_loss', loss)
    #     return loss
    #
    # def validation_step(self, val_batch, batch_idx):
    #     x, y = val_batch
    #     y_pred = self.net(x)
    #     # y_pred = torch.argmax(y_pred, dim=1)
    #     loss = self.LossFun(y_pred, y)
    #     self.log('val_loss', loss)


class TorchEnsembleWrapper(nn.Module):

    def __init__(self, models, out_dim, forwards=None):
        super(TorchEnsembleWrapper, self).__init__()
        self.models = models
        self.out_dim = out_dim
        self.forwards = forwards if forwards is not None else models

        self.eval()

    def forward(self, x):
        outputs = torch.zeros((len(self.forwards), x.shape[0],  self.out_dim))
        # forward full batch through each model
        for i, f in enumerate(self.forwards):
            outputs[i] = f(x)
        # re-order, s.t. predictions.shape = (len(x), len(forwards), out_dim)
        predictions = torch.swapaxes(outputs, 0, 1)
        return predictions

    def predict(self, x):
        '''Possible ties resolved by whatever torch.argmax chooses first'''
        outputs = self.forward(x)  # shape: batch, n_models, n_classes
        votes = torch.argmax(outputs, dim=-1)  # shape: batch, n_models
        counts = [torch.bincount(p) for p in votes]  # bincount inserts 0 for missing values
        predictions = [torch.argmax(count) for count in counts]  # get class label with highest count
        predictions = torch.tensor(predictions)
        return predictions

class TorchAdditiveEnsemble(nn.Module):

    def __init__(self, models):
        super(TorchAdditiveEnsemble, self).__init__()
        self.models = models
        self.eval()

    def forward(self, x):
        output = self.models[0](x)

        for model in self.models[1:]:
            output += model(x)

        # TODO: Softmax?
        return output

    def predict(self, x):
        output = self.forward(x)
        return torch.argmax(output, axis=-1)



class BiLSTMClassif(nn.Module):

    def __init__(self, nr_classes, embed_dim, hid_size, vocab_size):
        super(BiLSTMClassif, self).__init__()
        # sparse limits available optimizers to SGD, SparseAdam and Adagrad
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=False)
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hid_size,
            # dropout=False,
            bidirectional=True,
            batch_first=True,
        )
        self.fc_out = nn.Linear(2*hid_size, nr_classes)
        # self.return_type = -1

    def get_embeddings_variance(self):
        return torch.var(self.embedding.weight).item()

    def embed_sequences(self, seqs):
        with torch.no_grad():
            embedded = self.embedding(seqs)
        return embedded

    def forward(self, seqs): #, offsets):
        embedded = self.embedding(seqs) #, offsets)
        return self._forward_embedded(embedded)

    def _forward_embedded(self, embedded_seq):
        lstm_out, (_, _) = self.bilstm(embedded_seq)
        h_T = lstm_out[:, -1]
        y = self.fc_out(h_T)
        return y

    def _forward_embedded_softmax(self, embedded_seq):
        x = self._forward_embedded(embedded_seq)
        y = torch.softmax(x, dim=1)
        return y

    def forward_softmax(self, seqs):
        embedded = self.embedding(seqs) #, offsets)
        return self._forward_embedded_softmax(embedded)

class Encoder(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Encoder, self).__init__()
        self.fc_out = nn.Linear(in_dim, out_dim)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.fc_out(x))

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Decoder, self).__init__()
        self.fc_out = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc_out(x)

class Autoencoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=None):
        super(Autoencoder, self).__init__()
        if hidden_dim is None:
            hidden_dim = int(np.ceil(in_dim*0.2))
        self.encoder = Encoder(in_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, in_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))

class InverseLoss(nn.Module):
    def __init__(self, loss_fn):
        super(InverseLoss, self).__init__()
        self.loss_fn = loss_fn

    def forward(self, pred, target):
        return torch.pow(self.loss_fn(pred, target), -1)

class AutoencoderLossWrapper(nn.Module):
    def __init__(self, autoencoder, loss_fn=None):
        super(AutoencoderLossWrapper, self).__init__()
        self.ae = autoencoder
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()

    def forward(self, x):
        _ae_out = self.ae(x)
        return self.loss_fn(_ae_out, x)


def validate(model, data, verbose=False):  # on single batch
    start = time()
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        x, y = data
        x = x.to(device)
        _y = model(x)
        _y = _y.argmax(-1).to('cpu')
        acc = (_y==y.cpu()).to(torch.float32).mean().item()

    model.train()
    end = time()
    if verbose:
        print(f"took {end-start:.3f}s")
    return acc
def simple_training(model, optim, loss_fn, tr_data, te_data, n_batches=10_000, device='cuda', pth='./', modelname='model',
               verbose=False, threshold=np.inf, return_best_model=True):
    if verbose:
        print(device)
    model.to(device)
    losses = [0.]
    batchnum = 0
    accs = []
    if verbose:
        print("process testset")
    accs.append(validate(model, te_data, verbose=verbose))
    best_acc = accs[-1]
    best_model = None
    if verbose:
        print(accs[-1])
    while batchnum < n_batches:
        for i, (data, labels) in enumerate(tr_data, 1):
            acc = validate(model, te_data); accs.append(acc)
            if return_best_model:
                if acc > best_acc:
                    best_model = deepcopy(model)
            if accs[-1] > threshold:
                break
            # models.append(copy.deepcopy(model).to('cpu'))
            data = data.to(device)
            labels = labels.to(device)
            out = model(data)
            loss = loss_fn(out, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
            if verbose:
                print(f"test acc @ {batchnum}: {acc}")
                print(f"mean loss since last check {sum(losses)/len(losses)}")
            batchnum += 1
            if n_batches < batchnum:
                break
        if accs[-1] > threshold:
            break
        # print(f'keeping {len(models)} models in memory')
    # print("accuracies over test set")
    # print(accs)

    acc = validate(model, te_data)
    accs.append(acc)
    # print(f'test accuracies')
    # print(accs[:10])
    # print('.\n.\n.')
    # print(f'best test acc:  {max(accs)} @ batch# {np.argmax(accs)}')
    # print(f"test accs; {accs[:5]} ... {accs[-5:]}")
    # print(f'train acc: {validate(model, tr_data)}')
    if return_best_model:
        return accs, best_model
    return accs


def extend_Bootstrap(BSEs, in_dim, out_dim, layers=None):
    if layers is None:
        layers = [8]*4

    # if this is not a base model that has not ensemble
    in_dim = in_dim + len(BSEs) * 1 #out_dim
    model = make_ff([in_dim] + layers + [out_dim])
    for b in BSEs:
        b.train(False)
    return BootstrapEnsembleNested(model, ensemble=BSEs)

class BootstrapEnsembleStacked(torch.nn.Module):
    pass

class BootstrapEnsembleNested(torch.nn.Module):
    def __init__(self, model, ensemble=None):
        super(BootstrapEnsembleNested, self).__init__()
        self.model = model
        self.ensemble = ensemble  #
        self.ensemble_is_Module = type(ensemble[0]) == torch.nn.Module
        if ensemble is not None:
            for m in ensemble:
                m.train(False)
        self.softmax = nn.Softmax(dim=1)

    def _preprocess_inputs(self, x):
        '''obtained extended representation of samples using the ensemble'''
        if self.ensemble is None:  # if this is the base model
            return x
        _x = [m.predict_batch(x).unsqueeze(1) for m in self.ensemble]
        _x = torch.hstack(_x)
        _x = torch.hstack((x, _x))
        return _x

    def forward(self, x):
        if self.ensemble is None:
            return self.model(x)
        x = self._preprocess_inputs(x)
        out = self.model(x)
        return out

    def predict_batch(self, x):
        pred = self.forward(x)
        return torch.argmax(pred, dim=1)

    def predict_batch_softmax(self, x):
        pred = self.forward(x)
        sm_pred = self.softmax(pred)
        return sm_pred


if __name__ == '__main__':
    from time import time

    def train_loop(model, optim, loss_fn, tr_data, te_data, n_batches=1000, device='cuda', pth='./', modelname='model',
                   verbose=False):
        print(device)
        model.to(device)
        models = []
        losses = [0.]
        batchnum = 0
        accs = []
        if verbose:
            print("process testset")
        accs.append(validate(model, te_data, verbose=verbose))
        if verbose:
            print(accs[-1])
        while batchnum < n_batches:
            for i, (text, labels) in enumerate(tr_data, 1):
                # if i % 10 == 0:
                    # create_checkpoint(f"{pth}{modelname}_{epoch}-{i}.torch", model, optim, epoch, None)
                acc = validate(model, te_data); accs.append(acc)
                if verbose:
                    print(f"test acc @ {batchnum}: {acc}")
                    print(f"mean loss since last check {sum(losses)/len(losses)}")
                # models.append(copy.deepcopy(model).to('cpu'))
                text = text.to(device)
                labels = labels.to(device)
                out = model(text)
                loss = loss_fn(out, labels)
                optim.zero_grad()
                loss.backward()
                optim.step()
                losses.append(loss.item())
                batchnum += 1
                if n_batches < batchnum:
                    break
            # print(f'keeping {len(models)} models in memory')
        # print("accuracies over test set")
        # print(accs)
        acc = validate(model, te_data)
        accs.append(acc)
        print(accs[:10])
        print('.\n.\n.')
        print(accs[-10:])
        print(f'train acc: {validate(model, tr_data)}')
        return accs


    print("TEST MODELS")
    from torch.optim import Adam, AdamW

    import sys

    from OpenXAI.openxai.dgp_synthetic import generate_gaussians
    from torch.utils.data import TensorDataset, DataLoader, RandomSampler


    import matplotlib.pyplot as plt

    nc = 70
    ns = 50*nc
    cov_m = np.array([[-1., 1.2],
                      [0.2, -12.]
                      ])
    cov_m = cov_m.T @ cov_m
    print(cov_m)

    # _all, train, test = generate_gaussians().dgp_vars()
    _all, train, test = generate_gaussians(
        dimensions=2,
        n_clusters=nc,
        n_samples=ns,
        correlation_matrix=cov_m
    ).dgp_vars()

    colors = np.array(['r', 'b'])
    x, y = _all['data'], _all['target']
    plt.scatter(x[:, 0], x[:, 1], c=colors[y], s=0.2)
    plt.show()

    from sklearn.tree import DecisionTreeClassifier as DT
    # dt5 = DT(max_depth=5)
    # dt5.fit(train['data'], train['target'])
    # print('depth 5', dt5.score(test['data'], test['target']))
    # dt10 = DT(max_depth=10)
    # dt10.fit(train['data'], train['target'])
    # print('depth 10', dt10.score(test['data'], test['target']))

    dt20 = DT(max_depth=20)
    dt20.fit(train['data'], train['target'])
    print('depth 20', dt20.score(test['data'], test['target']))
    dnf20 = DNFClassifier.from_DT(dt20)
    for r in dnf20.rules:
        print(len(r), r)
    exit()


    # dtX = DT(min_samples_split=100)
    # dtX.fit(train['data'], train['target'])
    # print('min_samples_split 100', dtX.score(test['data'], test['target']))

    train_loader = DataLoader(TensorDataset(torch.from_numpy(train['data']).float(),
                                            torch.from_numpy(train['target']).long()),
                                            shuffle=True, batch_size=64)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(test['data']).float(),
                                            torch.from_numpy(test['target']).long()),
                                            shuffle=True, batch_size=64)
    cnn_emnist = make_ff([20, 64, 64, 2])
    path = './data/'
    modelname= 'cnn_emnist'
    optim = AdamW(cnn_emnist.parameters())
    lfun = torch.nn.CrossEntropyLoss()
    train_loop(cnn_emnist, optim, lfun, train_loader, test_loader, n_batches=10_000 ,device='cuda')
    #
    sys.exit()
    # exit()
    # train, test, n_dim, n_classes = get_ionosphere(random_state=42, batch_size=4)
    # train, test, n_dim, n_classes = get_iris(random_state=42, batch_size=8)
    # train, test, n_dim, n_classes = get_covtype(random_state=42, batch_size=1000)
    # train, test, n_dim, n_classes = get_classification(batch_sizes=(8, 300))
    # train, test, n_dim, n_classes = get_waveform(batch_size=8)
    # train, test, n_dim, n_classes = get_breast_cancer(42)
    from lxg.datasets import get_classification, _wrap_numpy_to_loader, get_beans
    from sklearn.tree import DecisionTreeClassifier as DTC
    hidden_layers = [8,8]
    n_noisy = 0
    n_informative = 5
    n_redundant = 0
    n_repeated = 20
    n_features = n_informative + n_redundant + n_repeated + n_noisy
    n_classes = int(np.ceil(0.5 * 2**n_informative + 1))
    kwargs_classif = dict(n_samples=5000, n_features=n_features, n_informative=n_informative,
                          n_redundant=n_redundant, n_repeated=n_repeated,
                          n_classes=n_classes, n_clusters_per_class=1, flip_y=0.01)

    np_train, np_test, n_dim, n_classes = get_classification(batch_sizes=(32, 1050), kwargs=kwargs_classif, as_torch=False)
    train, test = (_wrap_numpy_to_loader(X=np_train[0], Y=np_train[1], batch_size=32),
                   _wrap_numpy_to_loader(X=np_test[0], Y=np_test[1], batch_size=1050))


    # np_train, np_test, n_dim, n_classes = get_beans(42, as_torch=False)
    # train, test = (_wrap_numpy_to_loader(X=np_train[0], Y=np_train[1], batch_size=32),
    #                _wrap_numpy_to_loader(X=np_test[0], Y=np_test[1], batch_size=1050))

    # dt = DTC(max_depth=5, max_leaf_nodes=n_classes*2)
    # dt2 = DTC(max_depth=5, max_leaf_nodes=n_classes)
    # dt.fit(np_train[0], np_train[1])
    # dt_train_acc = dt.score(np_train[0], np_train[1])
    # dt_test_acc = dt.score(np_test[0], np_test[1])
    # dt2.fit(np_train[0], np_train[1])
    # dt2_train_acc = dt2.score(np_train[0], np_train[1])
    # dt2_test_acc = dt2.score(np_test[0], np_test[1])
    # print(f"dt test acc = {dt_test_acc}")
    # print(f" depth = {dt.get_depth()}, n_leaves = {dt.get_n_leaves()}")
    # dnf = DNFClassifier.from_DT(dt, verbose=True)
    # dnf2 = DNFClassifier.from_DT(dt2, verbose=True)
    #
    #
    def _get_coverage_dims(dnf, dims):
        matched_dims = []
        for _class in dnf:
            matched_dims.append([])
            for terms in _class:
                for literal in terms:
                    dim = literal[0]
                    if dim in dims and dim not in matched_dims[-1]:
                        matched_dims[-1].append(dim)
        return matched_dims

    def _eval_dnf_make_classification(dnf, _informative):
        _informative_covered = _get_coverage_dims(dnf, _informative)
        _informative_covered = set(chain.from_iterable(_informative_covered))  # set() takes care of unique
        _coverage_inf = len(_informative_covered) / len(_informative)
        return _coverage_inf


    def print_dnf(d):
        for t in d:
            print(t)
    def merge_dnfs_naive(DNF1, DNF2):
        DNF_merged = []
        for c1, c2 in zip(DNF1, DNF2):
            print(f"\nc1"); print_dnf(c1)
            print(f"c2"); print_dnf(c2)
            c_merged = c1
            for t in c2:
                if t not in c_merged:
                    c_merged.append(t)
            DNF_merged.append(c_merged)
            print(f"merged:"); print_dnf(c_merged);print("")
        return DNF_merged
    # merged_rules = merge_dnfs_naive(dnf.rules, dnf2.rules)
    # dnf3 = DNFClassifier(merged_rules)
    #
    # dnf2_desc_te = dnf2.describe(np_test[0])
    # dnf2_recall_te = np.mean([y in d for y, d in zip(np_test[1], dnf2_desc_te)])
    #
    # dnf3_desc_te = dnf2.describe(np_test[0])
    # dnf3_recall_te = np.mean([y in d for y, d in zip(np_test[1], dnf3_desc_te)])
    #
    #
    #
    # desc_tr = dnf.describe(np_train[0])
    # desc_te = dnf.describe(np_test[0])
    # descriptions = dnf.describe(np_test[0])
    # dnf_recall_tr = np.mean([y in d for y, d in zip(np_train[1], desc_tr)])
    # dnf_recall_te = np.mean([y in d for y, d in zip(np_test[1], desc_te)])
    # recall_mode = 'weighted'
    # dt_recall_tr = sklearn.metrics.recall_score(y_true=np_train[1], y_pred=dt.predict(np_train[0]), average=recall_mode)
    # dt_recall_te = sklearn.metrics.recall_score(y_true=np_test[1], y_pred=dt.predict(np_test[0]), average=recall_mode)
    # dnf_acc_tr = np.mean(dnf(np_train[0]) == np_train[1])
    # dnf_acc_te = np.mean(dnf(np_test[0]) == np_test[1])
    # print(f"\t\t acc DT,\t acc DNF,\tany-correct DNF")
    # print(f"train \t {dt_train_acc:.4f},\t{dnf_acc_tr:.4f},\t{dnf_recall_tr:.4f}")
    # print(f"test  \t {dt_test_acc:.4f},\t{dnf_acc_te:.4f},\t{dnf_recall_te:.4f}")
    # print("(multi-class -> DT recall+weighted == accuracy)")


    def _dnf_acc_recall(dnf, X, Y):
        dnf_acc = np.mean(dnf(X) == Y)
        descriptions = dnf.describe(X)
        dnf_recall = np.mean([y in d for y, d in zip(Y, descriptions)])
        return [dnf_acc, dnf_recall]


    n_trees = 20
    seeds = np.random.randint(0, 42000, n_trees)
    dnfs = []
    dnfs_acc_rec = []
    dnfs_cov = []
    important_dims = list(range(n_informative)) + list(np.arange(n_repeated) + n_informative + n_redundant)
    for i in range(n_trees):
        print(i)
        dt = DTC(max_depth=5, max_leaf_nodes=min(n_classes**2, 3*n_classes), random_state=seeds[i])
        dt.fit(np_train[0], np_train[1])
        dt_train_acc = dt.score(np_train[0], np_train[1])
        dt_test_acc = dt.score(np_test[0], np_test[1])
        dnf = DNFClassifier.from_DT(dt)
        dnfs.append(dnf)
        dnfs_acc_rec.append(_dnf_acc_recall(dnf, np_test[0], np_test[1]))
        _dims_covered = set(chain.from_iterable(_get_coverage_dims(dnf, important_dims)))
        dnfs_cov.append(_dims_covered)

    merged = dnfs[0]
    merged_cov = [_eval_dnf_make_classification(merged, important_dims)]
    merged_acc_rec = []
    for i in range(1, len(dnfs)):
        merged = DNFClassifier(merge_dnfs_naive(merged, dnfs[i]))
        merged_acc_rec.append(_dnf_acc_recall(merged, np_test[0], np_test[1]))
        merged_cov.append(_eval_dnf_make_classification(merged, important_dims))
    for a, b in zip(merged_acc_rec[:-1], merged_acc_rec[1:]):
        # sanity check: we should never get less coverage when adding more conjunctions
        assert a[1] <= b[1]


    n_loop = 1
    accs = []
    for i in range(n_loop):
        ff_cov = make_ff([n_dim]+hidden_layers+[n_classes])
        print(n_dim, n_classes)
        path = './data/'
        modelname= 'ff_wf'
        # optim = Adam(ff_cov.parameters())
        optim = AdamW(ff_cov.parameters())#, amsgrad=True)
        lfun = torch.nn.CrossEntropyLoss()
        accs.append(train_loop(ff_cov, optim, lfun, train, test, device='cpu',
                               pth=path, modelname=modelname, n_batches=2000))



    import matplotlib.pyplot as plt
    for acc in accs:
        plt.plot(acc)
    plt.show()

    # data, testset, size_vocab, n_classes = get_agnews(random_state=42, batch_sizes=(64, 200))
    #
    # nlp_model = BiLSTMClassif(nr_classes=n_classes, embed_dim=128, hid_size=256, vocab_size=size_vocab)
    # # nlp_model = nn.DataParallel(nlp_model)
    #
    # optim = Adam(nlp_model.parameters())
    # lfun = torch.nn.CrossEntropyLoss()
    # train_loop(nlp_model, optim, lfun, data, testset)