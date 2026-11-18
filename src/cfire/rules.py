

import sklearn.tree






from itertools import combinations


import numpy as np










from copy import deepcopy


def merge_rules(a, b):
    """Merge two rules (conjunctions of (dim, (lo, hi))) of one class if one can replace both; port of
    lxg.models.merge_rules (CFIRE paper 2025, tag wcxai2025), behaviour unchanged.
    Returns (merged, None) if mergeable, else (a, b).
    - same features: identical -> a; one contains the other in every feature -> the containing one; equal in all but
      one feature where the intervals overlap -> one rule with the union of the two intervals there
    - features of one rule a subset of the other's: the shorter rule, if its intervals contain the longer rule's
      intervals on the shared features
    - otherwise not mergeable"""

    def get_dims(r):
        return set([t[0] for t in r])

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

    if not (da <= db or db <= da):  # rules use different dims
        return a, b

    if da == db:
        new_rule = []
        if np.all([ta == tb for ta, tb in zip(a, b)]):  # identical
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
                    new_rule.append((ta[0], (min(ta[1][0], tb[1][0]), max(ta[1][1], tb[1][1]))))
                else:  # no overlap: keep both
                    return a, b
            return new_rule, None
        return a, b

    if len(db) < len(da):
        a, b = b, a
        da, db = db, da

    # a is the shorter, more general rule: mergeable if its intervals contain b's on a's features
    def term_for_dim(terms, dim):
        for t in terms:
            if t[0] == dim: return t
        return None
    if not all(contains(ta[1], term_for_dim(b, ta[0])[1]) for ta, tb in zip(a, b)):
        return a, b
    return a, None


class DNFClassifier:
    def __init__(self, rules: list[list[list[tuple[int, tuple[float, float]]]]], tie_break="first", encoding=None):
        """
        :param encoding: optional `cfire.encoding.Encoding`. The rules then refer to the encoded columns (indicators
            of categorical input columns); predict, describe and compute_rule_performance take the input columns and
            encode them first. None: the rules refer to the input columns directly.
        """
        # rule format: dimension, interval -> tuple(dimension, tuple(lower_limit, upper_limit))
        self.encoding = None if encoding is None or encoding.is_identity else encoding
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

    def _encode(self, samples):
        return samples if self.encoding is None else self.encoding.transform(samples)

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

    # ---- serialisation ------------------------------------------------------------------------------------------
    def to_dict(self, feature_names=None) -> dict:
        """JSON-serialisable form: per class a list of rules (conjunctions of [feature, lower, upper]; None = unbounded,
        a literal holds for lower <= x <= upper) and per rule its statistics on the data the model was scored on
        (the tie-break "accuracy" uses the rule's precision). With an encoding, features are encoded columns and the
        encoding is stored (format version 2)."""
        def bound(v):
            v = float(v)
            return None if not np.isfinite(v) else v

        classes = []
        for c, class_rules in enumerate(self.rules):
            stats = self.rule_performances.get(c, {})
            rules, rule_stats = [], []
            for clause in class_rules:
                rules.append([[int(d), bound(lo), bound(hi)] for d, (lo, hi) in clause])
                st = stats.get(tuple(clause))
                rule_stats.append(None if st is None else
                                  {"precision": float(st["accuracy"]), "recall": float(st["recall"]), "f1": float(st["f1"])})
            classes.append({"class": c, "rules": rules, "rule_stats": rule_stats})
        out = {"format": "cfire.DNFClassifier/1", "tie_break": self.tie_break, "classes": classes}
        if self.encoding is not None:
            out["format"] = "cfire.DNFClassifier/2"
            out["encoding"] = self.encoding.to_dict()
        if feature_names is not None:
            out["feature_names"] = list(map(str, feature_names))
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "DNFClassifier":
        def bound(v, default):
            return default if v is None else float(v)

        rules = [[[(int(f), (bound(lo, -np.inf), bound(hi, np.inf))) for f, lo, hi in rule] for rule in c["rules"]]
                 for c in d["classes"]]
        encoding = None
        if d.get("encoding") is not None:
            from .encoding import Encoding
            encoding = Encoding.from_dict(d["encoding"])
        dnf = cls(rules, d.get("tie_break", "first"), encoding=encoding)
        for c, cdict in enumerate(d["classes"]):
            perf = {}
            for clause, st in zip(dnf.rules[c], cdict.get("rule_stats", [])):
                if st is not None:
                    perf[tuple(clause)] = {"accuracy": st["precision"], "recall": st["recall"], "f1": st["f1"]}
            dnf.rule_performances[c] = perf
        return dnf

    def purge_dummy_rules(self):
        purged_rules = []
        for class_rules in self.rules:
            purged_class = []
            for term in class_rules:
                term = list(filter(lambda x: x != (-1, (np.nan, np.nan)), term))
                purged_class.append(term)
            purged_rules.append(purged_class)
        self.rules = purged_rules


    def __set_rules(self, new_rules):
        self.rules = deepcopy(new_rules)
        self.n_classes = len(self.rules)
        self.n_literals = self.__comp_n_literals()
        self.n_rules = self.__comp_n_rules()
        return

    def remove_empirically_redundant_rules(self, X) -> list:
        """Remove rules that are redundant on the samples X; port of lxg.models.DNFClassifier
        .remove_empirically_redundant_rules with min_complexity=False, the setting of the CFIRE paper (2025), whose
        reported rule models went through this step after the greedy cover (cfire_experiments/03_eval_rules.py at
        tag wcxai2025). Behaviour unchanged, including that the last rule of a class is not checked for empty support.

        Per class, with the support of a rule = the samples of X it matches (any label):
        - a rule without support is removed
        - a rule whose support is a strict subset of another rule's support is removed
        - of two rules with equal support, the one whose features are a subset of the other's is removed (both are
          kept if neither feature set contains the other)
        then rules are merged (`simplify_merge_rules`). Predictions on X with tie-break "first" must not change
        (checked; RuntimeError otherwise). X: rows in the rule model's columns (encoded columns if the rules refer to
        them; this method does not encode). Rule statistics are reset; recompute them with compute_rule_performance.
        Returns the removed rules as (class, rule)."""
        _tie_break = self.tie_break
        self.tie_break = "first"
        pred_before = self.__predict_raw(X)
        reduced_rules, removed = [], []
        for c, class_dnf in enumerate(self.rules):
            supports = [set(np.flatnonzero([self.__clause_match_sample(x, clause) for x in X])) for clause in class_dnf]
            rules_c = deepcopy(class_dnf)
            removed_clause = True
            while removed_clause:
                removed_clause = False
                for i in range(len(rules_c) - 1):
                    s1 = supports[i]
                    if len(s1) == 0:
                        removed.append((c, rules_c[i])); del rules_c[i]; del supports[i]
                        removed_clause = True; break
                    for j in range(i + 1, len(rules_c)):
                        s2 = supports[j]
                        if s1 == s2:
                            dims1 = set(int(t[0]) for t in rules_c[i])
                            dims2 = set(int(t[0]) for t in rules_c[j])
                            if dims1 <= dims2:  # rule j more specific
                                removed.append((c, rules_c[i])); del rules_c[i]; del supports[i]
                                removed_clause = True; break
                            elif dims2 <= dims1:
                                removed.append((c, rules_c[j])); del rules_c[j]; del supports[j]
                                removed_clause = True; break
                        elif s1 < s2:
                            removed.append((c, rules_c[i])); del rules_c[i]; del supports[i]
                            removed_clause = True; break
                        elif s2 < s1:
                            removed.append((c, rules_c[j])); del rules_c[j]; del supports[j]
                            removed_clause = True; break
                    if removed_clause:
                        break
            reduced_rules.append(rules_c)
        self.__set_rules(reduced_rules)
        self.simplify_merge_rules()
        self.rule_performances = {c: {} for c in range(self.n_classes)}
        if not np.array_equal(pred_before, self.__predict_raw(X)):
            raise RuntimeError("remove_empirically_redundant_rules changed predictions on X")
        self.tie_break = _tie_break
        return removed

    def simplify_merge_rules(self):
        """merge pairs of rules of a class with `merge_rules` until no pair is mergeable; port of
        lxg.models.DNFClassifier.simplify_merge_rules (CFIRE paper 2025)"""
        new_rules = []
        for cdnf in self.rules:
            new_cdnf = list(cdnf)
            merged = True
            while merged:
                merged = False
                for i in range(len(new_cdnf)):
                    r1 = new_cdnf[i]
                    for j in range(i + 1, len(new_cdnf)):
                        r2 = new_cdnf[j]
                        m1, m2 = merge_rules(a=r1, b=r2)
                        if m2 is None:
                            new_cdnf.remove(r1)
                            new_cdnf.remove(r2)
                            new_cdnf.append(m1)
                            merged = True
                            break
                    if merged:
                        break
            new_rules.append(new_cdnf)
        self.__set_rules(new_rules)

    def __predict_raw(self, X):
        """predict on rows already in the rules' columns (no encoding)"""
        return np.array([self.__predict_sample(x) for x in X])

    def __comp_n_literals(self) -> int:
        """literals summed over all rules; with an encoding, literals on the indicators of one categorical input column
        count as one (one condition on that feature)"""
        n_literals = 0
        for c in range(self.n_classes):
            c_dnf = self.rules[c]
            if self.encoding is None:
                n_literals += sum([len(term) for term in c_dnf])
            else:
                n_literals += sum([len({int(self.encoding.source[d]) for d, _ in term}) for term in c_dnf])
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
        X = self._encode(X)
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


    ## todo, not sure why remove here

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
        samples = self._encode(samples)
        predictions = []
        for sample in samples:
            predictions.append(self.__predict_sample(sample, explain))
        if not explain:
            return np.array(predictions)
        else:
            return predictions



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
        samples = self._encode(samples)
        dscriptions = []
        for sample in samples:
            dscriptions.append(self.__describe_sample(sample))
        return dscriptions






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


