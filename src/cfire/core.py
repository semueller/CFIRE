import logging
import warnings
import pickle
import gzip
from pathlib import Path

import numpy
import numpy as np

import time

from .rules import DNFClassifier

from .mining import gely_discriminatory, ItemsetNode
from .cover import _comp_greedy_cover
from .binarise import binarise_threshold, relevance_scores
from .encoding import Encoding

from typing import NamedTuple, Optional


class ItemsetNodeCollection(NamedTuple):
    """ holds a list of `ItemsetNodes` and their corresponding support set """
    class_support: list[set[int]]
    nodes: list[ItemsetNode]


class CFIRE:

    def __init__(self, localexplainer_fn, explanations: Optional[numpy.ndarray], inference_fn, expl_binarization_fn=None,
                 frequency_threshold=0.01, composition_strategy=_comp_greedy_cover, max_dt_depth=None, meta_data=None,
                 random_state=None, item_order="relevance", feature_types=None):
        """
        :param expl_binarization_fn: maps the explanation matrix (n_samples, n_features) to a binary matrix of relevant
            features; default: `binarise_threshold` (row-wise max-abs normalisation, values below 0.01 are not relevant).
        :param frequency_threshold: minimum support (fraction of samples of a class) of a closed itemset.
        :param max_dt_depth: exclusive upper bound of the depth grid searched (5-fold CV) for the shallow trees that turn
            an itemset into rules; the deepest tree considered has depth max_dt_depth - 1.
            None (default): adaptive, grid range(1, max(8 - |itemset|, 2)), i.e. depth up to 6 for single features and
            less for larger itemsets. This is what the CFIRE paper (2025) and, for almost all itemsets, the pruning
            study ran (both papers state "maximum depth 7").
            An int fixes the grid to range(1, max_dt_depth) for all itemsets.
            Small samples cap it further: fewer than 5 samples in a class -> depth <= 2; at most 10 samples -> depth 1;
            at most 2 samples in a class -> no tree, the itemset's box is kept as the rule.
        :param random_state: seed for the decision trees (ties between equally good splits are broken at random);
            None reproduces the unseeded behaviour of the original code.
        :param item_order: order in which features (items) are enumerated by the closed-itemset miner, per class.
            "relevance" (default; CFIRE paper 2025, gely_sort_items=True): descending column sums of the normalised,
            thresholded scores (`relevance_scores`) of the class's explanations; "natural": feature index order.
            The closed itemsets are the same either way, but each candidate's tree is fitted on the samples passed
            down from its parent node, which depends on the enumeration order.
            AUDIT: effect not yet measured systematically; remove if it makes no measurable difference.
        :param feature_types: optional list with "numeric" or "categorical" per column of the data passed to fit (the
            black box's input). Categorical columns (category codes; binary columns included) are one-hot encoded
            internally, one indicator per value observed in the fit data (`cfire.encoding`); the relevance of such a
            column (binarised explanation, relevance scores) is copied to all of its indicators, so mined itemsets
            contain a column's indicators all or none. The rule model stores the encoding and takes the input columns.
            None or all "numeric": no encoding, behaviour unchanged.
        """
        if item_order not in ("relevance", "natural"):
            raise ValueError(f"item_order must be 'relevance' or 'natural', got {item_order!r}")
        self._item_order = item_order
        self._feature_types = None if feature_types is None else list(feature_types)
        self.encoding: Optional[Encoding] = None  # set in fit if feature_types has categorical columns
        # MISSING:
        # - behavior of grid search during rule fitting (min/max_depth),
        # - inference strategy for DNF

        # Callables
        self._localexplainer_fn = localexplainer_fn
        self._inference_fn = inference_fn
        self._expl_binarization_fn = expl_binarization_fn if expl_binarization_fn is not None else binarise_threshold
        self._composition_strategy = composition_strategy

        # input samples used for fitting CFIRE object
        self._data = None
        self._labels = None

        # feature importance ("explanations") for the given input samples -> (n_samples, n_features)
        self._explanations = None
        self._binarized_explanations = None

        # optionally use pre-calculated local explanations instead of callable function
        if explanations is not None:
            self._explanations = explanations

            if localexplainer_fn is not None:
                logging.warning("localexplainer_fn passed although pre-calculated explanations are used")
            if localexplainer_fn is not None:
                logging.warning("inference_fn passed although pre-calculated explanations are used")

        # a list containing the results, with index corresponding to one class each
        # for each class a tuple with 2 elements is stored
        #   - *support sets*: The first tuple element is a list of sets with integer values, which correspond to sample inidices,
        #       when only considering samples of the current class, see -> `data_target` vs. `data_other`.
        #       Only the support sets for remaining *frequent* itemsets are kept (TODO: why?)
        #   - *item sets*: the second tuple element is a list of gely.ItemsetNode's, containing *all* (not only frequent) itemsets,
        #       - therefore not all item sets have a dnf_classifier/accuracy/...
        #       - the first itemnode is "root/parent" node, including all relevant features
        # TODO: typing probably not correct, list[tuple[list[set[int]], list[ItemsetNode]]]
        self._itemsetnodes: list[list[ItemsetNode]] = None

        # store input to composition strategy
        self.frequent_nodes: list[ItemsetNodeCollection] = []

        # final DNFClassifier
        self.dnf: DNFClassifier = None

        # Hyperparameters
        self._max_dt_depth = max_dt_depth
        self._random_state = random_state
        self._frequency_threshold = frequency_threshold

        self._is_fit = False
        self._compute_times = {}

        self._meta_data = None  # dictionary placeholder to hold any info, eg about hyperparameters of model
        self._verbose = True

    def __call__(self, X, explain=False):
        return self.dnf(X, explain=explain)

    def _calc_explanations(self):
        time_expl = time.time()

        if self._explanations is None:
            self._explanations = self._localexplainer_fn(self._inference_fn, self._data, self._labels)

            # cast to numpy before further processing (e.g. torch tensors), as it can crash future functions otherwise
            if hasattr(self._explanations, "detach"):
                self._explanations = self._explanations.detach().cpu().numpy()
            self._explanations = np.asarray(self._explanations)

        self._compute_times['_calc_explanations'] = time.time() - time_expl

        if self._explanations.shape != self._X_input.shape:
            raise ValueError(f"explanations have shape {self._explanations.shape}, data {self._X_input.shape}")
        time_bin_exp = time.time()
        # binarised in the black box's input space, then copied to the indicators of categorical columns
        self._binarized_explanations = self._expl_binarization_fn(self._explanations)
        if self.encoding is not None:
            self._binarized_explanations = self.encoding.expand(self._binarized_explanations)
        self._compute_times['expl_binarization_fn'] = time.time() - time_bin_exp
        return

    def _calculate_rule_candidates(self):

        self._itemsetnodes = []
        _start_time = time.time()
        # split by classes
        for _target_class in sorted(np.unique(self._labels)):
            mask_targets = self._labels == _target_class
            data_target = self._data[mask_targets]
            data_other = self._data[~mask_targets]
            assert len(data_target) > 0 and len(data_other) > 0
            expls_target = self._binarized_explanations[mask_targets]
            if self._item_order == "relevance":  # AUDIT, see __init__
                if self._expl_binarization_fn is not binarise_threshold:
                    raise ValueError('item_order="relevance" needs the default binariser; use item_order="natural"')
                scores = relevance_scores(np.asarray(self._explanations)[mask_targets])
                if self.encoding is not None:
                    scores = self.encoding.expand(scores)
                item_order = np.argsort(np.sum(scores, 0)).squeeze()[::-1]  # as cfire.py (2025), gely_sort_items
            else:
                item_order = np.arange(self._data.shape[1])
            gely_args = {'B': expls_target,
                         'threshold': self._frequency_threshold,
                         'X_target': data_target,
                         'X_other': data_other,
                         'item_order': item_order,
                         # 'model_callable': self._inference_fn,
                         'compute_rules': True,
                         'dt_kwargs': self._dt_kwargs(),
                         'subset_test': True # TODO: enable this?
                         }

            # TODO: uses `generate_synthetic_data`?
            # -> update: still exist in code, but supposedly not used
            # TODO: not only closed+frequent, but also "discriminatory"
            # -> update: discriminatory just means that each class is handled individually
            target_class_itemsetnodes = gely_discriminatory(**gely_args)
            self._itemsetnodes.append(target_class_itemsetnodes)
        self._compute_times['_calculate_rule_candidates'] = time.time() - _start_time
        return

    def _dt_kwargs(self):
        kw = {}
        if self._max_dt_depth is not None:
            kw['max_depth'] = self._max_dt_depth
        if self._random_state is not None:
            kw['random_state'] = self._random_state
        return kw

    def _compose_rule_model(self):
        _start_time = time.time()
        n_classes = len(np.unique(self._labels))
        _DNF = []

        assert len(self._itemsetnodes) == n_classes

        for class_idx in range(n_classes):
            nodes_c = self._itemsetnodes[class_idx]
            if nodes_c is None or len(nodes_c) == 0 or len(nodes_c[1]) == 0:
                warnings.warn(f"CFIRE: no closed frequent itemsets for class {class_idx}; the class gets no rules "
                              f"(frequency_threshold={self._frequency_threshold})")
                _DNF.append(DNFClassifier(rules=[[]]))
                self.frequent_nodes.append(ItemsetNodeCollection([], []))
                continue

            # idx-set of supporting samples, ItemSetNode
            class_support, _all_nodes = nodes_c
            root = _all_nodes[0]; assert root.parent is None
            freq_nodes = root.get_frequent_children()
            if freq_nodes is None or len(freq_nodes) == 0:
                warnings.warn(f"CFIRE: no frequent itemsets for class {class_idx}; the class gets no rules "
                              f"(frequency_threshold={self._frequency_threshold})")
                _DNF.append(DNFClassifier(rules=[[]]))
                self.frequent_nodes.append(ItemsetNodeCollection([], []))
                continue
            _f, _s = [], []
            for f, s in zip(freq_nodes, class_support):
                if len(s) > 0:
                    _f.append(f); _s.append(s)
            freq_nodes, class_support = _f, _s

            # save input to composition strategy for future analysis
            self.frequent_nodes.append(ItemsetNodeCollection(class_support, freq_nodes))

            class_dnf = self._composition_strategy(class_support, freq_nodes)
            _DNF.append(class_dnf)

        DNF = self.__merge_single_class_dnfs_multiclass_dnf(_DNF)
        if DNF.tie_break == "accuracy": # TODO: we check for accuracy here and hardcode it as accuracy?
            DNF.compute_rule_performance(self._data, self._labels)
        if self.encoding is not None:  # after scoring: self._data is already encoded
            DNF = self.__with_encoding(DNF)
        self._compute_times['_compose_rule_model'] = time.time() - _start_time
        self.dnf = DNF
        return

    def __with_encoding(self, dnf):
        out = DNFClassifier(dnf.rules, dnf.tie_break, encoding=self.encoding)
        out.rule_performances = dnf.rule_performances
        return out

    def __merge_single_class_dnfs_multiclass_dnf(self, dnfs):
        rules = [dnf.rules[0] for dnf in dnfs]  # 1 or 0? # TODO: we ignore single class dnf tie breaker, do we need it at any point?
        return DNFClassifier(rules, 'accuracy')

    def _verbose_print(self, message):
        """print wrapper, that only prints if verbose flag is set """
        if self._verbose:
            print(message)

    def fit(self, X=None, Y=None, save_interim=False):
        if self._is_fit:
            raise Exception('CFIRE is already fit')
        if save_interim:
            raise NotImplementedError
        self._X_input = np.asarray(X)
        if self._feature_types is not None:
            enc = Encoding.fit(X, self._feature_types)
            self.encoding = None if enc.is_identity else enc
        # the miner and the rule trees work on the encoded columns
        self._data = X if self.encoding is None else self.encoding.transform(X)
        self._labels = Y

        # get explanations / feature importances (self._binarized_explanations, self._explanations)
        self._verbose_print("CFIRE: Calc Expls")
        self._calc_explanations()
        self._verbose_print(f"took {self._compute_times['_calc_explanations']+self._compute_times['expl_binarization_fn']:.4f}s")

        # closed frequent itemset mining (self._itemsetnodes)
        self._verbose_print("CFIRE: Itemset Mining and Rule Candidates")
        self._calculate_rule_candidates()
        self._verbose_print(f"took {self._compute_times['_calculate_rule_candidates']:.4f}s")

        self._verbose_print("CFIRE: Select rules")
        self._compose_rule_model()
        self._verbose_print(f"took {self._compute_times['_compose_rule_model']:.4f}s")
        return

    def eval(self):
        pass

    def partial_dump(self, file_path: Path, *, compress_level: int = 6) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        attributes = ["dnf", "_itemsetnodes", "frequent_nodes", "_compute_times"]
        state = {k: getattr(self, k) for k in attributes}

        with gzip.open(file_path, mode="wb", compresslevel=compress_level) as file:
            pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
