import logging
import pickle
from argparse import ArgumentError
from copy import deepcopy
import gzip
from pathlib import Path

import numpy
import numpy as np
import torch

import time

from lxg.models import DNFClassifier

from cfire.gely import gely_discriminatory, ItemsetNode
from cfire.nodeselection import _comp_greedy_cover

from typing import NamedTuple, Optional


class ItemsetNodeCollection(NamedTuple):
    """ holds a list of `ItemsetNodes` and their corresponding support set """
    class_support: list[set[int]]
    nodes: list[ItemsetNode]


class CFIRE:

    def __init__(self, localexplainer_fn, explanations: Optional[numpy.ndarray], inference_fn, expl_binarization_fn=None,
                 frequency_threshold=0.01, composition_strategy=_comp_greedy_cover, max_dt_depth=7, meta_data=None):
        # MISSING:
        # - behavior of grid search during rule fitting (min/max_depth),
        # - inference strategy for DNF

        # Callables
        self._localexplainer_fn = localexplainer_fn
        self._inference_fn = inference_fn
        self._expl_binarization_fn = expl_binarization_fn if expl_binarization_fn is not None else -1
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

            # cast it to numpy before further processing, as it can crash future functions otherwise
            if isinstance(self._explanations, torch.Tensor):
                self._explanations = self._explanations.cpu().numpy()

        self._compute_times['_calc_explanations'] = time.time() - time_expl

        time_bin_exp = time.time()
        self._binarized_explanations = self._expl_binarization_fn(self._explanations)
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
            item_order = np.arange(self._data.shape[1])  #np.argsort(np.sum(expls_target, axis=1))[::-1]
            gely_args = {'B': expls_target,
                         'threshold': self._frequency_threshold,
                         'X_target': data_target,
                         'X_other': data_other,
                         'item_order': item_order,
                         # 'model_callable': self._inference_fn,
                         'compute_rules': True,
                         'dt_kwargs': {'max_depth': self._max_dt_depth},
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

    def _compose_rule_model(self):
        _start_time = time.time()
        n_classes = len(np.unique(self._labels))
        _DNF = []

        assert len(self._itemsetnodes) == n_classes

        for class_idx in range(n_classes):
            nodes_c = self._itemsetnodes[class_idx]
            if nodes_c is None or len(nodes_c) == 0 or len(nodes_c[1]) == 0:
                _DNF.append(DNFClassifier(rules=[[]]))
                self.frequent_nodes.append(ItemsetNodeCollection([], []))
                continue

            # idx-set of supporting samples, ItemSetNode
            class_support, _all_nodes = nodes_c
            root = _all_nodes[0]; assert root.parent is None
            freq_nodes = root.get_frequent_children()
            if freq_nodes is None or len(freq_nodes) == 0:
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
        self._compute_times['_compose_rule_model'] = time.time() - _start_time
        self.dnf = DNF
        return

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
        self._data = X
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
