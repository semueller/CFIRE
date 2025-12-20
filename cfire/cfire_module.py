import logging
from copy import deepcopy

import numpy as np

import time

from sklearn.metrics import accuracy_score

from cfire.pruning import decide_by_wins, prune_rules
from cfire.pruning_metrics import compute_rule_metrics
from lxg.models import DNFClassifier

from cfire.gely import gely_discriminatory, ItemsetNode
from cfire.nodeselection import _comp_greedy_cover

from typing import NamedTuple
class ItemsetNodeCollection(NamedTuple):
    """ holds a list of `ItemsetNodes` and their corresponding support set """
    class_support: list[set[int]]
    nodes: list[ItemsetNode]

class CFIRE:

    def __init__(self,
                 localexplainer_fn,
                 inference_fn,
                 expl_binarization_fn=None,
                 frequency_threshold=0.01,
                 explanations=None,
                 meta_data=None, ):

        # Callables
        self._localexplainer_fn = localexplainer_fn
        self._inference_fn = inference_fn
        self._expl_binarization_fn = expl_binarization_fn if expl_binarization_fn is not None else -1
        self._composition_strategy = _comp_greedy_cover

        # Data fields
        self._explanations = explanations
        if self._explanations is not None:
            if localexplainer_fn is not None:
                logging.warning("localexplainer_fn passed although pre-calculated explanations are used")
            if localexplainer_fn is not None:
                logging.warning("inference_fn passed although pre-calculated explanations are used")
        self._binarized_explanations = None
        self._data = None
        self._labels = None
        self._itemsetnodes: list[list[ItemsetNode]] = None
        # final DNFClassifier
        self.dnf: DNFClassifier = None

        # Hyperparameters
        self._frequency_threshold = frequency_threshold

        self._is_fit = False
        self._compute_times = {}

        self._meta_data = None  # dictionary placeholder to hold any info, eg about hyperparameters of model
        self._verbose = True

    def __call__(self, X, explain: bool = False):
        """
        Forward to the underlying DNFClassifier.
        """
        return self.dnf(X, explain=explain)

    def _calc_explanations(self):
        time_expl = time.time()
        self._explanations = self._localexplainer_fn(self._inference_fn, self._data, self._labels)
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
                         }
            target_class_itemsetnodes = gely_discriminatory(**gely_args)
            self._itemsetnodes.append(target_class_itemsetnodes)
        self._compute_times['_calculate_rule_candidates'] = time.time() - _start_time
        return


    def _compose_rule_model(self):
        _start_time = time.time()
        n_classes = len(np.unique(self._labels))
        _DNF = []
        dummy_rule = [(-1,(np.nan, np.nan))]
        for c in range(n_classes):
            nodes_c = self._itemsetnodes[c]
            if nodes_c is None or len(nodes_c) == 0:
                _DNF.append([deepcopy(dummy_rule)])
                continue
            supp, _all_nodes = nodes_c
            root = _all_nodes[0]; assert root.parent is None
            freq_nodes = root.get_frequent_children()
            if freq_nodes is None or len(freq_nodes) == 0:
                _DNF.append([deepcopy(dummy_rule)])
                continue
            _f, _s = [], []
            for f, s in zip(freq_nodes, supp):
                if len(s) > 0:
                    _f.append(f); _s.append(s)
            freq_nodes, supp = _f, _s
            class_dnf = self._composition_strategy(supp, freq_nodes)
            _DNF.append(class_dnf)
        DNF = self.__merge_single_class_dnfs_multiclass_dnf(_DNF)
        if DNF.tie_break == "accuracy":
            DNF.compute_rule_performance(self._data, self._labels)
        self._compute_times['_compose_rule_model'] = time.time() - _start_time
        self.dnf = DNF
        return

    def __merge_single_class_dnfs_multiclass_dnf(self, dnfs):
            rules = [dnf.rules[0] for dnf in dnfs]  # 1 or 0?
            return DNFClassifier(rules, 'accuracy')


    def fit(self, X=None, Y=None, save_interim=False):
        if self._is_fit:
            raise Exception('CFIRE is already fit')
        if save_interim:
            raise NotImplementedError
        self._data = X
        self._labels = Y
        if self._verbose:
            print("CFIRE: Calc Expls")
        self._calc_explanations()
        if self._verbose:
            print(f"took {self._compute_times['_calc_explanations']+self._compute_times['expl_binarization_fn']:.4f}s")
        if self._verbose:
            print("CFIRE: Itemset Mining and Rule Candidates")
        self._calculate_rule_candidates()
        if self._verbose:
            print(f"took {self._compute_times['_calculate_rule_candidates']:.4f}s")
        if self._verbose:
            print("CFIRE: Select rules")
        self._compose_rule_model()
        if self._verbose:
            print(f"took {self._compute_times['_compose_rule_model']:.4f}s")

        return

    def eval(self):
        pass

