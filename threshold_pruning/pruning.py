from __future__ import annotations

from copy import deepcopy

from dataclasses import dataclass
from typing import  Sequence, Set

from sklearn.metrics import accuracy_score

from threshold_pruning.metrics import Rules
from threshold_pruning.pruning_metrics import RuleKey, RuleMetrics, compute_rule_metrics

from lxg.models import DNFClassifier
from cfire.cfire_module import CFIRE

@dataclass(frozen=True)
class PruningDecision:
    to_remove: Set[RuleKey]
    rationale: str

def decide_by_wins(metrics: RuleMetrics, win_threshold: int) -> PruningDecision:
    """
    Classic policy from the original script:
    remove rules whose absolute win count <= win_threshold.
    """
    to_remove: Set[RuleKey] = {
        metrics.clause_keys[i]
        for i, w in enumerate(metrics.wins)
        if int(w) <= int(win_threshold)
    }
    rationale = f"Removed {len(to_remove)}/{len(metrics.clause_keys)} rules with wins <= {win_threshold}"
    return PruningDecision(to_remove=to_remove, rationale=rationale)

def prune_rules(rule_tree: Sequence[Sequence], to_remove: Set[RuleKey]) -> Rules:
    """
    Pure function: return a new rules structure with (class, clause) keys filtered out.
    No side effects; you decide outside whether to set cfire.dnf.rules = new_rules.
    """
    return [
        [r for cid, r in enumerate(rules) if (cls, cid) not in to_remove]
        for cls, rules in enumerate(rule_tree)
    ]

def threshold_pruning(cfire, data, targets, theta=0):
    og_metrics = compute_rule_metrics(cfire, data)
    og_accuracy = accuracy_score(cfire.dnf(data), targets)
    if theta == 0: # safe pruning
        decision = decide_by_wins(og_metrics, win_threshold=0)
        safe_rules = prune_rules(cfire.dnf, decision.to_remove)
        safe_dnf = DNFClassifier(safe_rules, "accuracy")
        safe_dnf.compute_rule_performance(data, targets)
        safe_acc = accuracy_score(safe_dnf(data), targets)

        assert safe_acc == og_accuracy  # safe_pruning should not change output behavior

        cfire_safe_pruned = CFIRE(localexplainer_fn=None, explanations=None, inference_fn=None)
        cfire_safe_pruned.dnf = deepcopy(safe_dnf)
        return cfire_safe_pruned, safe_acc, 0

    else:

        assert 0 < theta < 1
        pruned_dnf = deepcopy(cfire.dnf)
        prune_threshold = 0
        prev_pruned_acc = og_accuracy
        for win_threshold in range(0, cfire.dnf.n_rules):
            decision = decide_by_wins(og_metrics, win_threshold=win_threshold)
            pruned_candidate_rules = prune_rules(cfire.dnf.rules, decision.to_remove)
            pruned_candidate_dnf = DNFClassifier(pruned_candidate_rules, "accuracy")
            pruned_candidate_dnf.compute_rule_performance(data, targets)
            pruned_acc = accuracy_score(pruned_candidate_dnf(data), targets)
            if 0 < og_accuracy and pruned_acc < (1-theta)*og_accuracy: # 5% allowance on original performance
                break
            else:
                pruned_dnf = deepcopy(pruned_candidate_dnf)
                prev_pruned_acc = pruned_acc
                prune_threshold = win_threshold
        cfire_threshold_pruned = CFIRE(localexplainer_fn=None, explanations=None, inference_fn=None)
        cfire_threshold_pruned.dnf = deepcopy(pruned_dnf)
        return cfire_threshold_pruned, prev_pruned_acc, prune_threshold