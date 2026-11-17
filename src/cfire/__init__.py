"""CFIRE: combine local explanations into a global rule model of a black box."""
from .core import CFIRE
from .rules import DNFClassifier
from .binarise import binarise_threshold, binarise_topk, binarise_topk_abs
from .selection import CFIRESelect, covered_accuracy, accuracy, coverage
from .pruning import PruneResult, prune
from .encoding import Encoding
from .printing import format_rule, format_rules

__all__ = ["CFIRE", "CFIRESelect", "DNFClassifier", "binarise_threshold", "binarise_topk", "binarise_topk_abs", "covered_accuracy", "accuracy", "coverage", "prune", "PruneResult", "Encoding", "format_rule", "format_rules"]
