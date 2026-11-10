"""CFIRE: combine local explanations into a global rule model of a black box."""
from .core import CFIRE
from .rules import DNFClassifier
from .binarise import binarise_threshold
from .selection import CFIRESelect, covered_accuracy, accuracy, coverage

__all__ = ["CFIRE", "CFIRESelect", "DNFClassifier", "binarise_threshold", "covered_accuracy", "accuracy", "coverage"]
