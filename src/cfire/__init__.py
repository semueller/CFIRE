"""CFIRE: combine local explanations into a global rule model of a black box."""
from .core import CFIRE
from .rules import DNFClassifier
from .binarise import binarise_threshold

__all__ = ["CFIRE", "DNFClassifier", "binarise_threshold"]
