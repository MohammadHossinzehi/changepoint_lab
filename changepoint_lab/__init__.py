"""changepoint_lab: offline and online changepoint detection in NumPy."""

from .costs import Cost, L2Cost, NormalMeanVarCost, PoissonCost, make_cost, mad_sigma
from .metrics import annotation_error, hausdorff, precision_recall_f1, rand_index, score
from .offline import (CropsResult, Segmentation, binary_segmentation, brute_force, crops,
                      optimal_partitioning, pelt, segment_neighbourhood)
from .online import BOCPD, BOCPDResult, NormalGamma, PoissonGamma, bocpd, constant_hazard
from .penalties import resolve_penalty
from .synth import piecewise

__version__ = "1.0.0"

__all__ = [
    "Cost", "L2Cost", "NormalMeanVarCost", "PoissonCost", "make_cost", "mad_sigma",
    "Segmentation", "CropsResult", "optimal_partitioning", "pelt", "segment_neighbourhood",
    "binary_segmentation", "crops", "brute_force",
    "BOCPD", "BOCPDResult", "NormalGamma", "PoissonGamma", "bocpd", "constant_hazard",
    "hausdorff", "precision_recall_f1", "rand_index", "annotation_error", "score",
    "resolve_penalty", "piecewise",
]
