"""
Substrate-agnostic consciousness theory calculators.

Each theory takes a Substrate and computes a score from its activity
matrix, node groups, adjacency, and (for PPT/FEP) a PredictionAdapter.

These calculators read only the Substrate interface — no substrate-
type-specific code — making the same computation valid on transformer
activations, fMRI BOLD parcels, or any other dynamical system that
exposes an (T, N) activity matrix.
"""

from .iit import compute_iit_phi, IITPhiResult
from .gwt import compute_gwt, GWTResult
from .hot import compute_hot, HOTResult
from .ast import compute_ast, ASTResult
from .ppt import compute_ppt, PPTResult
from .fep import compute_fep, FEPResult
from .qit import compute_qit, QITResult

__all__ = [
    "compute_iit_phi", "IITPhiResult",
    "compute_gwt", "GWTResult",
    "compute_hot", "HOTResult",
    "compute_ast", "ASTResult",
    "compute_ppt", "PPTResult",
    "compute_fep", "FEPResult",
    "compute_qit", "QITResult",
]
