"""fastldsc: fast stratified LD score regression for scanning many annotations.

An independent re-implementation of the partitioned-heritability path of
ldsc (Bulik-Sullivan et al. 2015; Finucane et al. 2015) with the I/O
amortised across models, one-pass multi-annotation LD scores, and a
bordered-solve screen for thousands of candidate annotations.
"""

__version__ = "0.1.0b1"

from .sldsc import RefData, TraitData, fit_h2, run_model, hsq_weights  # noqa: F401
from .screen import screen_candidates  # noqa: F401
