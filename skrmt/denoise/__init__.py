"""
The :mod:`skrmt.denoise` module provides RMT-based signal denoising estimators.
"""

from .mp_pca import MarchenkoPasturPCADenoiser

__all__ = [
    "MarchenkoPasturPCADenoiser",
]
