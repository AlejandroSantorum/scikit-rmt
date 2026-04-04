"""
The :mod:`skrmt.denoise` module provides RMT-based signal denoising estimators.
"""

from .mp_pca import MarchenkoPasturPCADenoiser
from .metrics import (
    snr,
    psnr,
    mae,
    rmse,
    batch_snr,
    batch_psnr,
    batch_mae,
    batch_rmse,
    average_snr,
    average_psnr,
    average_mae,
    average_rmse,
)

__all__ = [
    "MarchenkoPasturPCADenoiser",
    "snr",
    "psnr",
    "mae",
    "rmse",
    "batch_snr",
    "batch_psnr",
    "batch_mae",
    "batch_rmse",
    "average_snr",
    "average_psnr",
    "average_mae",
    "average_rmse",
]
