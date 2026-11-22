"""Dependency-free Haar wavelet transform and FISTA solver for the MRI
compressed-sensing section of the lab.

Deliberately implemented with plain numpy instead of an external
convex-optimization library (e.g. PyLops) or wavelet library (e.g.
PyWavelets), since neither was part of the course's approved package list.
The Haar transform is exactly orthogonal (it's built from +-1/sqrt(2)
butterflies), which keeps the FISTA math simple: the adjoint of the inverse
transform is just the forward transform, and the same is true of an
orthonormal ("ortho"-normalized) 2-D FFT.
"""
import numpy as np


def _haar_step_rows(a):
    L = (a[:, 0::2] + a[:, 1::2]) / np.sqrt(2)
    H = (a[:, 0::2] - a[:, 1::2]) / np.sqrt(2)
    return np.concatenate([L, H], axis=1)


def _haar_step_cols(a):
    L = (a[0::2, :] + a[1::2, :]) / np.sqrt(2)
    H = (a[0::2, :] - a[1::2, :]) / np.sqrt(2)
    return np.concatenate([L, H], axis=0)


def _haar_inv_step_cols(a):
    n = a.shape[0]
    L, H = a[:n // 2, :], a[n // 2:, :]
    out = np.zeros_like(a)
    out[0::2, :] = (L + H) / np.sqrt(2)
    out[1::2, :] = (L - H) / np.sqrt(2)
    return out


def _haar_inv_step_rows(a):
    n = a.shape[1]
    L, H = a[:, :n // 2], a[:, n // 2:]
    out = np.zeros_like(a)
    out[:, 0::2] = (L + H) / np.sqrt(2)
    out[:, 1::2] = (L - H) / np.sqrt(2)
    return out


def haar2_forward(img, level=3):
    """Multilevel 2-D Haar transform. Returns a coefficient array the same
    shape as `img`: the coarsest approximation sits in the top-left corner,
    surrounded by detail bands, exactly like a typical wavelet-coefficient
    visualization (this is the same packing `pywt.coeffs_to_array` produces)."""
    coeffs = np.array(img, dtype=complex if np.iscomplexobj(img) else float)
    h, w = coeffs.shape
    for _ in range(level):
        sub = _haar_step_cols(_haar_step_rows(coeffs[:h, :w]))
        coeffs[:h, :w] = sub
        h, w = h // 2, w // 2
    return coeffs


def haar2_inverse(coeffs, level=3):
    """Inverse of `haar2_forward`. Because the Haar transform is orthogonal,
    this is also its adjoint (transpose-conjugate)."""
    out = np.array(coeffs)
    h = out.shape[0] // (2 ** (level - 1))
    w = out.shape[1] // (2 ** (level - 1))
    for _ in range(level):
        sub = _haar_inv_step_rows(_haar_inv_step_cols(out[:h, :w]))
        out[:h, :w] = sub
        h, w = h * 2, w * 2
    return out


def soft_threshold(x, thresh):
    """Complex soft-thresholding (shrinkage): shrinks each coefficient's
    magnitude toward zero by `thresh`, keeping its phase."""
    mag = np.abs(x)
    scale = np.maximum(mag - thresh, 0) / (mag + 1e-12)
    return x * scale


def bpd(b, A, AH, shape, niter=60, thresh=0.01):
    """Solve Basis Pursuit Denoising with FISTA (accelerated ISTA),
    implemented directly with numpy. With a fixed threshold, FISTA solves the
    penalized form of the problem,

        minimize 1/2 ||Ax - b||_2^2 + thresh * ||x||_1,

    which plays the same role as "minimize ||x||_1 subject to Ax ~ b": `thresh`
    sets how much data consistency is traded for sparsity. `thresh` controls how aggressively
    small wavelet coefficients are shrunk toward zero each iteration
    (larger thresh -> sparser, more aggressively denoised x); `niter`
    controls how many gradient + shrinkage steps are taken."""
    x = np.zeros(shape, dtype=complex)
    y = x.copy()
    t = 1.0
    for _ in range(niter):
        grad = AH(A(y) - b)
        x_new = soft_threshold(y - grad, thresh)
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        y = x_new + ((t - 1) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return x
