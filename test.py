import importlib

import cupy as cp
import numpy as np
import numpy.random as npr
from cupyx.profiler import benchmark

import dvc

importlib.reload(dvc)

from spambind.DIC.DICToolkit import computeDICoperators as spam_computeDICoperators

rng = npr.default_rng(0)


def mask_at_random(x: np.ndarray, r: float, rng: npr.Generator) -> np.ndarray:
    """
    Insert NaNs at random locations in `x`.

    Parameters
    ----------
    x: ndarray
        Array to NaN-ify.
    r: [0, 1]
        Ratio of NaNs to insert.
    rng: Generator

    Returns
    -------
    y: ndarray
        `x` containing a ratio of `r` NaNs.
    """
    assert 0 <= r <= 1
    idx = rng.integers(0, x.size, size=int(r * x.size))

    y = x.flatten()
    y[idx] = np.nan
    y = y.reshape(x.shape)
    return y


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    """
    Returns
    -------
    r: float
        norm2(a - b) / norm2(b)
    """
    assert a.shape == b.shape

    r = np.sqrt(np.sum((a - b) ** 2) / np.sum(b**2))
    return r


allclose = lambda a, b: rel_error(a, b) <= 1e-6

# generate some sample 3D data ----------------------------
D = 3
sh = rng.integers(300, 310, size=D)  # ZYX order

im1 = rng.uniform(0, 1, size=sh).astype(np.single)
im2 = rng.uniform(0, 1, size=sh).astype(np.single)
im2_grad = np.gradient(im2)  # [grad_Z(*sh), grad_Y(*sh), grad_X(*sh)]

nan_rate = 5e-2
im1 = mask_at_random(im1, nan_rate, rng)
im2 = mask_at_random(im2, nan_rate, rng)
# ---------------------------------------------------------

# spam solution -------------------------------------------
M_spam = np.zeros((12, 12), dtype=np.double)
A_spam = np.zeros((12,), dtype=np.double)
spam_computeDICoperators(im1, im2, *im2_grad, M_spam, A_spam)
# ---------------------------------------------------------

# dvc solution (CPU; drop-in spam replacement) ------------
M_dvc = np.zeros((12, 12), dtype=np.double)
A_dvc = np.zeros((12,), dtype=np.double)
dvc.computeDICoperators(im1, im2, *im2_grad, M_dvc, A_dvc)

assert allclose(M_spam, M_dvc)
assert allclose(A_spam, A_dvc)
# ---------------------------------------------------------

# dvc solution 2 (CPU) ------------------------------------
# A faster solution, assuming we can change API of computeDICoperators()
im2_grad2 = np.stack(im2_grad, axis=0)  # (D, *sh)
M_dvc2, A_dvc2, MA_expr = dvc._computeDICoperators(im1, im2, im2_grad2)
M_dvc2 = M_dvc2.reshape(12, 12)
A_dvc2 = A_dvc2.reshape(12)

assert allclose(M_spam, M_dvc2)
assert allclose(A_spam, A_dvc2)
# ---------------------------------------------------------


# benchmarks (CPU) ----------------------------------------
cpu_kwargs = dict(
    n_repeat=10,
    n_warmup=3,
)
t_spam = benchmark(
    spam_computeDICoperators, (im1, im2, *im2_grad, M_spam, A_spam), **cpu_kwargs
)
t_dvc = benchmark(
    dvc.computeDICoperators, (im1, im2, *im2_grad, M_dvc, A_dvc), **cpu_kwargs
)
t_dvc2 = benchmark(dvc._computeDICoperators, (im1, im2, im2_grad2), **cpu_kwargs)
print("[CPU]", t_spam.to_str(show_gpu=False))
print("[CPU]", t_dvc.to_str(show_gpu=False))
print("[CPU]", t_dvc2.to_str(show_gpu=False))
# ---------------------------------------------------------

# benchmarks (GPU) ----------------------------------------
im1 = cp.asarray(im1)
im2 = cp.asarray(im2)
im2_grad = map(cp.asarray, im2_grad)
im2_grad2 = cp.asarray(im2_grad2)
M_dvc = cp.asarray(M_dvc)
A_dvc = cp.asarray(A_dvc)

gpu_kwargs = dict(
    n_repeat=1_000,
    n_warmup=10,
)
t_dvc = benchmark(
    dvc.computeDICoperators, (im1, im2, *im2_grad, M_dvc, A_dvc), **gpu_kwargs
)
t_dvc2 = benchmark(dvc._computeDICoperators, (im1, im2, im2_grad2), **gpu_kwargs)
print("[GPU]", t_dvc.to_str(show_gpu=True))
print("[GPU]", t_dvc2.to_str(show_gpu=True))
# ---------------------------------------------------------
