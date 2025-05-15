import importlib

import numpy as np
import numpy.random as npr

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

# dvc solution (drop-in spam replacement) -----------------
M_dvc = np.zeros((12, 12), dtype=np.double)
A_dvc = np.zeros((12,), dtype=np.double)
dvc.computeDICoperators(im1, im2, *im2_grad, M_dvc, A_dvc)

assert np.allclose(M_spam, M_dvc)
assert np.allclose(A_spam, A_dvc)
# ---------------------------------------------------------

# dvc solution 2 ------------------------------------------
# A potentially faster solution, assuming we can change API of computeDICoperators()
im2_grad2 = np.stack(im2_grad, axis=0)  # (D, *sh)
M_dvc2, A_dvc2, MA_expr = dvc._computeDICoperators(im1, im2, im2_grad2)
M_dvc2 = M_dvc2.reshape(12, 12)
A_dvc2 = A_dvc2.reshape(12)

assert np.allclose(M_spam, M_dvc2)
assert np.allclose(A_spam, A_dvc2)
# ---------------------------------------------------------
