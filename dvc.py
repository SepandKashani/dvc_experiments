import numpy as np
import opt_einsum as oe


def computeDICoperators(
    im1: np.ndarray,  # f32(Nz, Ny, Nx)
    im2: np.ndarray,  # f32(Nz, Ny, Nx)
    im2gz: np.ndarray,  # f32(Nz, Ny, Nx)
    im2gy: np.ndarray,  # f32(Nz, Ny, Nx)
    im2gx: np.ndarray,  # f32(Nz, Ny, Nx)
    M: np.ndarray,  # f64(12, 12)
    A: np.ndarray,  # f64(12,)
):
    # Different implementation of `spambind.DIC.DICToolkit.computeDICoperators()`.

    im2g = np.stack([im2gz, im2gy, im2gx], axis=0)  # (3, Nz, Ny, Nx)
    _M, _A = _computeDICoperators(im1, im2, im2g)
    M[:] = _M.reshape(12, 12)
    A[:] = _A.reshape(12)


def _computeDICoperators(
    im1: np.ndarray,  # f32(Nz, Ny, Nx)
    im2: np.ndarray,  # f32(Nz, Ny, Nx)
    im2g: np.ndarray,  # f32(3, Nz, Ny, Nx)
) -> tuple[np.ndarray, np.ndarray]:
    # Low-level backend of computeDICoperators().
    #
    # Returns
    # -------
    # M: ndarray[f64]
    #     (3, 4, 3, 4)
    # A: ndarray[f64]
    #     (3, 4)

    assert im1.ndim == 3
    assert im1.shape == im2.shape
    assert im2g.shape == (3, *im1.shape)
    assert all(_.dtype is np.dtype(np.single) for _ in (im1, im2, im2g))

    sh = im1.shape  # (Nz, Ny, Nx)
    x = [None] * 3
    for i in range(3):
        N = sh[i]
        _x = np.ones((4, N))
        _x[i] = np.arange(N) - 0.5 * (N - 1)
        x[i] = _x

    oe_kwargs = dict(
        use_blas=True,
        optimize="auto",
    )

    nan_mask = np.isnan(im1) | np.isnan(im2)
    A = oe.contract(
        "qrs,qrs,iqrs,jq,jr,js->ij",
        ~nan_mask,
        np.nan_to_num(im1 - im2),
        im2g,
        *x,
        **oe_kwargs,
    )
    M = oe.contract(
        "qrs,iqrs,jq,jr,js,kqrs,lq,lr,ls->ijkl",
        ~nan_mask,
        im2g,
        *x,
        im2g,
        *x,
        **oe_kwargs,
    )

    return M, A
