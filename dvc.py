import numpy as np
import opt_einsum as oe

MA_Contract = tuple[
    "oe.contract.ContractExpression",
    "oe.contract.ContractExpression",
]


def computeDICoperators(
    im1: np.ndarray,  # f32(Nz, Ny, Nx)
    im2: np.ndarray,  # f32(Nz, Ny, Nx)
    im2gz: np.ndarray,  # f32(Nz, Ny, Nx)
    im2gy: np.ndarray,  # f32(Nz, Ny, Nx)
    im2gx: np.ndarray,  # f32(Nz, Ny, Nx)
    M: np.ndarray,  # f32(12, 12)
    A: np.ndarray,  # f32(12,)
):
    # Different implementation of `spambind.DIC.DICToolkit.computeDICoperators()`.

    im2g = np.stack([im2gz, im2gy, im2gx], axis=0)  # (3, Nz, Ny, Nx)
    _M, _A, _ = _computeDICoperators(im1, im2, im2g)
    M[:] = _M.reshape(12, 12)
    A[:] = _A.reshape(12)


def _computeDICoperators(
    im1: np.ndarray,  # f32(Nz, Ny, Nx)
    im2: np.ndarray,  # f32(Nz, Ny, Nx)
    im2g: np.ndarray,  # f32(3, Nz, Ny, Nx)
    expr: MA_Contract = None,  # (M_expr, A_expr)
) -> tuple[np.ndarray, np.ndarray, MA_Contract]:
    # Low-level backend of computeDICoperators().
    #
    # Returns
    # -------
    # M: ndarray[f32]
    #     (3, 4, 3, 4)
    # A: ndarray[f32]
    #     (3, 4)
    # (M_expr, A_expr): MA_Contract
    #     Good contraction rules to eval (M, A).
    #     Re-use them between calls to avoid the cost of finding good contractions each time.

    assert im1.ndim == 3
    assert im1.shape == im2.shape
    assert im2g.shape == (3, *im1.shape)
    assert all(_.dtype is np.dtype(np.single) for _ in (im1, im2, im2g))

    sh = (Nz, Ny, Nx) = im1.shape  # (Nz, Ny, Nx)
    x = [None] * 3
    for i in range(3):
        N = sh[i]
        _x = np.ones((4, N), dtype=np.single)
        _x[i] = np.arange(N) - 0.5 * (N - 1)
        x[i] = _x

    if expr:
        M_expr, A_expr = expr
    else:
        oe_kwargs = dict(
            use_blas=True,
            optimize="greedy",  # seems to work best during my tests
        )

        M_expr = oe.contract_expression(
            "qrs,iqrs,jq,jr,js,kqrs,lq,lr,ls->ijkl",
            sh,
            (3, *sh),
            (4, Nz),
            (4, Ny),
            (4, Nx),
            (3, *sh),
            (4, Nz),
            (4, Ny),
            (4, Nx),
            **oe_kwargs,
        )
        A_expr = oe.contract_expression(
            "qrs,iqrs,jq,jr,js->ij",
            sh,
            (3, *sh),
            (4, Nz),
            (4, Ny),
            (4, Nx),
            **oe_kwargs,
        )
        expr = (M_expr, A_expr)

    nan_mask = np.isnan(im1) | np.isnan(im2)
    diff = np.where(nan_mask, 0, im1 - im2)
    M = M_expr(~nan_mask, im2g, *x, im2g, *x)
    A = A_expr(diff, im2g, *x)

    assert all(_.dtype is np.dtype(np.single) for _ in (M, A))
    return M, A, (M_expr, A_expr)
