import numpy as np
from scipy.sparse.linalg import svds
from numba import njit
try:
    from sklearn.utils.extmath import randomized_svd
except ImportError:
    randomized_svd = None


def log_status(msg, verbose=True):
    if verbose:
        print(msg)


def valid_mlrank(mlrank):
    prod = np.prod(mlrank)
    return all(prod//r > r for r in mlrank)


def mode_selector(modes):
    mode, = set.difference({0, 1, 2}, modes)
    return mode


def core_growth_callback(growth_tol, verbose=False):
    def check_core_growth(step, core, factors):
        singular_values, _ = core
        core_norm = np.linalg.norm(singular_values)
        g_growth = (core_norm - check_core_growth.core_norm) / core_norm
        check_core_growth.core_norm = core_norm
        log_status(f'Step {step} growth of the core: {g_growth}', verbose=verbose)
        if g_growth < growth_tol:
            log_status(f'Core is no longer growing. Norm of the core: {core_norm}.', verbose=verbose)
            raise StopIteration
    check_core_growth.core_norm = 0
    return check_core_growth


def initialize_factors(shape, mlrank, iter_modes, update_order, seed, dtype=None):
    factors = [None]*3
    update_modes = iter_modes[update_order[0]]
    random_state = np.random if seed is None else np.random.RandomState(seed)
    for mode in update_modes:
        factor_shape = shape[mode], mlrank[mode]
        if mode == update_modes[0]:
            factors[mode] = np.empty(factor_shape, dtype=dtype)
        else:
            factors[mode] = columnwise_orthonormal(factor_shape, random_state, dtype=dtype)
    return tuple(factors)


def columnwise_orthonormal(dims, random_state=None, dtype=None):
    if random_state is None:
        random_state = np.random
    u = random_state.rand(*dims)
    q, _ = np.linalg.qr(u, mode='reduced')
    return np.asarray(q, dtype=dtype)


def sparse_hooi(
    idx, val, shape, mlrank,
    max_iters=15, update_order=None, iter_callback=None,
    growth_tol=0.001, materialize_core=False,
    seed=None, dtype=None, verbose=True
):
    '''
    Computes HOOI decomposition of a sparse tensor provided in COO format.
    Usage:
    u0, u1, u2, g = sparse_hooi(idx, val, shape, mlrank)
    '''
    assert valid_mlrank(mlrank)
    if update_order is None:
        update_order = (2, 1, 0)
    iter_modes = [(0, 2, 1), (1, 2, 0), (2, 1, 0)]

    factors = initialize_factors(shape, mlrank, iter_modes, update_order, seed, dtype=dtype)

    if iter_callback is None:
        iter_callback = core_growth_callback(growth_tol, verbose=verbose)
    iter_callback.stop_reason = 'Exceeded max iterations limit.'
    
    for i in range(max_iters):
        for step, order in enumerate(update_order, start=1):
            mode, *mul_modes = iter_modes[order]
            factors[mode][:], core_factors = sdttm_leading_factors(
                idx, val, shape, mlrank, factors, mode, mul_modes,
                left_only=step<len(update_order), dtype=dtype
            )
        try:
            iter_callback(i, core_factors, factors)
        except StopIteration:
            iter_callback.stop_reason = 'Stopping criteria met.'
            break
    
    g = core_factors
    if materialize_core:
        g = compute_core(g, iter_modes[update_order[-1]], mlrank)
    u0, u1, u2 = factors
    return u0, u1, u2, g


def sdttm_leading_factors(idx, val, shape, mlrank, factors, main_mode, mul_modes, left_only=True, dtype=None):
    u_mode, v_mode = mul_modes
    u, v = factors[u_mode], factors[v_mode]
    compressed = sparse_double_ttm(
        idx, val, shape, u, v,
        main_mode, ((u_mode, 0), (v_mode, 0)),
        dtype=dtype
    )
    return_vectors = 'u' if left_only else True
    matrisized = compressed.reshape(shape[main_mode], -1)
    left, *rest = svds(
        matrisized, k=mlrank[main_mode], return_singular_vectors=return_vectors
    )
    left_leading = left[:, ::-1]
    rest_leading = tuple(f[::-1] if f is not None else None for f in rest)
    return left_leading, rest_leading


def sparse_double_ttm(idx, val, shape, U, V, main_mode, ttm_modes, dtype=None):
    res_mode1, mul_mode1 = ttm_modes[0]
    res_mode2, mul_mode2 = ttm_modes[1]
    res_shape = (shape[main_mode], U.shape[1-mul_mode1], V.shape[1-mul_mode2])

    u = U.T if mul_mode1 == 1 else U
    v = V.T if mul_mode2 == 1 else V

    res = np.zeros(res_shape, dtype=dtype)
    sdttm(idx, val, u, v, main_mode, res_mode1, res_mode2, res)
    return res


@njit(nogil=True)
def sdttm(idx, val, u, v, mode0, mode1, mode2, res):
    _, new_shape1 = u.shape
    _, new_shape2 = v.shape
    for i in range(len(val)):
        i0 = idx[i, mode0]
        i1 = idx[i, mode1]
        i2 = idx[i, mode2]
        res_i = res[i0, :, :]
        tmp_res = np.zeros_like(res_i)
        for j in range(new_shape1):
            uij = u[i1, j]
            for k in range(new_shape2):
                vik = v[i2, k]
                tmp_res[j, k] += uij * vik
        res_i += val[i] * tmp_res


def compute_core(core_factors, core_shape_modes, mlrank):
    sv, vt = core_factors
    core = np.ascontiguousarray(
        (sv[:, np.newaxis] * vt)
        .reshape([mlrank[mode] for mode in core_shape_modes])
        .transpose(*core_shape_modes)
    )
    return core