import numpy as np

from numba import jit, njit, prange


@njit
def vec(A):
    # https://stackoverflow.com/questions/25248290/most-elegant-implementation-of-matlabs-vec-function-in-numpy
    return A.T.ravel().reshape(-1, 1)


@njit
def vech(A):
    m = np.tril(A).T.ravel()
    return m[np.where(m != 0)].reshape(-1, 1)


@njit(fastmath=True)
def kron(A, B=None):
    if B is not None:
        return np.kron(np.ascontiguousarray(A), np.ascontiguousarray(B))
    else:
        return np.kron(np.ascontiguousarray(A), np.ascontiguousarray(A))


@njit(fastmath=True,parallel=False)
def kron_preallocated(A, B, out):
#     #https://stackoverflow.com/questions/56067643/speeding-up-kronecker-products-numpy
    m, n = A.shape
    p, q = B.shape
    for i in prange(m):
        for j in range(p):
            for k in range(n):
                for l in range(q):
                    out[i, j, k, l] = A[i, k] * B[j, l]
    return out


# @njit(fastmath=True)
# def kron_vector(a, b):
#     return np.outer(a, b).reshape(-1, 1)


@njit
def unit_vector(n, i, j, p=None):
    if p is None:
        p = int(0.5 * n * (n + 1))
    u = np.zeros(p)
    i += 1
    j += 1
    u_idx = int((j - 1) * n + i - 0.5 * j * (j - 1)) - 1
    u[u_idx] = 1
    return u.reshape(p, 1)


@njit
def tij_matrix(n, i, j):
    T = np.zeros(shape=(n, n))
    T[i, j] = 1
    T[j, i] = 1
    return T


@njit(parallel=True)
def dup_matrix(n):
    # TODO: Deprecated
    p = int(n * (n + 1) / 2)
    nsq = n ** 2
    DT = np.zeros(shape=(p, nsq))
    for i in prange(n):
        for j in range(i + 1):
            u = unit_vector(n, i, j, p)
            T = tij_matrix(n, i, j)
            DT += u @ vec(T).T
    return DT.T


# @njit
# def diag_dup_prod(A, premul=True):
#     n2, m2 = A.shape
#     if premul:
#         n = int(np.sqrt(n2))
#         return A[::n+1, :]
#     else:
#         m = int(np.sqrt(m2))
#         return A[:, ::m+1]

@njit(parallel=True)
def dupl_cols(n):
    k = 0
    col = np.empty(n ** 2, dtype=np.int32)
    for i in prange(n):
        for j in range(i, n):
            col[i + j * n] = col[j + i * n] = k
            k += 1
    return col


@njit
def duplication(n):
    # TODO: Validar que está funcionando bien!
    col = dupl_cols(n)
    dn = np.zeros((n ** 2, int(n * (n + 1) / 2)), dtype=np.int32)
    for i in range(n ** 2):
        dn[i, col[i]] = 1
    return dn


@njit(fastmath=True, parallel=True)
def kron_diag_dup(A, B):
    assert A.shape[1] == B.shape[1]
    q, n = A.shape
    m, _ = B.shape
    C = np.empty(shape=(m * q, n))
    for i in prange(n):
        C[:, i] = np.outer(A[:, i], B[:, i]).flatten()
    return C