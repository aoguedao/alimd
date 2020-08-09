import numpy as np

from numba import jit, prange


@jit(nopython=True)
def diag_dup_prod(A, premul=True):
    n2, m2 = A.shape
    if premul:
        n = int(np.sqrt(n2))
        return A[::n+1, :]
    else:
        m = int(np.sqrt(m2))
        return A[:, ::m+1]


@jit(nopython=True)
def vec(A):
    return A.T.ravel().reshape(-1, 1)


@jit(nopython=True)
def vech(A):
    m = np.tril(A).T.ravel()
    return m[np.where(m != 0)].reshape(-1, 1)


@jit(nopython=True)
def unit_vector(n, i, j, p=None):
    if p is None:
        p = int(0.5) * n * (n + 1)
    u = np.zeros(p)
    i += 1
    j += 1
    u_idx = int((j - 1) * n + i - 0.5 * j * (j - 1)) - 1
    u[u_idx] = 1
    return u.reshape(p, 1)


@jit(nopython=True)
def tij_matrix(n, i, j):
    T = np.zeros(shape=(n, n))
    T[i, j] = 1
    T[j, i] = 1
    return T


@jit(nopython=True, parallel=True)
def dup_matrix(n):
    p = int(n * (n + 1) / 2)
    nsq = n ** 2
    DT = np.zeros(shape=(p, nsq))
    for i in prange(n):
        for j in prange(i + 1):
            u = unit_vector(n, i, j, p)
            T = tij_matrix(n, i, j)
            DT += u @ vec(T).T
    return DT.T


def kronecker(A, B):
    m, n = A.shape
    p, q = B.shape
    return (A[:, None, :, None] * B[None, :, None, :]).reshape(m * p, n * q)


def self_kronecker(A):
    return kronecker(A, A)