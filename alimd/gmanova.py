import logging
import numpy as np

from alimd import utils as u

logger = logging.getLogger(__name__)

def mle(Y, X, Z):
    logger.info("Maximum Likehood Estimation.")
    n, p = Y.shape
    XTXinv = np.linalg.inv(X.T @ X)
    S = Y.T @ Y - np.linalg.multi_dot([Y.T, X, XTXinv, X.T, Y])
    Sinv = np.linalg.inv(S)
    left = XTXinv @ X.T
    right = Z.T @ np.linalg.inv(np.linalg.multi_dot([Z, Sinv, Z.T]))
    B = np.linalg.multi_dot([left, Y, Sinv, right])
    res = Y - np.linalg.multi_dot([X, B, Z])
    SGM = (res.T @ res) / n
    return B, SGM


def local_influence(Y, X, Z):
    logger.info("Local influence for GMANOVA model.")
    logger.info(f"Y shape: {Y.shape}")
    logger.info(f"X shape: {X.shape}")
    logger.info(f"Z shape: {Z.shape}")

    n, p = Y.shape
    B, SGM = mle(Y, X, Z)
    logger.info("Computing local influence matrices.")
    SGMI = np.linalg.inv(SGM)
    res = Y - np.linalg.multi_dot([X, B, Z])
    Dp = u.dup_matrix(p)
    L11 = u.kron(np.linalg.multi_dot([Z, SGMI, Z.T]), X.T @ X)
    L12 = np.zeros(shape=L11.shape)
    # L12 = u.kron(Z @ SGMI, np.linalg.multi_dot([X.T, res, SGMI])) @ Dp
    L22 = n / 2 * np.linalg.multi_dot([Dp.T, u.kron(SGMI, SGMI), Dp])
    L = - np.block([[L11, L12], [L12.T, L22]])
    # Delta1 = u.diag_dup_prod(
    #     u.kron(np.linalg.multi_dot([Z, SGMI, res.T]),X.T),
    #     premul=False
    # )
    Delta1 = u.kron_diag_dup(Z @ SGMI @ res.T, X.T)
    # Delta2 = 0.5 * Dp.T @ u.diag_dup_prod(u.kron_self(SGMI @ res.T), premul=False)
    Delta2 = 0.5 * Dp.T @ u.kron_diag_dup(SGMI @ res.T, SGMI @ res.T)
    Delta = np.block([[Delta1], [Delta2]]).astype(np.int32)
    F = 2 * np.linalg.multi_dot(
        [
            Delta.T,
            np.linalg.inv(-L).astype(np.int32),
            Delta
        ]
    )
    logger.info(f"L shape: {L.shape}")
    logger.info(f"Delta shape: {Delta.shape}")
    logger.info(f"F shape: {F.shape}")  
    return L, Delta, F