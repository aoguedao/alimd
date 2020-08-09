import logging
import numpy as np

import utils as u


def mle(Y, X, Z):
    logging.info("Maximum Likehood Estimation.")
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
    logging.info("Local influence for GMANOVA model.")
    logging.info(f"Y shape: {Y.shape}")
    logging.info(f"X shape: {X.shape}")
    logging.info(f"Z shape: {Z.shape}")

    n, p = Y.shape
    B, SGM = mle(Y, X, Z)
    logging.info("Computing local influence matrices.")
    SGMI = np.linalg.inv(SGM)
    res = Y - np.linalg.multi_dot([X, B, Z])
    Dp = u.dup_matrix(p)
    L11 = u.kronecker(np.linalg.multi_dot([Z, SGMI, Z.T]), X.T @ X)
    L12 = np.zeros(shape=L11.shape)
    # L12 = u.kronecker(Z @ SGMI, np.linalg.multi_dot([X.T, res, SGMI])) @ Dp
    L22 = n / 2 * np.linalg.multi_dot([Dp.T, u.kronecker(SGMI, SGMI), Dp])
    L = - np.block([[L11, L12], [L12.T, L22]])
    Delta1 = u.diag_dup_prod(
        u.kronecker(np.linalg.multi_dot([Z, SGMI, res.T]),X.T),
        premul=False
    )
    Delta2 = 0.5 * Dp.T @ u.diag_dup_prod(u.self_kronecker(SGMI @ res.T), premul=False)
    Delta = np.block([[Delta1], [Delta2]])
    F = 2 * np.linalg.multi_dot([Delta.T, np.linalg.inv(-L), Delta])
    logging.info(f"L shape: {L.shape}")
    logging.info(f"Delta shape: {Delta.shape}")
    logging.info(f"F shape: {F.shape}")  
    return L, Delta, F