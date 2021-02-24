import logging
import numpy as np

from scipy import linalg

from alimd import utils as u

logger = logging.getLogger(__name__)

def mle(Y, X, Z):
    logger.info("Maximum Likehood Estimation.")
    n, p = Y.shape
    B = mle_B(Y, X, Z)
    res = Y - np.linalg.multi_dot([X, B, Z])
    SGM = (res.T @ res) / n
    return B, SGM


def mle_B(Y, X, Z):
    xtx = X.T @ X
    xtxixty = linalg.solve(xtx, X.T @ Y, assume_a="sym")
    S = Y.T @ ( Y - X @ xtxixty)  # descomponer S para obtener resultado anterior
    sizt = linalg.solve(S, Z.T, assume_a="sym")
    siztzsizti = linalg.solve(Z @ sizt, sizt.T , assume_a="sym").T
    B = xtxixty @ siztzsizti
    return B


def local_influence(Y, X, Z):
    logger.info("Local influence for GMANOVA model.")
    logger.info(f"Y shape: {Y.shape}")
    logger.info(f"X shape: {X.shape}")
    logger.info(f"Z shape: {Z.shape}")

    n, p = Y.shape
    B, SGM = mle(Y, X, Z)
    logger.info("Computing local influence matrices.")
    # SGMI = np.linalg.inv(SGM)
    res = Y - np.linalg.multi_dot([X, B, Z])
    sgmizt = linalg.solve(SGM, Z.T, assume_a="pos")
    sgmirest = linalg.solve(SGM, res.T, assume_a="pos")
    Dp = u.dup_matrix(p)
    xtx = X.T @ X

    L11 = np.kron(Z @ sgmizt, xtx)
    L12 = Dp.T @ u.kron(sgmizt, sgmirest @ X)
    L22 = n / 2 * Dp.T @ linalg.solve(u.kron(SGM, SGM), Dp, assume_a="sym")
    L = - np.block([[L11, L12], [L12.T, L22]])

    Delta1 = u.kron_diag_dup(Z @ sgmirest, X.T)
    Delta2 = 0.5 * Dp.T @ u.kron_diag_dup(sgmirest, sgmirest)
    Delta = np.block([[Delta1], [Delta2]])

    F = - 2 * Delta.T @ linalg.solve(L, Delta, assume_a="sym")

    logger.info(f"L shape: {L.shape}")
    logger.info(f"Delta shape: {Delta.shape}")
    logger.info(f"F shape: {F.shape}")

    return L, Delta, F