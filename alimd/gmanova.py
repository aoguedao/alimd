import logging
import numpy as np

from scipy import linalg

from alimd import utils as u

logger = logging.getLogger(__name__)

def mle(Y, X, Z, cov_structure):
    logger.info("Maximum Likehood Estimation.")
    if cov_structure == "uc":
        B, SIGMA = mle_uc(Y, X, Z)
        return B, SIGMA
    elif cov_structure == "scs":
        B, GAMMA, PHI, G, SIGMA = mle_scs(Y, X, Z)
        return B, GAMMA, PHI, G, SIGMA
    else:
        raise NotImplementedError


def mle_uc(Y, X, Z):
    n, _ = Y.shape
    # -- B
    xtx = X.T @ X
    xtxixty = linalg.solve(xtx, X.T @ Y, assume_a="sym")
    S = Y.T @ ( Y - X @ xtxixty)  # descomponer S para obtener resultado anterior
    sizt = linalg.solve(S, Z.T, assume_a="sym")
    siztzsizti = linalg.solve(Z @ sizt, sizt.T , assume_a="sym").T
    B = xtxixty @ siztzsizti
    # Sigma
    res = Y - np.linalg.multi_dot([X, B, Z])
    SIGMA = (res.T @ res) / n
    return B, SIGMA


def mle_scs(Y, X, Z):
    n, m = X.shape
    q, p = Z.shape
    xtx = X.T @ X
    xtxixty = linalg.solve(xtx, X.T @ Y, assume_a="sym")
    # zzt = Z @ Z.T
    zztiz = linalg.solve(Z @ Z.T, Z, assume_a="sym")
    # -- B
    B = xtxixty @ zztiz.T
    # Gamma
    S = Y.T @ ( Y - X @ xtxixty)
    GAMMA = zztiz @ S @ zztiz.T / n
    # Phi
    G = linalg.solve(Z, np.zeros(shape=(p - m, q))).T  # TODO
    ggtigyt = linalg.solve(G @ G.T, G @ Y.T, assume_a="sym")
    PHI = ggtigyt @ ggtigyt.T / n
    # SIGMA
    iqztzztizyt =  (np.identity(q) - Z.T @ zztiz) @ Y.T
    SIGMA = Z.T @ GAMMA @ Z @ iqztzztizyt @ iqztzztizyt.T / n
    return B, GAMMA, PHI, G, SIGMA


def local_influence(Y, X, Z, cov_structure):
    logger.info("Local influence for GMANOVA model.")
    logger.info(f"Y shape: {Y.shape}")
    logger.info(f"X shape: {X.shape}")
    logger.info(f"Z shape: {Z.shape}")

    logger.info("Computing local influence matrices.")
    if cov_structure == "uc":
        L, Delta, F = local_influence_uc(Y, X, Z) 
    elif cov_structure == "scs":
        L, Delta, F = local_influence_scs(Y, X, Z) 
    else:
        raise NotImplementedError

    logger.info(f"L shape: {L.shape}")
    logger.info(f"Delta shape: {Delta.shape}")
    logger.info(f"F shape: {F.shape}")

    return L, Delta, F


def local_influence_uc(Y, X, Z):
    n, p = Y.shape
    B, SIGMA = mle_uc(Y, X, Z)
    res = Y - np.linalg.multi_dot([X, B, Z])
    sigmaizt = linalg.solve(SIGMA, Z.T, assume_a="pos")
    sigmairest = linalg.solve(SIGMA, res.T, assume_a="pos")
    Dp = u.dup_matrix(p)
    xtx = X.T @ X

    L11 = np.kron(Z @ sigmaizt, xtx)
    L12 = Dp.T @ u.kron(sigmaizt, sigmairest @ X)
    L22 = n / 2 * Dp.T @ linalg.solve(u.kron(SIGMA, SIGMA), Dp, assume_a="sym")
    L = - np.block([[L11, L12], [L12.T, L22]])

    Delta1 = u.kron_diag_dup(Z @ sigmairest, X.T)
    Delta2 = 0.5 * Dp.T @ u.kron_diag_dup(sigmairest, sigmairest)
    Delta = np.block([[Delta1], [Delta2]])

    F = - 2 * Delta.T @ linalg.solve(L, Delta, assume_a="sym")
    return L, Delta, F


def local_influence_scs(Y, X, Z):
    n, m = X.shape
    q, p = Z.shape
    _, GAMMA, PHI, G, _ = mle_scs(Y, X, Z)
    xtx = X.T @ X
    Dq = u.dup_matrix(q)
    Dpq = u.dup_matrix(p * q)
    gammai = linalg.inv(GAMMA)  # TODO: Find analytic structure
    phii = linalg.inv(PHI)  # TODO: Find analytic structure

    L11 = u.kron(gammai, xtx)
    L12 = 0
    L13 = 0
    L22 = n / 2 * Dq.T @ linalg.solve(u.kron(GAMMA, GAMMA), Dq, assume_a="sym")
    L23 = 0
    L33 = n / 2 * Dpq.T @ linalg.solve(u.kron(PHI, PHI), Dpq, assume_a="sym")
    L = - np.block([[L11, L12, L13], [L12.T, L22, L23], [L13.T, L23.T, L33]])

    zzt = Z @ Z.T
    yztzzti = linalg.solve(zzt, Z @ Y.T, assume_a="sym").T
    ihx = np.identity(n) - X @ linalg.solve(xtx, X.T, assume_a="sym")
    giihxyztzzti = linalg.solve(GAMMA, ihx @ yztzzti, assume_a="sym")
    piggtigyt = linalg.solve(G @ G.T @ PHI, G @ Y.T)

    Delta1 = u.kron_diag_dup(giihxyztzzti, X.T)
    Delta2 = 0.5 * Dq.T @ u.kron_diag_dup(giihxyztzzti, giihxyztzzti)
    Delta3 = 0.5 * Dpq.T @ u.kron_diag_dup(piggtigyt, piggtigyt)
    Delta = np.block([[Delta1], [Delta2], [Delta3]])

    F = - 2 * Delta.T @ linalg.solve(L, Delta, assume_a="sym")
    return L, Delta, F