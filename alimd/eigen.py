import logging
import numpy as np
from ristretto import eigen

logger = logging.getLogger(__name__)


def brute_force_eigen(F, outputpath):
    logger.info(f"Brute force method.")
    w_bf, v_bf = np.linalg.eigh(F)
    logger.info(f"Brute force larger eigen value: {np.abs(w_bf).max()}")
    bf_file = open(outputpath / "brute_force_eigen.npz", "wb")
    np.savez(bf_file, w=w_bf, v=v_bf)


def smart_eigen(F, L, Delta, outputpath):
    logger.info(f"Smart method.")
    lw, lv = np.linalg.eigh(np.linalg.inv(-L))
    B = np.linalg.multi_dot([Delta.T, lv, np.sqrt(np.diag(lw))])
    u, dsr, vh = np.linalg.svd(B, full_matrices=False)
    w_smart = 2 * dsr ** 2
    logger.info(f"Smart decomposition larger eigen value: {np.abs(w_smart).max()}")
    smart_file = open(outputpath / "smart_eigen.npz", "wb")
    np.savez(smart_file, w=w_smart, v=u)


def stochastic_eigen(F, rank, outputpath):
    logger.info(f"Stochastic method.")
    w_reigh, v_reigh = eigen.compute_reigh(F, rank, oversample=10, n_subspace=2)
    logger.info(f"Stochastic larger eigen value: {np.abs(w_reigh).max()}")
    reigh_file = open(outputpath / "reigh_eigen.npz", "wb")
    np.savez(reigh_file, w=w_reigh, v=v_reigh)