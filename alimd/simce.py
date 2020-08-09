import numpy as np
import pandas as pd
import click
import logging

from pathlib import Path
from datetime import datetime, timedelta, timezone
from ristretto import eigen

import gmanova


def input_preprocessing(filepath, sample_by_type=False):
    logging.info("Preprocessing files.")
    simce = (
        pd.read_csv(filepath)
        .groupby("RUT")
        .filter(
            lambda x: (x["type"].nunique() == 1)
    #         and (x["gender"].nunique() == 1)
            and (x["grade"].nunique() == 3)
        )
        .assign(score=lambda x: x["math"])
        .loc[: ,["RUT", "type", "grade", "score"]]
        .pivot_table(index=["type", "RUT"], columns="grade", values="score")
        .sort_index()
    )

    if sample_by_type and isinstance(sample_by_type, int):
        logging.info(f"Getting {sample_by_type} students per school type.")
        simce = simce.groupby("type").head(sample_by_type)
    else:
        logging.info("Using all students.")

    Y = simce.values
    X = pd.get_dummies(simce.index.get_level_values("type")).values
    Z = np.vstack([np.ones(simce.shape[1]), simce.columns.values])

    return Y, X, Z


def brute_force_eigen(F, outputpath):
    w_bf, v_bf = np.linalg.eigh(F)
    logging.info(f"Brute force larger eigen value: {np.abs(w_bf).max()}")
    bf_file = open(outputpath / "brute_force_eigen.npz", "wb")
    np.savez(bf_file, w=w_bf, v=v_bf)


def smart_eigen(F, L, Delta, outputpath):
    lw, lv = np.linalg.eigh(np.linalg.inv(-L))
    B = np.linalg.multi_dot([Delta.T, lv, np.sqrt(np.diag(lw))])
    u, dsr, vh = np.linalg.svd(B, full_matrices=False)
    w_smart = 2 * dsr ** 2
    logging.info(f"Smart decomposition larger eigen value: {np.abs(w_smart).max()}")
    smart_file = open(outputpath / "smart_eigen.npz", "wb")
    np.savez(smart_file, w=w_smart, v=u)
   

def stochastic_eigen(F, rank, outputpath):
    w_reigh, v_reigh = eigen.compute_reigh(F, rank, oversample=10, n_subspace=2)
    logging.info(f"Stochastic larger eigen value: {np.abs(w_reigh).max()}")
    reigh_file = open(outputpath / "reigh_eigen.npz", "wb")
    np.savez(reigh_file, w=w_reigh, v=v_reigh)


@click.command()
@click.option("--filepath", type=str)
@click.option("--outputpath", default=None, type=str)
@click.option("--preprocessed/--no-preprocessed", default=False)
@click.option("--sample_by_type", default=0, type=int)
def main(filepath, outputpath, preprocessed, sample_by_type):
    logging.info(f"File path: {filepath}")
    logging.info(f"Input files preprocessed: {preprocessed}")

    if outputpath is None:
        logging.info("Creating output folder.")
        tz = timezone(-timedelta(hours=4))
        start = datetime.now(tz).strftime('%Y-%m-%d-%H%M')
        outputpath = Path(__file__).resolve().parent.parent / "experiments" / f"{start}"
        outputpath.mkdir(parents=True, exist_ok=True)
    else:
        outputpath = Path(outputpath)
    logging.info(f"Output path: {outputpath}")

    if not preprocessed:
        Y, X, Z = input_preprocessing(filepath, sample_by_type)
    else:
        logging.info("Reading preprocessed files.")
        inputfile = open(filepath, 'rb')
        npzfile = np.load(inputfile)
        Y, X, Z = npzfile["Y"], npzfile["X"], npzfile["Z"]

    L, Delta, F = gmanova.local_influence(Y, X, Z)
    l_rank = np.linalg.matrix_rank(L, hermitian=True)
    logging.info("Computing eigen values.")
    try:
        brute_force_eigen(F, outputpath)  # Brute Force
    except Exception as e:
        logging.warning(e)
    
    try:
        smart_eigen(F, L, Delta, outputpath)  # Smart decomposition
    except Exception as e:
        logging.warning(e)

    try:    
        stochastic_eigen(F, l_rank, outputpath)  # Stochastic
    except Exception as e:
        logging.warning(e)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
