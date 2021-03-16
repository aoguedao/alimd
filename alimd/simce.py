import logging
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)

def read_simce_csv(filepath):
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
    return simce

def read_input(filepath, sample=None):
    logger.info("Reading input files.")
    filepath = Path(filepath)
    if filepath.suffix == ".csv":
        logger.info("Reading .csv file")
        simce =  read_simce_csv(filepath)
        Y = simce.to_numpy()
        X = pd.get_dummies(simce.index.get_level_values("type")).to_numpy()
        Z = np.vstack([np.ones(simce.shape[1]), simce.columns.to_numpy()])
    elif filepath.suffix == ".npz":
        logger.info("Reading .npz file")
        inputfile = open(filepath, 'rb')
        npzfile = np.load(inputfile)
        Y, X, Z = npzfile["Y"], npzfile["X"], npzfile["Z"]
    else:
        raise NotImplementedError
    # sample
    if isinstance(sample, int):
        logger.info(f"Getting {sample} students.")
        X, _, Y, _ = train_test_split(
            X,
            Y,
            train_size=sample,
            stratify=X.argmax(axis=1),
            random_state=42
        )
    else:
        logger.info("Getting all students.")
    return Y, X, Z
