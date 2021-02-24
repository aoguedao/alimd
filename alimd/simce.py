import numpy as np
import pandas as pd
import logging


logger = logging.getLogger(__name__)


def input_preprocessing(filepath, sample_by_type=False):
    logger.info("Preprocessing files.")
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
        logger.info(f"Getting {sample_by_type} students per school type.")
        simce = simce.groupby("type").head(sample_by_type)
    else:
        logger.info("Using all students.")

    Y = simce.to_numpy()
    X = pd.get_dummies(simce.index.get_level_values("type")).to_numpy()
    Z = np.vstack([np.ones(simce.shape[1]), simce.columns.to_numpy()])

    return Y, X, Z
