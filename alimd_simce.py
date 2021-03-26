import sys
import logging
import click
import numpy as np

from pathlib import Path
from datetime import date, datetime, timedelta, timezone

import alimd


@click.command()
@click.option("--filepath", type=str)
@click.option("--cov", default="uc", type=str)
@click.option("--outputpath", default=None, type=str)
@click.option("--sample", default=None, type=int)
def main(filepath, cov, outputpath, sample):
    logger.info(f"File path: {filepath}")
    if cov in ["uc", "scs"]:
        logger.info(f"Covariance structure: {cov}")
    else:
        raise f"Covariance structure must be 'uc' or 'scs'."
    if outputpath is None:
        logger.info("Making output folder.")
        if sample is not None:
            outputpath = Path(__file__).resolve().parent / "output" / cov / f"sample_{sample}"
        else:
            outputpath = Path(__file__).resolve().parent / "output" / cov / "all_data"
    else:
        outputpath = Path(outputpath)
    outputpath.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output path: {outputpath}")
    Y, X, Z = alimd.simce.read_input(filepath, sample)
    yxz_filename = "yxz" if sample is None else f"yxz_sample_{sample}"
    yxz_file = open(outputpath / f"{yxz_filename}.npz", "wb")
    np.savez(yxz_file, X=X, Y=Y, Z=Z)

    L, Delta, F = alimd.gmanova.local_influence(Y, X, Z, cov_structure=cov)
    l_rank = np.linalg.matrix_rank(L, hermitian=True)

    logger.info("Computing eigen values.")
    try:
        alimd.eigen.brute_force_eigen(F, outputpath)  # Brute Force
    except Exception as e:
        logger.warning("Brute force method error.", exc_info=True)
    
    try:
        alimd.eigen.smart_eigen(F, L, Delta, outputpath)  # Smart decomposition
    except Exception as e:
        logger.warning("Smart method error.", exc_info=True)

    try:    
        alimd.eigen.stochastic_eigen(F, l_rank, outputpath)  # Stochastic
    except Exception as e:
        logger.warning("Stochastic method error.", exc_info=True)


if __name__ == "__main__":
    now = datetime.now().strftime("%Y-%m-%d-%H:%M")
    log_filepath = Path() / "logs" / f"alimd-simce-{now}.log"
    file_handler = logging.FileHandler(filename=log_filepath, mode="w")
    file_handler.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    handlers = [file_handler, stdout_handler]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        # filename=f"basic-alimd-simce-{now}.log"
    )

    logger = logging.getLogger("alimd-simce")
    main()