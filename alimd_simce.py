import sys
import logging
import click
import numpy as np

from pathlib import Path
from datetime import date, datetime, timedelta, timezone

import alimd


# logger = logging.getLogger("alimd-simce")


@click.command()
@click.option("--filepath", type=str)
@click.option("--outputpath", default=None, type=str)
@click.option("--preprocessed/--no-preprocessed", default=False)
@click.option("--sample_by_type", default=0, type=int)
def main(filepath, outputpath, preprocessed, sample_by_type):
    logger.info(f"File path: {filepath}")
    logger.info(f"Input files preprocessed: {preprocessed}")

    if outputpath is None:
        logger.info("Creating output folder.")
        # tz = timezone(-timedelta(hours=4))
        # start = datetime.now(tz).strftime('%Y-%m-%d-%H%M')
        if sample_by_type != 0:
            outputpath = Path(__file__).resolve().parent / "output" / f"sample_{sample_by_type}"
        else:
            outputpath = Path(__file__).resolve().parent / "output" / f"all_data"
    else:
        outputpath = Path(outputpath)
    outputpath.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output path: {outputpath}")

    if not preprocessed:
        Y, X, Z = alimd.simce.input_preprocessing(filepath, sample_by_type)
        yxz_filename = "yxz" if not sample_by_type else f"yxz_sample_{sample_by_type}"
        yxz_file = open(outputpath / f"{yxz_filename}.npz", "wb")
        np.savez(yxz_file, X=X, Y=Y, Z=Z)
    else:
        logger.info("Reading preprocessed files.")
        inputfile = open(filepath, 'rb')
        npzfile = np.load(inputfile)
        Y, X, Z = npzfile["Y"], npzfile["X"], npzfile["Z"]

    L, Delta, F = alimd.gmanova.local_influence(Y, X, Z, cov_structure="uc")
    l_rank = np.linalg.matrix_rank(L, hermitian=True)

    logger.info("Computing eigen values.")
    try:
        alimd.eigen.brute_force_eigen(F, outputpath)  # Brute Force
    except Exception as e:
        logger.warning("Brute force method error.", exc_info=True)
        # logger.warning(e)
    
    try:
        alimd.eigen.smart_eigen(F, L, Delta, outputpath)  # Smart decomposition
    except Exception as e:
        logger.warning("Smart method error.", exc_info=True)
        # logger.warning(e)

    try:    
        alimd.eigen.stochastic_eigen(F, l_rank, outputpath)  # Stochastic
    except Exception as e:
        logger.warning("Stochastic method error.", exc_info=True)
        # logger.warning(e)


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