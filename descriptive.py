import logging
import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

from alimd.simce import read_simce_csv 

sns.set_theme(style="ticks", palette="pastel")
sns.set_context("paper")

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option("--filepath", type=str)
@click.option("--imagespath", default=None, type=str)
def descriptive(filepath, imagespath):
    filepath = Path(filepath)
    if imagespath is None:
        imagespath = Path(__file__).resolve().parent / "images"
    else:
        imagespath = Path(imagespath)
    imagespath.mkdir(parents=True, exist_ok=True)
    logging.info("Reading file...")
    simce = read_simce_csv(filepath)
    # Number of students
    n_students = simce.index.get_level_values("RUT").nunique()
    logging.info(f"# Students: {n_students}")
    # Distribution by school type
    school_type_distr = (
        simce.reset_index()
        .assign(
            school_type=lambda x: x["type"].map({1: "Municipal", 2: "Subvencionado", 3: "Particular"}).astype("category")
        )
        .loc[:, "school_type"]
        .pipe(
            lambda x:
            pd.concat(
                [
                    x.value_counts().rename("n_students"),
                    x.value_counts(normalize=True).rename("perc_students") * 100
                ],
                axis=1
            )
        )
        .loc[["Municipal", "Subvencionado", "Particular"]]
    )
    print(school_type_distr.to_latex())
    # Boxplot distribution
    simce_melted = (
        simce.reset_index().melt(
            id_vars=["RUT", "type"],
            value_vars=[4, 8, 10],
            var_name="grade",
            value_name="Puntaje",
        )
        .assign(
            school_type=lambda x: x["type"].map({1: "Municipal", 2: "Subvencionado", 3: "Particular"}).astype("category"), 
            grade=lambda x: x["grade"].map({4: "2007 - 4° Básico", 8: "2011 - 8° Básico", 10: "2013 - II Medio"}).astype("category")
        )
        .rename(columns={"grade": "Año - Curso", "school_type": "Tipo Establecimiento"})
    )
    logging.info("Getting barplot")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x="Año - Curso",
        y="Puntaje",
        hue="Tipo Establecimiento",
        hue_order=["Municipal", "Subvencionado", "Particular"],
        ci="sd",
        capsize=.05,
        linewidth=1.25,
        edgecolor=".2",
        data=simce_melted,
        ax=ax,
    )
    sns.despine()
    fig.suptitle("Promedio de puntajes Simce - Matemáticas según tipo de Establecimiento")
    fig.tight_layout()
    fig.savefig(imagespath / f"simce_score_barplot.png")
    # fig.show()
    plt.close()


    logging.info("Getting boxplot")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        x="Año - Curso",
        y="Puntaje",
        hue="Tipo Establecimiento",
        hue_order=["Municipal", "Subvencionado", "Particular"],
        data=simce_melted,
        ax=ax
    )
    sns.despine()
    fig.suptitle("Distribución de puntajes Simce - Matemáticas según tipo de Establecimiento")
    fig.tight_layout()
    fig.savefig(imagespath / f"simce_score_boxplot.png", dpi=300)
    # fig.show()
    plt.close()

if __name__ == "__main__":
    descriptive()