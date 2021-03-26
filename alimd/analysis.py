import logging
import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

from alimd.constants import SCHOOL_TYPE_NAMES_DICT

sns.set_theme(style="whitegrid", palette="pastel")
sns.set_context("paper")

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option("--path", type=str)
@click.option("--imagespath", default=None, type=str)
def main(path, imagespath):
    path = Path(path)
    logging.info(f"Reading input folder: {path}")
    with open(path / "yxz.npz", 'rb') as f:
        npzfile = np.load(f)
        Y, X, Z = npzfile["Y"], npzfile["X"], npzfile["Z"]
    with open(path / "reigh_eigen.npz", 'rb') as f:
        npzfile = np.load(f)
        w, v = npzfile["w"], npzfile["v"]
    
    # --- Make dataframe ----
    logging.info("Making dataframe")
    idx_max = np.argmax(np.abs(w))
    v_max = np.abs(v[:, idx_max])
    v_max_df = (
        pd.DataFrame(
            {
                "id": np.arange(len(v_max)) + 1,
                "v_max": v_max,
                "school_type": X.argmax(axis=1) + 1
            }
        )
        .assign(**{"Tipo Establecimiento": lambda x: x["school_type"].map(SCHOOL_TYPE_NAMES_DICT)})
    )
  
    # ---Plot 
    if imagespath is None:
        imagespath = Path(__file__).resolve().parent.parent / "images"
    else:
        imagespath = Path(imagespath)
    imagespath.mkdir(parents=True, exist_ok=True)

    index_plot(v_max_df, imagespath, n_std=3)
    school_type_index_plot(v_max_df, imagespath, n_std=5)
    threshold_df = threshold_summary(v_max_df, n_std=3)
    school_type_threshold_df = (
        v_max_df.groupby("Tipo Establecimiento")
        .apply(threshold_summary, n_std=5)
    )
    print(threshold_df.to_latex())
    print(school_type_threshold_df.to_latex())

def index_plot(v_max_df, imagespath, n_std=3):
    logging.info("Index plot using all data.")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.despine(fig)
    g = sns.scatterplot(
        x="id",
        y="v_max",
        hue="Tipo Establecimiento",
        palette="pastel",
        alpha=.75,
        linewidth=0,
        data=v_max_df,
        ax=ax,
    )
    scatter_legend  = g.legend_
    ax.add_artist(scatter_legend)
    for i in range(n_std + 1):
        label = r"mean($\left|h_{\max}\right|$) + $i \times$ std($\left|h_{\max}\right|$)"
        line = ax.hlines(
            v_max_df["v_max"].mean() + i * v_max_df["v_max"].std(),
            xmin=v_max_df["id"].min(),
            xmax=v_max_df["id"].max(),
            linewidth=1.5,
            linestyle="dashed",
            color='r',
            label=label
        )
    ax.legend(handles=[line], loc="upper left", title="Umbral de decisión")

    ax.set_ylabel(r"$\left|h_{\max}\right|$")
    ax.set_xlabel("Índice")
    fig.suptitle("Vector dirección de máxima curvatura")
    fig.tight_layout()
    fig.savefig(imagespath / f"index_plot_uc.png", dpi=300)
    fig.show()
    plt.close()


def school_type_index_plot(v_max_df, imagespath, n_std=3):
    logging.info("Index plot per each school type.")
    g = sns.relplot(
        data=v_max_df,
        x="id",
        y="v_max",
        row="Tipo Establecimiento",
        hue="Tipo Establecimiento",
        kind="scatter",
        palette="pastel",
        alpha=.75,
        linewidth=0,
        aspect=1.6,
        facet_kws=dict(sharey=False)
    )
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Vector dirección de máxima curvatura por tipo de establecimiento\n")
    (
        g.set_axis_labels("Índice", r"$\left|h_{\max}\right|$")
        .set_titles("{row_name}")
        .tight_layout(w_pad=0)
    )
    for school_type, ax in enumerate(g.axes.flat, start=1):
        v_max_mean = v_max_df.loc[lambda x: x["school_type"] == school_type, "v_max"].mean()
        v_max_std = v_max_df.loc[lambda x: x["school_type"] == school_type, "v_max"].std()
        for i in range(n_std + 1):
            label = r"mean($\left|h_{\max}\right|$) + $i \times$ std($\left|h_{\max}\right|$)"
            line = ax.hlines(
                v_max_mean + i * v_max_std,
                xmin=v_max_df["id"].min(),
                xmax=v_max_df["id"].max(),
                linewidth=1.5,
                linestyle="dashed",
                color='r',
                label=label
            )
        ax.legend(handles=[line], title="Umbral de decisión")
    g.savefig(imagespath / f"index_plot_school_type_uc.png", dpi=300)
    plt.close()


def threshold_summary(df, n_std=3):
    N = df.shape[0]
    mean = df["v_max"].mean()
    std = df["v_max"].std()
    threshold_df = pd.DataFrame().rename_axis("i")
    for i in range(n_std + 1):
        n = df["v_max"].gt(mean + i * std).sum()
        threshold_df.loc[i, ["# Estudiantes", "% Estudiantes"]] = [n, n / N * 100]
    return threshold_df

if __name__ == "__main__":
    main()