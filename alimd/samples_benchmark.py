import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path


sns.set_theme(style="whitegrid", palette="pastel")
sns.set_context("paper")


def read_log_as_serie(filepath):
    s_log = (
        pd.read_csv(filepath, sep=" - ", engine="python", header=None, usecols=[0, 3])
        .rename(columns={0: "dt_log", 2: "str_log"})
        .assign(dt_log=lambda x: pd.to_datetime(x["dt_log"], errors="ignore"))
        .set_index("dt_log")
        .squeeze()
    )
    return s_log


def get_n_sample(s):
    try:
        n_sample = s.loc[s.str.contains("Sample: ")].str.extract("(\d+)").astype(int).iloc[0, 0]
        return n_sample
    except:
        return None


def method_benchmark(s, method):
    s_method = s.loc[s.str.contains(method)]
    if s_method.size != 2 or s_method.str.lower().str.contains("error").any():
        raise "idk bro"
    method_minutes = (s_method.index[1] - s_method.index[0]).total_seconds() / 60
    largest_eigen = s_method.str.extract("(\d+\.\d+)", expand=False).astype(float).iloc[-1]
    return method_minutes, largest_eigen

@click.command()
@click.option("--path", type=str)
@click.option("--imagespath", default=None, type=str)
def main(path, imagespath):
    if imagespath is None:
        imagespath = Path(__file__).resolve().parent.parent / "images"
    else:
        imagespath = Path(imagespath)
    imagespath.mkdir(parents=True, exist_ok=True)

    log_files = Path(path).glob("*.log")
    s_logs = [read_log_as_serie(filepath) for filepath in log_files]
    methods = ["Brute", "Smart", "Stochastic"]
    benchmarks_dict = {}
    for s in s_logs:
        n_sample = get_n_sample(s)
        print(n_sample)
        if n_sample is None:
            continue 
        benchmarks_dict[n_sample] = (
            pd.DataFrame.from_dict(
                {method: method_benchmark(s, method) for method in methods}
            )
            .set_index(pd.Index(["minutes", "largest_eigen"]))
            .T
        )

    bench_df = (
        pd.concat(benchmarks_dict)
        .rename_axis(["n_sample", "method"])
        .assign(
            pct_diff_minutes=lambda x: (
                x.groupby("n_sample")["minutes"]
                .transform(lambda s: s.loc[(slice(None), "Brute")] / s)
            ),
            mape_largest_eigen=lambda x: (
                x.groupby("n_sample")["largest_eigen"]
                .transform(
                    lambda s: (s - s.loc[(slice(None), "Brute")]) / s.loc[(slice(None), "Brute")] * 100
                )  # Mean absolute percentage error
            )
        )
    )

    bench_error = (
        bench_df.loc[:, ["largest_eigen", "mape_largest_eigen"]]
        .reset_index()
        .pivot_table(
            index=["n_sample"],
            columns=["method"]
        )
        .drop(columns=("mape_largest_eigen", "Brute"))
    )

    bench_time = (
        bench_df.loc[:, ["minutes", "pct_diff_minutes"]]
        .reset_index()
        .pivot_table(
            index=["n_sample"],
            columns=["method"]
        )
        .drop(columns=("pct_diff_minutes", "Brute"))
    )

    print("##### Error largest eigenvalue benchmark #####")
    print(bench_error.to_latex())

    print("##### Execution time benchmark #####")
    print(bench_time.to_latex())

    ### Ploting ###
    method_name_dict = {
        "Brute": "Base",
        "Smart": "Descomposición",
        "Stochastic": "Probabilística"
    }
    bench_plot_df = (
        bench_df.rename_axis(["Tamaño Muestral", "Método"])
        .rename(
            columns={
                "minutes": "Tiempo de ejecución [min]",
                "largest_eigen": "Mayor valor propio",
                "times_faster": "Incremento tiempo ejecución",
                "mape_largest_eigen": "MAPE mayor valor propio"
            }
        )
        .reset_index()
        .assign(Método=lambda x: x["Método"].map(method_name_dict))
    )

    # Time
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x="Tamaño Muestral",
        y="Tiempo de ejecución [min]",
        hue="Método",
        capsize=.05,
        linewidth=1.25,
        edgecolor=".2",
        data=bench_plot_df,
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"Tiempo de ejecución [$\log$ min]")
    sns.despine()
    fig.suptitle("Tiempo de ejecución de cada método por tamaño de muestra")
    fig.tight_layout()
    fig.savefig(imagespath / f"sample_benchmark_time.png", dpi=300)
    fig.show()
    plt.close()

    # Eigen
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x="Tamaño Muestral",
        y="Mayor valor propio",
        hue="Método",
        capsize=.05,
        linewidth=1.25,
        edgecolor=".2",
        data=bench_plot_df,
        ax=ax,
    )
    ax.set_ylabel(r"$\left|\lambda_{\max}\right|$")
    sns.despine()
    fig.suptitle("Mayor valor propio de cada método por tamaño de muestra")
    fig.tight_layout()
    fig.savefig(imagespath / f"sample_benchmark_eigen.png", dpi=300)
    fig.show()
    plt.close()

    # Eigen error
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x="Tamaño Muestral",
        y="MAPE mayor valor propio",
        hue="Método",
        palette=sns.color_palette("pastel")[1:],
        capsize=.05,
        linewidth=1.25,
        edgecolor=".2",
        data=bench_plot_df.query("Método != 'Base'"),
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"$\log$ MAPE $\left| \lambda_{\max} \right|$")
    sns.despine()
    fig.suptitle("Error porcentual absoluto medio de cada método por tamaño de muestra")
    fig.tight_layout()
    fig.savefig(imagespath / f"sample_benchmark_eigen_mape.png", dpi=300)
    fig.show()
    plt.close()


if __name__ == "__main__":
    main()