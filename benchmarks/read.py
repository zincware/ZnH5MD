import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src import (
    ASEIO,
    MDAIO,
    PLAMSIO,
    ASECreate,
    ChemfilesIO,
    MDTrajIO,
    benchmark_read,
    create_frames,
)
from tqdm import tqdm

IO_CLASSES = [ASEIO, MDAIO, ChemfilesIO, MDTrajIO, PLAMSIO, ASECreate]


def benchmark_io_for_frame_count(
    num_frames: int, num_atoms: int = 100, filename: str = "test_ase_io"
):
    frames = create_frames(num_frames=num_frames, num_atoms=num_atoms)
    results = {}

    for IOClass in IO_CLASSES:
        io_obj = IOClass(
            filename=filename, format="xyz", num_atoms=num_atoms, num_frames=num_frames
        )
        io_obj.setup()
        if IOClass is ASEIO:
            io_obj.write(frames)
        metrics = benchmark_read(io_obj)
        results[IOClass.__name__] = metrics.asdict()

    os.unlink(filename)
    return results


def compute_benchmark_dfs(full_results):
    df_mean = pd.DataFrame.from_dict(
        {
            k: {
                name: values["mean"] - v[ASECreate.__name__]["mean"]
                for name, values in v.items()
            }
            for k, v in full_results.items()
        },
        orient="index",
    )
    df_mean.index.name = "Number of Frames"

    df_std = pd.DataFrame.from_dict(
        {
            k: {name: values["std"] for name, values in v.items()}
            for k, v in full_results.items()
        },
        orient="index",
    )

    df_avg = df_mean.divide(df_mean.index, axis=0) * 1000
    df_std_avg = df_std.divide(df_std.index, axis=0) * 1000

    return df_avg, df_std_avg


def plot_benchmarks(df_avg, df_std_avg):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    markers = ["o", "s", "^", "v", "p", "*"]

    for i, library_name in enumerate(df_avg.columns):
        if library_name == ASECreate.__name__:
            continue
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax.plot(
            df_avg.index,
            df_avg[library_name],
            label=library_name,
            marker=marker,
            markersize=8,
            color=color,
            linewidth=2,
        )
        ax.errorbar(
            df_avg.index,
            df_avg[library_name],
            yerr=df_std_avg[library_name],
            fmt="none",
            capsize=5,
            color=color,
            alpha=0.7,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of Frames", fontsize=12)
    ax.set_ylabel("Mean runtime per atom (ms/atom)", fontsize=12)
    ax.set_title("Read Benchmark", fontsize=14, fontweight="bold")
    ax.legend(title="Library", fontsize=10)
    plt.tight_layout()
    plt.show()


def main():
    full_results = {
        num_frames: benchmark_io_for_frame_count(num_frames)
        for num_frames in tqdm(np.logspace(2, 3, num=10, dtype=int))
    }

    df_avg, df_std_avg = compute_benchmark_dfs(full_results)
    plot_benchmarks(df_avg, df_std_avg)


if __name__ == "__main__":
    main()
