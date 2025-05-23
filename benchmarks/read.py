import contextlib
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
    ZnH5MDFixedShapeIO,
    ZnH5MDIO,
    benchmark_read,
    create_frames,
)
from tqdm import tqdm

IO_CLASSES = [
    ASEIO,
    MDTrajIO,
    MDAIO,
    ChemfilesIO,
    PLAMSIO,
    ASECreate,
    ZnH5MDIO,
    ZnH5MDFixedShapeIO,
]


def benchmark_io_for_frame_count(
    num_frames: int,
    num_atoms: int,
    file_format: str,
    filename: str = "test_io_benchmark",
):
    frames = create_frames(num_frames=num_frames, num_atoms=num_atoms)
    results = {}

    # Determine the actual format and compression based on the input file_format
    actual_format = file_format
    if file_format == "h5md-uncompressed":
        actual_format = "h5md"

    # Create the appropriate writer instance
    writer_instance = None
    if actual_format == "xtc":
        writer_instance = MDTrajIO(
            filename=filename,
            format=actual_format,
            num_atoms=num_atoms,
            num_frames=num_frames,
        )
    elif actual_format == "h5md":
        writer_instance = ZnH5MDIO(
            filename=filename,
            format=actual_format,
            num_atoms=num_atoms,
            num_frames=num_frames,
            compression=None if file_format == "h5md-uncompressed" else "gzip",
        )
    else:
        writer_instance = ASEIO(
            filename=filename,
            format=actual_format,
            num_atoms=num_atoms,
            num_frames=num_frames,
        )

    writer_instance.setup()
    writer_instance.write(frames)

    try:
        # Benchmark all reader classes
        for io_cls in IO_CLASSES:
            # Create reader instance, handling h5md compression for ZnH5MDIO
            if io_cls == ZnH5MDIO:
                io_obj = io_cls(
                    filename=filename,
                    format=actual_format,
                    num_atoms=num_atoms,
                    num_frames=num_frames,
                )
            elif io_cls == ZnH5MDFixedShapeIO:
                io_obj = io_cls(
                    filename=filename,
                    format=actual_format,
                    num_atoms=num_atoms,
                    num_frames=num_frames,
                )
            else:
                io_obj = io_cls(
                    filename=filename,
                    format=actual_format,
                    num_atoms=num_atoms,
                    num_frames=num_frames,
                )

            io_obj.setup()
            with contextlib.suppress(
                ValueError
            ):  # Catch ValueError for unsupported formats
                metrics = benchmark_read(io_obj)
                results[io_cls.__name__] = metrics.asdict()
    finally:
        if os.path.exists(filename):
            os.unlink(filename)
    return results


def compute_benchmark_dfs(full_results, num_atoms: int):
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
    # convert to mus / atom
    df_avg = df_mean.divide(df_mean.index, axis=0).divide(num_atoms) * 1e6
    df_std_avg = df_std.divide(df_std.index, axis=0).divide(num_atoms) * 1e6

    return df_avg, df_std_avg


def plot_benchmarks(df_avg, df_std_avg, format: str):
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
    ax.set_xlabel("Number of Frames")
    ax.set_ylabel(r"Mean runtime per atom / $\mu$s/atom")
    ax.set_title(f"Read Benchmark ({format.upper()} format)")
    ax.set_yscale("log")
    ax.legend(title="Library")
    ax.grid(which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    ax.set_ylim(bottom=1e-3, top=1e2)
    plt.savefig(f"benchmark_{format}.png", dpi=300)


def main():
    num_atoms = 512
    for format in ["xtc", "h5md", "xyz", "pdb", "h5md-uncompressed"]:
        print(f"Running benchmark for {format.upper()} format")
        full_results = {
            num_frames: benchmark_io_for_frame_count(
                num_atoms=num_atoms, num_frames=num_frames, file_format=format
            )
            for num_frames in tqdm(np.logspace(2, 3, num=10, dtype=int))
        }

        df_avg, df_std_avg = compute_benchmark_dfs(full_results, num_atoms=num_atoms)
        plot_benchmarks(df_avg, df_std_avg, format=format)


if __name__ == "__main__":
    main()
