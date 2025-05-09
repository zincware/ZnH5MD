import os
import time

import ase
import ase.io
import chemfiles
import matplotlib.pyplot as plt
import MDAnalysis as mda
import mdtraj as md
import numpy as np
import pandas as pd
from ase.io.proteindatabank import write_proteindatabank
from MDAnalysis.coordinates.H5MD import H5MDReader

import znh5md

np.random.seed(42)


def create_trajectory_ase(n_frames=10, n_atoms=5) -> list[ase.Atoms]:
    """
    Create a random trajectory with n_frames and n_atoms using ASE.
    Each frame will have a random selection of H, O, C atoms.
    """
    frames = []
    for _ in range(n_frames):
        symbols = np.random.choice(["H", "O", "C"], size=n_atoms)
        positions = np.random.rand(n_atoms, 3) * 10
        cell_vectors = np.eye(3) * 100
        atoms = ase.Atoms(
            symbols=symbols, positions=positions, cell=cell_vectors, pbc=False
        )
        frames.append(atoms)
    return frames


class ZnH5MDIO:
    def __init__(self):
        self.filename = None
        self.current_format = None

    def setup(self, filename: str, n_atoms: int, format: str):
        self.filename = filename
        self.current_format = format

    def read(self):
        if self.current_format == "h5":
            io = znh5md.IO(self.filename)
            return io[:]
        else:
            raise ValueError(f"Unsupported format: {self.current_format}")

    def write(self, frames: list[ase.Atoms]):
        pass


class ASEIO:
    def __init__(self):
        self.filename = None
        self.current_format = None

    def setup(self, filename: str, n_atoms: int, format: str):
        self.filename = filename
        self.current_format = format

    def read(self):
        if self.current_format in ["xyz", "h5"]:
            return ase.io.read(self.filename, index=":")
        elif self.current_format == "pdb":
            from ase.io.proteindatabank import read_proteindatabank

            return read_proteindatabank(self.filename)
        else:
            raise ValueError(f"Unsupported format: {self.current_format}")

    def write(self, frames: list[ase.Atoms]):
        if self.current_format in ["xyz"]:
            ase.io.write(self.filename, frames)
        elif self.current_format == "pdb":
            write_proteindatabank(self.filename, frames)
        elif self.current_format == "h5":
            io = znh5md.IO(self.filename, save_units=False, store="time")
            io.extend(frames)
        else:
            raise ValueError(f"Unsupported format: {self.current_format}")


class MDTrajIO:
    def __init__(self):
        self.filename = None
        self.topology = None
        self.current_format = None

    def setup(self, filename: str, n_atoms: int, format: str):
        self.filename = filename
        self.current_format = format
        # Create a minimal topology DataFrame.
        atom_names = [f"H{i + 1}" for i in range(n_atoms)]  # Unique atom names
        elements = [
            "H"
        ] * n_atoms  # All elements are H as per original user code structure

        top_df = pd.DataFrame(
            {
                "name": atom_names,  # Atom names (e.g., H1, H2)
                "element": elements,  # Element symbols (e.g., H, C, O)
                "resSeq": [1] * n_atoms,
                "resName": ["MOL"] * n_atoms,  # Using a generic residue name
                "chainID": ["A"] * n_atoms,
                "serial": np.arange(n_atoms),
            }
        )
        self.topology = md.Topology.from_dataframe(top_df)

    def read(self):
        if self.current_format == "xyz":
            # Loads an XYZ trajectory. Uses the predefined topology.
            return md.load_xyz(self.filename, top=self.topology)
        elif self.current_format == "pdb":
            return md.load_pdb(self.filename)
        elif self.current_format == "xtc":
            return md.load_xtc(self.filename, top=self.topology)
        else:
            raise ValueError(f"Unsupported format: {self.current_format}")

    def write(self, frames: list[ase.Atoms]):
        # Writing is done by ASEIO for this benchmark.
        if self.current_format == "xtc":
            # use ase to write xyz, load that file and then save as xtc
            write_proteindatabank(
                "ase_pdb_to_xtc_tmp.pdb", frames
            )  # Write to a temporary XYZ file
            traj = md.load_pdb("ase_pdb_to_xtc_tmp.pdb")
            traj.save_xtc(self.filename)
            # remove the temporary file
            os.remove("ase_pdb_to_xtc_tmp.pdb")


class ChemfilesIO:
    def __init__(self):
        self.filename = None
        self.current_format = None

    def setup(self, filename: str, n_atoms: int, format: str):
        self.filename = filename
        self.current_format = format

    def read(self):
        if self.current_format in ["xyz", "pdb", "xtc"]:
            trajectory = chemfiles.Trajectory(self.filename, "r")
            # access data to ensure loading
            for frame in trajectory:
                _  = frame.positions
            trajectory.close()
            return trajectory
        else:
            raise ValueError(f"Unsupported format: {self.current_format}")

    def write(self, frames: list[ase.Atoms]):
        # Writing is done by ASEIO for this benchmark.
        pass


class MDAanalysisIO:
    def __init__(self):
        self.filename = None
        self.universe = None
        self.current_format = None
        self.n_atoms = None

    def setup(self, filename: str, n_atoms: int, format: str):
        self.filename = filename
        self.current_format = format
        self.n_atoms = n_atoms

    def read(self):
        # Loads the entire trajectory using MDAnalysis.
        if self.current_format == "xyz":
            self.universe = mda.Universe(self.filename, format="XYZ")
        elif self.current_format == "pdb":
            self.universe = mda.Universe(self.filename)
        elif self.current_format == "h5":
            self.universe = mda.Universe.empty(self.n_atoms, trajectory=True)
            reader = H5MDReader(
                self.filename,
                convert_units=False,
                dt=2,
                time_offset=10,
                foo="bar",
            )
            self.universe.trajectory = reader
        else:
            raise ValueError(f"Unsupported format: {self.current_format}")

        self.universe.transfer_to_memory()
        return self.universe.trajectory


def benchmark_read(reader_object, num_repeats: int = 5) -> float:
    """
    Benchmark the read performance of a given reader object.
    Returns the mean time taken in seconds.
    """
    times = []
    for _ in range(num_repeats):
        start_time = time.time()
        _ = reader_object.read()  # Execute the read operation
        end_time = time.time()
        times.append(end_time - start_time)
    return np.mean(times)


def main():
    n_frames_list = np.logspace(1, 4, num=10, dtype=int)
    n_atoms_list = [100, 200]
    num_benchmark_repeats = 10
    formats_to_benchmark = ["xyz", "pdb", "h5", "xtc"]

    file_readers = {
        "ASE": ASEIO(),
        "MDTraj": MDTrajIO(),
        "Chemfiles": ChemfilesIO(),
        "MDAnalysis": MDAanalysisIO(),
        "ZnH5MD": ZnH5MDIO(),
    }

    # Dictionary to store results: results[format][n_atoms][lib_name] = [time1, time2, ...]
    results = {
        fmt: {
            atoms: {lib_name: [] for lib_name in file_readers} for atoms in n_atoms_list
        }
        for fmt in formats_to_benchmark
    }
    actual_n_frames_plotted = {
        fmt: {atoms: list(n_frames_list) for atoms in n_atoms_list}
        for fmt in formats_to_benchmark
    }

    # ASE writer instance to create trajectory files
    ase_writer = ASEIO()
    tmp_file_counter = 0

    for fmt in formats_to_benchmark:
        print(f"\n--- Benchmarking format: {fmt.upper()} ---")
        for n_atoms in n_atoms_list:
            print(f"\n  --- Benchmarking for {n_atoms} atoms ---")
            for n_frames in n_frames_list:
                # Generate a unique temporary filename
                filename = (
                    f"temp_trajectory_{n_frames}f_{n_atoms}a_{tmp_file_counter}.{fmt}"
                )
                tmp_file_counter += 1

                print(
                    f"    Generating trajectory: {n_frames} frames, {n_atoms} atoms. File: {filename}"
                )
                ase_frames_data = create_trajectory_ase(
                    n_frames=n_frames, n_atoms=n_atoms
                )

                if fmt == "xtc":
                    # use mdtraj to write xtc
                    writer = MDTrajIO()
                    writer.setup(filename, n_atoms, fmt)
                    writer.write(ase_frames_data)
                else:
                    ase_writer.setup(filename, n_atoms, fmt)
                    ase_writer.write(ase_frames_data)

                print(
                    f"    Benchmarking read performance (repeats={num_benchmark_repeats}):"
                )
                for lib_name, reader_object in file_readers.items():
                    try:
                        reader_object.setup(filename, n_atoms, fmt)
                        avg_time = benchmark_read(
                            reader_object, num_repeats=num_benchmark_repeats
                        )
                        results[fmt][n_atoms][lib_name].append(avg_time)
                        print(f"      {lib_name:<12}: {avg_time:.4f} seconds")
                    except ValueError as e:
                        print(
                            f"      {lib_name:<12}: Not supported for {fmt.upper()} ({e})"
                        )
                        results[fmt][n_atoms][lib_name].append(
                            np.nan
                        )  # Mark as not supported

                os.remove(filename)
            print(f"  --- Finished benchmarking for {n_atoms} atoms ---")
        print(f"--- Finished benchmarking format: {fmt.upper()} ---")

    plot_filename = "trajectory_read_benchmark.png"
    num_formats = len(formats_to_benchmark)
    fig, axes = plt.subplots(
        num_formats * len(n_atoms_list),
        1,
        figsize=(6, 3 * num_formats * len(n_atoms_list)),
        squeeze=False,
    )
    plot_index = 0

    for fmt in formats_to_benchmark:
        for i, n_atoms in enumerate(n_atoms_list):
            ax = axes[plot_index, 0]
            plotted_frames = actual_n_frames_plotted[fmt][n_atoms]
            for lib_name in file_readers:
                times_to_plot = results[fmt][n_atoms][lib_name]
                ax.plot(
                    plotted_frames,
                    times_to_plot,
                    marker="o",
                    linestyle="-",
                    label=lib_name,
                )

            ax.set_xlabel("Number of Frames")
            ax.set_ylabel("Average Read Time (seconds)")
            ax.set_title(f"{fmt.upper()} Read Performance ({n_atoms} Atoms)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend()
            ax.grid(True, which="both", ls="-", alpha=0.5)
            plot_index += 1

    plt.tight_layout()
    plt.savefig(plot_filename)
    # plt.show()  # Show the plot interactively
    print(f"\nBenchmark complete. Plot saved to {plot_filename}")

    # find the fastest reader for each format
    fastest_readers = {}
    for fmt in formats_to_benchmark:
        fastest_readers[fmt] = {}
        for n_atoms in n_atoms_list:
            fastest_time = np.inf
            fastest_lib = None
            for lib_name, times in results[fmt][n_atoms].items():
                mean_time = np.nanmean(times)
                if mean_time < fastest_time:
                    fastest_time = mean_time
                    fastest_lib = lib_name
            fastest_readers[fmt][n_atoms] = (fastest_lib, fastest_time)
            print(
                f"Fastest reader for {fmt.upper()} with {n_atoms} atoms: {fastest_lib} ({fastest_time:.4f} seconds)"
            )
    # now make a plot of the fastest readers
    fig, ax = plt.subplots(figsize=(8, 6))
    for fmt in formats_to_benchmark:
        n_atoms = n_atoms_list[0]  # Use the first n_atoms for plotting
        fastest_lib, fastest_time = fastest_readers[fmt][n_atoms]
        ax.bar(fmt, fastest_time, label=fastest_lib)
    ax.set_xlabel("File Format")
    ax.set_ylabel("Fastest Read Time (seconds)")
    ax.set_title("Fastest Reader for Each File Format")
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig("fastest_readers.png")



if __name__ == "__main__":
    main()
