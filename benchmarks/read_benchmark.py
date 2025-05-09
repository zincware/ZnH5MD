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


class ASEIO:
    def __init__(self):
        self.filename = None

    def setup_xyz(self, filename: str, n_atoms: int):
        self.filename = filename

    def read_xyz(self):
        # Reads all frames from an XYZ file
        return ase.io.read(self.filename, index=":")

    def write_xyz(self, frames: list[ase.Atoms]):
        # Writes a list of ASE Atoms objects to an XYZ file
        ase.io.write(self.filename, frames, format="xyz")


class MDTrajIO:
    def __init__(self):
        self.filename = None
        self.topology = None

    def setup_xyz(self, filename: str, n_atoms: int):
        self.filename = filename
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

    def read_xyz(self):
        # Loads an XYZ trajectory. Uses the predefined topology.
        return md.load_xyz(self.filename, top=self.topology)

    def write_xyz(self, frames: list[ase.Atoms]):
        # Writing is done by ASEIO for this benchmark.
        pass


class ChemfilesIO:
    def __init__(self):
        self.filename = None

    def setup_xyz(
        self, filename: str, n_atoms: int
    ):  # n_atoms not strictly needed for Chemfiles setup
        self.filename = filename

    def read_xyz(self):
        # Reads all frames using Chemfiles
        # Chemfiles is generally good at inferring types from XYZ.
        trajectory = chemfiles.Trajectory(self.filename, "r")
        frames = [frame for frame in trajectory]  # Read all frames into a list
        trajectory.close()
        return frames

    def write_xyz(self, frames: list[ase.Atoms]):
        # Writing is done by ASEIO for this benchmark.
        pass


class MDAanalysisIO:  # Corrected typo from MDAanalysis to MDAnalysis if library is MDAnalysis
    def __init__(self):
        self.filename = None
        self.universe = None

    def setup_xyz(
        self, filename: str, n_atoms: int
    ):  # n_atoms not strictly needed for MDAnalysis setup
        self.filename = filename

    def read_xyz(self):
        # Loads the entire trajectory using MDAnalysis.
        # MDAnalysis should infer atom types from standard XYZ format.
        self.universe = mda.Universe(self.filename, format="XYZ", in_memory=True)
        return (
            self.universe
        )  # Or self.universe.trajectory to return the trajectory object


def benchmark_read(reader_object, num_repeats: int = 5) -> float:
    """
    Benchmark the read performance of a given reader object.
    Returns the mean time taken in seconds.
    """
    times = []
    for _ in range(num_repeats):
        start_time = time.time()
        _ = reader_object.read_xyz()  # Execute the read operation
        end_time = time.time()
        times.append(end_time - start_time)
    return np.mean(times)


def main():
    n_frames_list = np.logspace(1, 3, num=10, dtype=int)
    n_atoms_list = [100]
    num_benchmark_repeats = 3

    file_readers = {
        "ASE": ASEIO(),
        "MDTraj": MDTrajIO(),
        "Chemfiles": ChemfilesIO(),
        "MDAnalysis": MDAanalysisIO(),
    }

    # Dictionary to store results: results[n_atoms][lib_name] = [time1, time2, ...]
    results = {
        atoms: {lib_name: [] for lib_name in file_readers} for atoms in n_atoms_list
    }
    actual_n_frames_plotted = {atoms: list(n_frames_list) for atoms in n_atoms_list}

    # ASE writer instance to create trajectory files
    ase_writer = ASEIO()
    tmp_file_counter = 0

    for n_atoms in n_atoms_list:
        print(f"\n--- Benchmarking for {n_atoms} atoms ---")
        for n_frames in n_frames_list:
            # Generate a unique temporary filename
            filename = f"temp_trajectory_{n_frames}f_{n_atoms}a_{tmp_file_counter}.xyz"
            tmp_file_counter += 1

            print(
                f"  Generating trajectory: {n_frames} frames, {n_atoms} atoms. File: {filename}"
            )
            ase_frames_data = create_trajectory_ase(n_frames=n_frames, n_atoms=n_atoms)

            ase_writer.setup_xyz(
                filename, n_atoms
            )
            ase_writer.write_xyz(ase_frames_data)

            print(f"  Benchmarking read performance (repeats={num_benchmark_repeats}):")
            for lib_name, reader_object in file_readers.items():
                reader_object.setup_xyz(
                    filename, n_atoms
                )
                avg_time = benchmark_read(
                    reader_object, num_repeats=num_benchmark_repeats
                )
                results[n_atoms][lib_name].append(avg_time)
                print(f"    {lib_name:<12}: {avg_time:.4f} seconds")

            os.remove(filename)
        print(f"--- Finished benchmarking for {n_atoms} atoms ---")

    num_atom_configs = len(n_atoms_list)
    fig, axes = plt.subplots(
        num_atom_configs, 1, figsize=(12, 6 * num_atom_configs), squeeze=False
    )

    plot_filename = "xyz_read_benchmark.png"

    for i, n_atoms in enumerate(n_atoms_list):
        ax = axes[i, 0]
        plotted_frames = actual_n_frames_plotted[n_atoms]
        for lib_name in file_readers:
            times_to_plot = results[n_atoms][lib_name]
            ax.plot(
                plotted_frames, times_to_plot, marker="o", linestyle="-", label=lib_name
            )

        ax.set_xlabel("Number of Frames")
        ax.set_ylabel("Average Read Time (seconds)")
        ax.set_title(f"XYZ Read Performance ({n_atoms} Atoms)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    # plt.savefig(plot_filename)
    plt.show()  # Show the plot interactively
    print(f"\nBenchmark complete. Plot saved to {plot_filename}")


if __name__ == "__main__":
    main()
