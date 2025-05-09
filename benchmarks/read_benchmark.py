import numpy as np
import ase.io
import mdtraj as md
import chemfiles
import matplotlib.pyplot as plt
import time
import os
import pandas as pd

np.random.seed(42)

def create_trajectory_ase(n_frames=10, n_atoms=5) -> list[ase.Atoms]:
    """
    Create a random trajectory with n_frames and n_atoms using ASE.
    """
    frames = []
    for _ in range(n_frames):
        symbols = np.random.choice(['H', 'O', 'C'], size=n_atoms)
        positions = np.random.rand(n_atoms, 3)
        atoms = ase.Atoms(symbols=symbols, positions=positions)
        frames.append(atoms)
    return frames

def write_ase_xyz(filename: str, frames: list[ase.Atoms]):
    """
    Write a list of ASE Atoms objects to an XYZ file.
    """
    ase.io.write(filename, frames)

def read_ase(filename: str) -> list[ase.Atoms]:
    """
    Read a list of ASE Atoms objects from an XYZ file using ASE.
    """
    return ase.io.read(filename, index=':')

def read_mdtraj(filename: str) -> md.Trajectory:
    """
    Read a trajectory from an XYZ file using MDTraj.
    """
    # create a topology with 0 atoms
    top_df = pd.DataFrame({
        'residue': ['H'] * 100,
        'atom': ['H'] * 100,
        'element': ['H'] * 100,
        "name": ['H'] * 100,
        "resSeq": [1] * 100,
        "resName": ['H'] * 100,
        "chainID": ['A'] * 100,
        "serial": np.arange(100),
    })

    topology = md.Topology.from_dataframe(top_df)
    return md.load_xyz(filename, top=topology)

def benchmark_read(read_function, filename: str, num_repeats: int = 5) -> float:
    """
    Benchmark the read performance of a given function.
    """
    times = []
    for _ in range(num_repeats):
        start_time = time.time()
        read_function(filename)
        end_time = time.time()
        times.append(end_time - start_time)
    return np.mean(times)

def main():
    n_frames_list = np.logspace(1, 2, num=5, dtype=int)
    n_atoms_list = [100]
    file_formats = {
        'ASE': ('traj.xyz', 'xyz', read_ase),
        'MDTraj': ('traj.xyz', 'xyz', read_mdtraj),
    }
    results = {}

    print("Starting benchmark...")

    for n_atoms in n_atoms_list:
        results[n_atoms] = {}
        print(f"\nBenchmarking with {n_atoms} atoms:")
        for format_name, (filename, format_string, read_func) in file_formats.items():
            results[n_atoms][format_name] = []
            print(f"  Benchmarking {format_name} reading {format_string} file...")
            for n_frames in n_frames_list:
                print(f"    - {n_frames} frames...", end=' ')
                # Create and write the trajectory using ASE
                trajectory = create_trajectory_ase(n_frames=n_frames, n_atoms=n_atoms)
                ase.io.write(filename, trajectory, format=format_string)

                # Benchmark reading
                avg_time = benchmark_read(read_func, filename)
                results[n_atoms][format_name].append(avg_time)
                print(f"{avg_time:.4f} seconds")
                os.remove(filename) # Clean up the temporary file

    print("\nBenchmark complete. Generating plots...")

    # Plotting the results
    fig, axes = plt.subplots(1, len(n_atoms_list), figsize=(15, 5), sharey=True)
    if len(n_atoms_list) == 1:
        axes = [axes]  # Make sure axes is always a list for consistent indexing

    for i, n_atoms in enumerate(n_atoms_list):
        ax = axes[i]
        for format_name, times in results[n_atoms].items():
            ax.plot(n_frames_list, times, marker='o', label=format_name)

        ax.set_xlabel("Number of Frames")
        ax.set_ylabel("Read Time (seconds)")
        ax.set_title(f"{n_atoms} Atoms")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="-")
        ax.legend()

    fig.suptitle("Read Performance Benchmark", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
    plt.show()

if __name__ == "__main__":
    main()
