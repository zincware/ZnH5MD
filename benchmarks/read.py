from ase import Atoms
from src import (
    ASEIO,
    MDAIO,
    PLAMSIO,
    ChemfilesIO,
    MDTrajIO,
    benchmark_read,
    create_frames,
)


def main():
    # Create a set of frames for testing
    frames = create_frames(num_frames=10, num_atoms=100)

    # Initialize the ASEIO object with the generated frames
    ase_io = ASEIO(filename="test_ase_io", format="xyz", num_atoms=100, num_frames=10)
    ase_io.setup()
    ase_io.write(frames)

    # Benchmark the read performance
    metrics = benchmark_read(ase_io)
    print(f"Read Benchmark Metrics: {metrics}")

    # Initialize the MDAIO object with the generated frames
    mda_io = MDAIO(filename="test_ase_io", format="xyz", num_atoms=100, num_frames=10)
    mda_io.setup()
    metrics = benchmark_read(mda_io)
    print(f"Read Benchmark Metrics: {metrics}")

    # Initialize the ChemfilesIO object with the generated frames
    chemfiles_io = ChemfilesIO(
        filename="test_ase_io", format="xyz", num_atoms=100, num_frames=10
    )
    chemfiles_io.setup()
    metrics = benchmark_read(chemfiles_io)
    print(f"Read Benchmark Metrics: {metrics}")

    # Initialize the MDTrajIO object with the generated frames
    mdtraj_io = MDTrajIO(
        filename="test_ase_io", format=".xyz", num_atoms=100, num_frames=10
    )
    mdtraj_io.setup()
    metrics = benchmark_read(mdtraj_io)
    print(f"Read Benchmark Metrics: {metrics}")

    # Initialize the PLAMSIO object with the generated frames
    plams_io = PLAMSIO(
        filename="test_ase_io", format=".xyz", num_atoms=100, num_frames=10
    )
    plams_io.setup()
    metrics = benchmark_read(plams_io)
    print(f"Read Benchmark Metrics: {metrics}")


if __name__ == "__main__":
    main()
