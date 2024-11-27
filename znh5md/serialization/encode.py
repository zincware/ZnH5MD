import ase
import numpy as np

from znh5md.serialization import MISSING, Frames

CONTENT_TYPE = dict[str, np.ndarray|dict|float|int|str|bool]

def process_category(target: dict[str, list], content: CONTENT_TYPE, index) -> None:
    """
    Process a category (arrays, info, calc) for a single atoms object, ensuring
    that keys are added and missing values are backfilled.

    Parameters:
        target (dict): The main dictionary storing data for the category.
        atoms_data (dict): The data from the current atoms object (arrays, info, or calc).
        index (int): The index of the current atoms object in the trajectory used for backfilling.
    """

    seen = set(content.keys())
    unseen = set(target.keys()) - seen

    for key in content:
        if key not in target:
            # Backfill existing entries with MISSING for the new key
            target[key] = [MISSING] * index + [content[key]]
        else:
            # Add the new data to the existing key
            target[key].append(content[key])
    
    for key in unseen:
        # Backfill missing entries with MISSING for the unseen key
        target[key].append(MISSING)



def encode(frames: list[ase.Atoms]) -> Frames:
    positions = np.array([atoms.positions for atoms in frames], dtype=object)
    numbers = np.array([atoms.numbers for atoms in frames], dtype=object)
    pbc = np.array([atoms.pbc for atoms in frames])
    cell = np.array([atoms.cell.array for atoms in frames])

    arrays = {}
    info = {}
    calc = {}

    for idx, atoms in enumerate(frames):
        # Process arrays
        atoms_arrays = {
            key: value
            for key, value in atoms.arrays.items()
            if key not in ["positions", "numbers"]
        }
        process_category(arrays, atoms_arrays, idx)

        # Process info
        process_category(info, atoms.info,  idx)

        # Process calc
        if atoms.calc is not None:
            process_category(calc, atoms.calc.results,  idx)
    
    for key in arrays:
        arrays[key] = np.array(arrays[key], dtype=object)
    for key in info:
        info[key] = np.array(info[key], dtype=object)
    for key in calc:
        calc[key] = np.array(calc[key], dtype=object)

    return Frames(
        positions=positions,
        numbers=numbers,
        pbc=pbc,
        cell=cell,
        arrays=arrays,
        info=info,
        calc=calc,
    )
