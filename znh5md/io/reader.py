import dataclasses
import typing

import ase.io
import numpy as np
import tqdm
from ase.calculators.calculator import PropertyNotImplementedError

from znh5md.format import GRP
from znh5md.io.base import DataReader, FixedStepTimeChunk


@dataclasses.dataclass
class AtomsReader(DataReader):
    """Yield ase.Atoms objects from a list of Atoms Objects.

    Parameters
    ----------
    atoms : list[ase.Atoms]
        List of ase.Atoms objects.
    frames_per_chunk : int, optional
        Number of frames to read at once, by default 100
    step : int, optional
        Step size, by default 1
    time : float, optional
        Time step, by default 1
    """

    atoms: list[ase.Atoms]
    frames_per_chunk: int = 100  # must be larger than 1
    step: int = 1
    time: float = 1

    def _fill_with_nan(self, data: list) -> np.ndarray:
        max_n_particles = max(x.shape[0] for x in data)
        dimensions = data[0].shape[1:]

        result = np.full((len(data), max_n_particles, *dimensions), np.nan)
        for i, x in enumerate(data):
            result[i, : x.shape[0], ...] = x
        return result

    def _get_positions(self, atoms: list[ase.Atoms]) -> np.ndarray:
        data = [x.get_positions() for x in atoms]
        try:
            return np.array(data).astype(float)
        except ValueError:
            return self._fill_with_nan(data).astype(float)

    def _get_energy(self, atoms: list[ase.Atoms]) -> np.ndarray:
        return np.array([x.get_potential_energy() for x in atoms]).astype(float)

    def _get_species(self, atoms: list[ase.Atoms]) -> np.ndarray:
        data = [x.get_atomic_numbers() for x in atoms]
        try:
            return np.array(data).astype(float)
        # NaN is only supported for float, not int
        except ValueError:
            return self._fill_with_nan(data).astype(float)

    def _get_forces(self, atoms: list[ase.Atoms]) -> np.ndarray:
        data = [x.get_forces() for x in atoms]
        try:
            return np.array(data).astype(float)
        except ValueError:
            return self._fill_with_nan(data).astype(float)

    def _get_stress(self, atoms: list[ase.Atoms]) -> np.ndarray:
        return np.array([x.get_stress() for x in atoms]).astype(float)

    def _get_edges(self, atoms: list[ase.Atoms]) -> np.ndarray:
        return np.array([x.get_cell() for x in atoms]).astype(float)

    def _get_boundary(self, atoms: list[ase.Atoms]) -> np.ndarray:
        return np.array([[x.get_pbc()] for x in atoms]).astype(bool)

    def yield_chunks(
        self, group_names: list = None
    ) -> typing.Iterator[typing.Dict[str, FixedStepTimeChunk]]:
        start_index = 0
        stop_index = 0

        pbar = tqdm.tqdm(
            total=len(self.atoms), disable=len(self.atoms) // self.frames_per_chunk < 10
        )

        while stop_index < len(self.atoms):
            stop_index = start_index + self.frames_per_chunk
            data = {}

            functions = {
                GRP.position: self._get_positions,
                GRP.energy: self._get_energy,
                GRP.species: self._get_species,
                GRP.forces: self._get_forces,
                GRP.stress: self._get_stress,
                GRP.edges: self._get_edges,
                GRP.boundary: self._get_boundary,
            }

            for name in group_names or functions:
                if name not in functions:
                    raise ValueError(f"Value {name} not supported")

                try:
                    value = functions[name](self.atoms[start_index:stop_index])
                    data[name] = FixedStepTimeChunk(
                        value=value,
                        step=self.step,
                        time=self.time,
                    )
                except (PropertyNotImplementedError, RuntimeError) as err:
                    if group_names is not None:
                        # if the property was specifically selected, raise the error
                        raise err
                    else:
                        continue
            yield data
            start_index = stop_index
            pbar.update(self.frames_per_chunk)

        pbar.close()


@dataclasses.dataclass
class ASEFileReader(DataReader):
    """Use ASE to read files.

    Parameters
    ----------
    filename : str
        Path to the file. Any format supported by ASE is supported.
    frames_per_chunk : int, optional
        Number of frames to read at once
    time : float, optional
        Time step
    step : int, optional
        Step size
    """

    filename: str
    frames_per_chunk: int = 5000
    time: float = 1
    step: int = 1

    def yield_chunks(self) -> typing.Iterator[typing.Dict[str, FixedStepTimeChunk]]:
        """Yield chunks using AtomsReader."""
        atoms_list = []
        reader = AtomsReader(
            None,  # we set atoms in the reading loop
            frames_per_chunk=self.frames_per_chunk,
            time=self.time,
            step=self.step,
        )

        for atoms in tqdm.tqdm(ase.io.iread(self.filename)):
            atoms_list.append(atoms)
            if len(atoms_list) == self.frames_per_chunk:
                reader.atoms = atoms_list
                yield from reader.yield_chunks()
                atoms_list = []
        if len(atoms_list) > 0:
            reader.atoms = atoms_list
            yield from reader.yield_chunks()
