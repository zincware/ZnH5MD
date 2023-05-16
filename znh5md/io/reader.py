import dataclasses
import logging
import typing

import ase.io
import chemfiles
import numpy as np
import tqdm
from ase.calculators.calculator import PropertyNotImplementedError

from znh5md.format import GRP
from znh5md.io.base import DataReader, FixedStepTimeChunk

log = logging.getLogger(__name__)


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
    use_pbc_group : bool, optional
        In addition to the 'boundary' group also
        use the 'pbc' group. This will allow time dependent
        periodic boundary conditions. This is not part of H5MD
        and might cause issues with other software!
    save_atoms_results : bool, optional
        Save 'atoms.calc.results' which can contain custom keys.
    """

    atoms: list[ase.Atoms]
    frames_per_chunk: int = 100  # must be larger than 1
    step: int = 1
    time: float = 1
    use_pbc_group: bool = False
    save_atoms_results: bool = True

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

    def _get_momenta(self, atoms: list[ase.Atoms]) -> np.ndarray:
        data = [x.arrays["momenta"] for x in atoms]
        try:
            return np.array(data).astype(float)
        except ValueError:
            return self._fill_with_nan(data).astype(float)

    def _get_stress(self, atoms: list[ase.Atoms]) -> np.ndarray:
        return np.array([x.get_stress() for x in atoms]).astype(float)

    def _get_edges(self, atoms: list[ase.Atoms]) -> np.ndarray:
        return np.array([x.get_cell() for x in atoms]).astype(float)

    def _get_pbc(self, atoms: list[ase.Atoms]) -> np.ndarray:
        return np.array([[x.get_pbc()] for x in atoms]).astype(bool)

    def _get_boundary(self, atoms: list[ase.Atoms]) -> np.ndarray:
        data = atoms[0].get_pbc()
        # boundary is constant and should be the same for all atoms
        return GRP.encode_boundary(data)

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
                GRP.momentum: self._get_momenta,
            }
            if self.use_pbc_group:
                functions[GRP.pbc] = self._get_pbc

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
                except (PropertyNotImplementedError, RuntimeError, KeyError) as err:
                    if group_names is not None:
                        # if the property was specifically selected, raise the error
                        raise err
                    else:
                        log.debug(f"Skipping {name} because {err}")

            if self.atoms[0].calc is not None and self.save_atoms_results:
                # we only gather the keys that are present in the first Atoms object.
                # We assume they occur in all the others as well.
                for key in self.atoms[0].calc.results:
                    if key not in functions:
                        value = [x.calc.results[key] for x in self.atoms]
                        try:
                            value = np.array(value).astype(float)
                        except ValueError:
                            value = self._fill_with_nan(value).astype(float)
                        data[key] = FixedStepTimeChunk(
                            value=value,
                            step=self.step,
                            time=self.time,
                        )
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
    use_pbc_group: bool = False

    def yield_chunks(self) -> typing.Iterator[typing.Dict[str, FixedStepTimeChunk]]:
        """Yield chunks using AtomsReader."""
        atoms_list = []
        reader = AtomsReader(
            None,  # we set atoms in the reading loop
            frames_per_chunk=self.frames_per_chunk,
            time=self.time,
            step=self.step,
            use_pbc_group=self.use_pbc_group,
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


@dataclasses.dataclass
class ChemfilesReader(DataReader):
    filename: str
    format: str = ""
    frames_per_chunk: int = 5000
    step: int = 1
    time: float = 1

    def yield_chunks(self) -> typing.Iterator[typing.Dict[str, FixedStepTimeChunk]]:
        with chemfiles.Trajectory(self.filename, format=self.format) as trajectory:
            positions = []
            species = []
            energy = []
            cell = []
            pbc = []
            for frame in tqdm.tqdm(trajectory, total=trajectory.nsteps):
                positions.append(np.copy(frame.positions))
                species.append(np.copy([atom.atomic_number for atom in frame.atoms]))
                cell.append(np.copy(frame.cell.lengths))

                if "energy" in frame.list_properties():
                    energy.append(np.copy(frame["energy"]).astype(float))
                if "pbc" in frame.list_properties():
                    pbc.append(
                        [True if x == "T" else False for x in frame["pbc"].split()]
                    )

                if len(positions) == self.frames_per_chunk:
                    positions = self._fill_with_nan(positions).astype(float)
                    species = self._fill_with_nan(species).astype(float)

                    data = {
                        GRP.position: FixedStepTimeChunk(
                            value=positions,
                            step=self.step,
                            time=self.time,
                        ),
                        GRP.species: FixedStepTimeChunk(
                            value=species,
                            step=self.step,
                            time=self.time,
                        ),
                        GRP.edges: FixedStepTimeChunk(
                            value=np.stack(cell),
                            step=self.step,
                            time=self.time,
                        ),
                    }
                    if len(energy) > 0:
                        data[GRP.energy] = FixedStepTimeChunk(
                            value=np.stack(energy),
                            step=self.step,
                            time=self.time,
                        )
                    if len(pbc) > 0:
                        data[GRP.pbc] = FixedStepTimeChunk(
                            value=np.stack(pbc),
                            step=self.step,
                            time=self.time,
                        )

                    yield data

                    positions = []
                    species = []
                    energy = []
                    cell = []
                    pbc = []
            if len(positions) > 0:
                positions = self._fill_with_nan(positions).astype(float)
                species = self._fill_with_nan(species).astype(float)

                data = {
                    GRP.position: FixedStepTimeChunk(
                        value=positions,
                        step=self.step,
                        time=self.time,
                    ),
                    GRP.species: FixedStepTimeChunk(
                        value=species,
                        step=self.step,
                        time=self.time,
                    ),
                    GRP.edges: FixedStepTimeChunk(
                        value=np.stack(cell),
                        step=self.step,
                        time=self.time,
                    ),
                }
                if len(energy) > 0:
                    data[GRP.energy] = FixedStepTimeChunk(
                        value=np.stack(energy),
                        step=self.step,
                        time=self.time,
                    )
                if len(pbc) > 0:
                    data[GRP.pbc] = FixedStepTimeChunk(
                        value=np.stack(pbc),
                        step=self.step,
                        time=self.time,
                    )
                yield data
