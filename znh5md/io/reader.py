import dataclasses
import typing

import ase
import numpy as np
import tqdm
from ase.calculators.calculator import PropertyNotImplementedError

from znh5md.io.base import DataReader, ExplicitStepTimeChunk


@dataclasses.dataclass
class AtomsReader(DataReader):
    atoms: list[ase.Atoms]
    frames_per_chunk: int = 100

    def _get_positions(self, atoms: list[ase.Atoms]) -> np.ndarray:
        return np.array([x.get_positions() for x in atoms])

    def _get_energy(self, atoms: list[ase.Atoms]) -> np.ndarray:
        return np.array([x.get_potential_energy() for x in atoms])

    def _get_species(self, atoms: list[ase.Atoms]) -> np.ndarray:
        return np.array([x.get_atomic_numbers() for x in atoms])

    def _get_forces(self, atoms: list[ase.Atoms]) -> np.ndarray:
        return np.array([x.get_forces() for x in atoms])

    def _get_stress(self, atoms: list[ase.Atoms]) -> np.ndarray:
        return np.array([x.get_stress() for x in atoms])

    def _get_box(self, atoms: list[ase.Atoms]) -> np.ndarray:
        return np.array([x.get_cell() for x in atoms])

    def yield_chunks(
        self, group_name: list = None
    ) -> typing.Iterator[typing.Dict[str, ExplicitStepTimeChunk]]:
        start_index = 0
        stop_index = 0

        pbar = tqdm.tqdm(
            total=len(self.atoms), disable=len(self.atoms) // self.frames_per_chunk < 10
        )

        while stop_index < len(self.atoms):
            stop_index = start_index + self.frames_per_chunk
            data = {}

            functions = {
                "position": self._get_positions,
                "energy": self._get_energy,
                "species": self._get_species,
                "forces": self._get_forces,
                "stress": self._get_stress,
                "box": self._get_box,
            }

            for name in group_name or functions:
                if name not in functions:
                    raise ValueError(f"Value {name} not supported")

                try:
                    value = functions[name](self.atoms[start_index:stop_index])
                    data[name] = ExplicitStepTimeChunk(
                        value=value,
                        step=np.arange(start_index, start_index + len(value)),
                        time=np.arange(start_index, start_index + len(value)),
                    )
                except PropertyNotImplementedError as err:
                    if group_name is not None:
                        # if the property was specifcally selected, raise the error
                        raise err
                    else:
                        continue
            yield data
            start_index = stop_index
            pbar.update(self.frames_per_chunk)

        pbar.close()
