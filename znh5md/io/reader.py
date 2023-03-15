import dataclasses
import typing

import ase
import numpy as np
from ase.calculators.calculator import PropertyNotImplementedError

from znh5md.io.base import DataReader, ExplicitStepTimeChunk


@dataclasses.dataclass
class MockAtomsReader(DataReader):
    atoms: list[ase.Atoms]
    frames_per_chunk: int

    def yield_chunks(
        self, group_name: list = None
    ) -> typing.Iterator[typing.Dict[str, ExplicitStepTimeChunk]]:
        start_index = 0
        stop_index = 0
        if group_name is None:
            group_name = ["position"]

        while stop_index < len(self.atoms):
            stop_index = start_index + self.frames_per_chunk
            data = {}
            for name in group_name:
                if name == "position":
                    value = np.array(
                        [x.get_positions() for x in self.atoms[start_index:stop_index]]
                    )
                elif name == "species":
                    value = np.array(
                        [
                            x.get_atomic_numbers()
                            for x in self.atoms[start_index:stop_index]
                        ]
                    )
                else:
                    raise ValueError(f"Value {name} not supported")
                data[name] = ExplicitStepTimeChunk(
                    value=value,
                    step=np.arange(start_index, start_index + len(value)),
                    time=np.arange(start_index, start_index + len(value)),
                )
            yield data
            start_index = stop_index


@dataclasses.dataclass
class AtomsReader(DataReader):
    atoms: list[ase.Atoms]
    frames_per_chunk: int

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

        while stop_index < len(self.atoms):
            stop_index = start_index + self.frames_per_chunk
            data = {}

            functions = {
                "position": self._get_positions,
                "energy": self._get_energy,
                "species": self._get_species,
                "forces": self._get_forces,
                "stress": self._get_stress,
                "cell": self._get_box,  # TODO this should be saved as box/edges
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
