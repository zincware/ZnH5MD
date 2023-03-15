import dataclasses
import typing

import ase
import numpy as np

from znh5md.writing.base import DataReader, ExplicitStepTimeChunk


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
