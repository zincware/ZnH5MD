from dataclasses import dataclass

import ase


@dataclass(kw_only=True)
class IOBase:
    filename: str
    format: str
    num_atoms: int
    num_frames: int

    def setup(self):
        pass

    def read(self) -> list[ase.Atoms]: ...

    def write(self, atoms: list[ase.Atoms]) -> None: ...
