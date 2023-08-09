import ase
import contextlib
from znh5md.format import GRP


class Atoms(ase.Atoms):
    """Extension of ase.Atoms with additional methods for serialization."""
    # def to_dict(self) -> dict:
    #     data = {}

    #     func_map = {
    #         GRP.species: lambda x: x.get_chemical_symbols(),
    #         GRP.position: lambda x: x.get_positions(),
    #         GRP.edges: lambda x: x.get_cell(),
    #         GRP.momentum: lambda x: x.get_momenta(),
    #         GRP.pbc: lambda x: x.get_pbc(),
    #         GRP.energy: lambda x: x.get_potential_energy(),
    #         GRP.forces: lambda x: x.get_forces(),
    #     }

    #     for key, func in func_map.items():
    #         with contextlib.suppress(RuntimeError):
    #             data[key] = func(self)
    #     return data
    
    # @classmethod
    # def from_dict(cls, d: dict):
    #     atoms = cls(
    #         symbols=d[GRP.species],
    #         positions=d[GRP.position],
    #         cell=d[GRP.edges],
    #         momenta=d[GRP.momentum],
    #         pbc=d[GRP.pbc],
    #     )
    #     if GRP.energy in d:
    #         atoms.set_calculator(ase.calculators.singlepoint.SinglePointCalculator(
    #             atoms=atoms,
    #             energy=d[GRP.energy],
    #             forces=d[GRP.forces],
    #         ))
    #     return atoms

