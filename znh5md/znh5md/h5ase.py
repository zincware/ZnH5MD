import contextlib
import dataclasses
import typing

import ase
import h5py
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

from znh5md.format import GRP, OBSERVABLES_GRP, PARTICLES_GRP
from znh5md.utils import rm_nan
from znh5md.znh5md.base import H5MDBase


def _gather_value(particles_data, key, idx):
    """Helper to gather the value for a given key and index.

    Returns None if the key is not present in the data.
    """
    try:
        if key in particles_data:
            if key in [GRP.species, GRP.position, GRP.velocity, GRP.forces, GRP.momentum]:
                # use PARTICLES_GRP
                return rm_nan(particles_data[key][idx])
            return particles_data[key][idx]
    except IndexError:
        # a property might not be available at all frames.
        pass
    return None


@dataclasses.dataclass
class ASEH5MD(H5MDBase):
    """ASE interface for H5MD files.

    Attributes
    ----------
    load_all_observables: bool, default=True
        if True, all observables are loaded into 'atoms.calc.results'.
        Otherwise, only the standard ASE observables are loaded.
    """

    load_all_observables: bool = True

    def __getitem__(self, item):
        return self.get_atoms_list(item=item)

    def __getattr__(self, item) -> h5py.Dataset:
        """Get the h5py.Dataset for a given item.

        We only return the value here, because the time and step
        information is not used in the ASE interface.
        """
        # not using dask here was 8x faster on a 32 MB h5 file
        if item == GRP.boundary:
            return getattr(self.format_handler, item)
        return getattr(self.format_handler, item)["value"]

    def _get_particles_dict(self, item) -> dict:
        """Get particles group data."""
        data = {}
        for key in PARTICLES_GRP:
            with contextlib.suppress(AttributeError, KeyError):
                if key == GRP.boundary:
                    data[key] = getattr(self, key)[:]
                else:
                    data[key] = (
                        getattr(self, key)[item] if item else getattr(self, key)[:]
                    )

        if GRP.boundary in data and GRP.pbc not in data:
            data[GRP.pbc] = np.repeat(
                [GRP.decode_boundary(data[GRP.boundary])], len(data[GRP.position]), axis=0
            )

        return data

    def _get_observables_dict(self, item, particles_data: dict) -> dict:
        """Get observables group data."""
        observables = {}

        if self.load_all_observables:
            # TODO this can be removed in future versions
            groups = list(set(self.format_handler.observables_groups + OBSERVABLES_GRP))
        else:
            groups = OBSERVABLES_GRP

        for group_name in groups:
            if group_name not in particles_data:
                with contextlib.suppress(AttributeError, KeyError):
                    # TODO this can be removed in future versions
                    observables[group_name] = (
                        getattr(self, group_name)[item]
                        if item
                        else getattr(self, group_name)[:]
                    )
        return observables

    def get_atoms_list(self, item=None) -> typing.List[ase.Atoms]:
        """Get an 'ase.Atoms' list for all data."""
        single_item = isinstance(item, int)
        if single_item:
            item = [item]

        particles_data = self._get_particles_dict(item=item)
        observables_data = self._get_observables_dict(
            item=item, particles_data=particles_data
        )

        atoms = []

        for idx in range(len(particles_data[GRP.position])):
            obj = ase.Atoms(
                symbols=_gather_value(particles_data, GRP.species, idx),
                positions=_gather_value(particles_data, GRP.position, idx),
                momenta=_gather_value(particles_data, GRP.momentum, idx),
                cell=_gather_value(particles_data, GRP.edges, idx),
                pbc=_gather_value(particles_data, GRP.pbc, idx),
            )
            if GRP.forces in particles_data or len(observables_data):
                obj.calc = SinglePointCalculator(
                    obj, forces=_gather_value(particles_data, GRP.forces, idx)
                )
                for key in observables_data:
                    obj.calc.results[key] = observables_data[key][idx]

            atoms.append(obj)

        return atoms[0] if single_item else atoms
