import typing as t
from enum import Enum

if t.TYPE_CHECKING:
    from znh5md.serialization import Frames


class AttributePath(str, Enum):
    origin = "ASE_ENTRY_ORIGIN"
    unit = "unit"
    boundary = "boundary"
    dimension = "dimension"


class H5MDPath(str, Enum):
    # https://www.nongnu.org/h5md/h5md.html#particles-group
    position = "/particles/{}/position"
    velocity = "/particles/{}/velocity"
    force = "/particles/{}/force"
    mass = "/particles/{}/mass"
    species = "/particles/{}/species"
    id = "/particles/{}/id"
    charge = "/particles/{}/charge"

    # https://www.nongnu.org/h5md/modules/thermodynamics.html
    pressure = "/observables/{}/pressure"
    temperature = "/observables/{}/temperature"
    density = "/observables/{}/density"
    potential_energy = "/observables/{}/potential_energy"
    kinetic_energy = "/observables/{}/kinetic_energy"
    internal_energy = "/observables/{}/internal_energy"
    enthalpy = "/observables/{}/enthalpy"

    # cell
    box_edges = "/particles/{}/box/edges"
    box_pbc = "/particles/{}/box/pbc"


class H5MDToASEMapping(str, Enum):
    position = "positions"
    force = "forces"
    mass = "masses"
    charge = "charges"
    potential_energy = "energy"
    box_edges = "cell"
    box_pbc = "pbc"
    species = "numbers"
    velocity = "velocities"


def get_h5md_path(name: str, particles_group: str, frames: "Frames") -> str:
    try:
        h5md_name = H5MDToASEMapping(name).name
    except ValueError:
        h5md_name = name
    try:
        return H5MDPath[h5md_name].format(particles_group)
    except KeyError:
        if name in frames.arrays.keys():
            return "/particles/{}/{}".format(particles_group, name)
        elif name in frames.info.keys():
            return "/observables/{}/{}".format(particles_group, name)
        elif name in frames.calc.keys():
            in_particles = False
            try:
                in_particles = (
                    frames.calc[name][0].shape[0] == frames.positions[0].shape[0]
                )
                # it could be a coincidence that the shape of the property is the same
                #  as the shape of the positions
                # e.g for 3, or 6 particles!
            except (AttributeError, IndexError):
                # e.g. object has no attribute 'shape'
                # or tuple index out of range
                pass

            if in_particles:
                return "/particles/{}/{}".format(particles_group, name)
            else:
                return "/observables/{}/{}".format(particles_group, name)
        else:
            raise ValueError(f"Unable to determine path for '{name}'")
