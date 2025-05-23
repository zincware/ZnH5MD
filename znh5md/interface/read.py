import json
import typing as t

import ase
import numpy as np
from ase.calculators.calculator import all_properties

from znh5md.misc import MISSING, decompose_varying_shape_arrays, open_file
from znh5md.path import AttributePath, H5MDToASEMapping
from znh5md.serialization import ORIGIN_TYPE, Frames

if t.TYPE_CHECKING:
    from znh5md.interface.io import IO


def update_frames(
    self: Frames,
    name: str,
    value: np.ndarray,
    origin: ORIGIN_TYPE,
    use_ase_calc: bool,
    variable_shape: bool,
) -> None:
    """
    Updates the specified frame data with appropriate transformations and storage logic.

    Parameters
    ----------
    name : str
        The name of the attribute to update.
    value : np.ndarray
        The array containing the new values to be set.
    origin : ORIGIN_TYPE
        The origin of the data (e.g., "calc", "info", "arrays").
    use_ase_calc : bool
        Whether to use ASE calculator for storing the data.
    variable_shape : bool
        Whether the input data has a variable shape.
    """
    if name in ["positions", "numbers", "pbc", "cell"]:
        if variable_shape:
            setattr(self, name, decompose_varying_shape_arrays(value, np.nan))
        else:
            setattr(self, name, value)
        return

    data = preprocess_data(value, variable_shape)
    store_data(self, name, data, origin, use_ase_calc)


def preprocess_data(value: np.ndarray, variable_shape: bool) -> list:
    """
    Preprocess the input data by handling object, string, or numeric types.

    Parameters
    ----------
    value : np.ndarray
        The array containing the input data.
    variable_shape : bool
        Whether the input data has a variable shape.

    Returns
    -------
    list
        The processed list of data.
    """
    if value.dtype.kind in ["O", "S", "U"]:
        try:
            return [json.loads(v) if v != b"" else MISSING for v in value]
        except json.JSONDecodeError:
            # compatibility for non-JSON data from other sources
            return [v.decode() if v != b"" else MISSING for v in value]
    else:
        if variable_shape:
            return decompose_varying_shape_arrays(value, np.nan)
        else:
            return value


def store_data(
    self: Frames, name: str, data: list, origin: ORIGIN_TYPE, use_ase_calc: bool
) -> None:
    """
    Store processed data into the appropriate attribute based on origin and conditions.

    Parameters
    ----------
    name : str
        The name of the property to store.
    data : list
        The processed data to store.
    origin : ORIGIN_TYPE
        The origin of the data (e.g., "calc", "info", "arrays").
    use_ase_calc : bool
        Whether to use ASE calculator for storing the data.

    Raises
    ------
    ValueError
        If the origin is invalid or disallowed.
    """
    if origin is not None and use_ase_calc:
        handle_origin_data(self, name, data, origin)
    else:
        assign_data_to_property(self, name, data, use_ase_calc)


def handle_origin_data(
    self: Frames, name: str, data: list, origin: ORIGIN_TYPE
) -> None:
    """
    Handle data storage based on the specified origin.

    Parameters
    ----------
    name : str
        The name of the property to store.
    data : list
        The processed data to store.
    origin : ORIGIN_TYPE
        The origin of the data (e.g., "calc", "info", "arrays").

    Raises
    ------
    ValueError
        If the origin is invalid or disallowed.
    """
    if origin == "calc":
        self.calc[name] = data
    elif origin == "info":
        self.info[name] = data
    elif origin == "arrays":
        try:
            self.arrays[name] = np.array(data)
        except ValueError:
            # Try individual arrays
            self.arrays[name] = [np.array(d) for d in data]
    elif origin == "atoms":
        raise ValueError(f"Origin 'atoms' is not allowed for {name}")
    else:
        raise ValueError(f"Unknown origin: {origin}")


def assign_data_to_property(
    self: Frames, name: str, data: list, use_ase_calc: bool
) -> None:
    """
    Assign data to the appropriate property based on its size and conditions.

    Parameters
    ----------
    name : str
        The name of the property to store.
    data : list
        The processed data to store.
    use_ase_calc : bool
        Whether to use ASE calculator for storing the data.
    """
    if name in all_properties:
        if use_ase_calc:
            self.calc[name] = data
        else:
            if isinstance(data[0], t.Sized) and len(data[0]) == len(self.numbers[0]):
                self.arrays[name] = data
            else:
                self.info[name] = data
    else:
        if isinstance(data[0], t.Sized) and len(data[0]) == len(self.numbers[0]):
            self.arrays[name] = data
        else:
            self.info[name] = data


def getitem(
    self: "IO", index: int | np.int_ | slice | np.ndarray | list[int]
) -> ase.Atoms | list[ase.Atoms]:
    """
    Retrieve frames of atoms based on the given index.

    Parameters
    ----------
    self : IO
        The IO object containing file information and settings.
    index : int, np.int_, slice, np.ndarray, or list[int]
        Index or indices specifying the frames to retrieve.

    Returns
    -------
    ase.Atoms or list[ase.Atoms]
        A single `ase.Atoms` object or a list of `ase.Atoms` objects,
        depending on the index.
    """
    frames = Frames()
    is_single_item = isinstance(index, int)
    if is_single_item:
        index = [index]

    with open_file(self.filename, self.file_handle, mode="r") as f:
        particles = f[f"/particles/{self.particles_group}"]
        process_species_group(self, frames, particles, index)
        process_particle_groups(self, frames, particles, index)

        if f"/observables/{self.particles_group}" in f:
            observables = f[f"/observables/{self.particles_group}"]
            process_observables(self, frames, observables, index)
    return list(frames) if not is_single_item else frames[0]


def process_species_group(self, frames: Frames, particles, index) -> None:
    """
    Process the 'species' group and update the frames.

    Parameters
    ----------
    self : IO
        The IO object containing file information and settings.
    frames : Frames
        The frames to update.
    particles : h5py.Group
        The particles group from the HDF5 file.
    index : list[int]
        Indices specifying the frames to retrieve.
    """
    grp = particles["species/value"]
    update_frames(
        frames,
        H5MDToASEMapping.species.value,
        grp[index],
        None,
        self.use_ase_calc,
        variable_shape=self.variable_shape,
    )


def process_particle_groups(self: "IO", frames: Frames, particles, index) -> None:
    """
    Process particle groups other than 'species' and update the frames.

    Parameters
    ----------
    self : IO
        The IO object containing file information and settings.
    frames : Frames
        The frames to update.
    particles : h5py.Group
        The particles group from the HDF5 file.
    index : list[int]
        Indices specifying the frames to retrieve.
    """
    for grp_name in particles:
        if grp_name == "species":
            continue
        if self.include is not None and grp_name not in self.include:
            continue
        try:
            grp = particles[grp_name]
            origin = grp.attrs.get(AttributePath.origin.value, None)
            if grp_name == "box":
                process_box_group(self, frames, grp, index, origin)
            else:
                process_generic_group(self, frames, grp_name, grp, index, origin)
        except Exception as err:
            raise ValueError(f"Error processing group '{grp_name}'") from err


def process_box_group(self, frames: Frames, grp, index, origin) -> None:
    """
    Process the 'box' group and update the frames.

    Parameters
    ----------
    self : IO
        The IO object containing file information and settings.
    frames : Frames
        The frames to update.
    grp : h5py.Group
        The 'box' group from the HDF5 file.
    index : list[int]
        Indices specifying the frames to retrieve.
    origin : str or None
        The origin attribute of the group.
    """
    update_frames(
        frames,
        "cell",
        grp["edges/value"][index],
        origin,
        self.use_ase_calc,
        variable_shape=self.variable_shape,
    )
    try:
        update_frames(
            frames,
            "pbc",
            grp["pbc/value"][index],
            origin,
            self.use_ase_calc,
            variable_shape=self.variable_shape,
        )
    except KeyError:
        pbc = grp.attrs.get(AttributePath.boundary.value, ["none"] * 3)
        pbc = np.array([b != "none" for b in pbc], dtype=bool)
        frames.pbc = np.array([pbc] * len(frames))


def process_generic_group(
    self, frames: Frames, grp_name: str, grp, index, origin
) -> None:
    """
    Process generic particle groups and update the frames.

    Parameters
    ----------
    self : IO
        The IO object containing file information and settings.
    frames : Frames
        The frames to update.
    grp_name : str
        The name of the group.
    grp : h5py.Group
        The particle group from the HDF5 file.
    index : list[int]
        Indices specifying the frames to retrieve.
    origin : str or None
        The origin attribute of the group.

    Raises
    ------
    KeyError
        If the group does not contain a valid 'value' dataset.
    """
    try:
        try:
            update_frames(
                frames,
                H5MDToASEMapping[grp_name].value,
                grp["value"][index],
                origin,
                self.use_ase_calc,
                variable_shape=self.variable_shape,
            )
        except KeyError:
            update_frames(
                frames,
                grp_name,
                grp["value"][index],
                origin,
                self.use_ase_calc,
                variable_shape=self.variable_shape,
            )
        except (OSError, IndexError):
            pass  # Handle backfilling for invalid values
    except KeyError:
        raise KeyError(
            f"Key '{grp_name}' does not seem to be a valid H5MD group"
            " - missing 'value' dataset."
        )


def process_observables(self: "IO", frames: Frames, observables, index) -> None:
    """
    Process observables and update the frames.

    Parameters
    ----------
    self : IO
        The IO object containing file information and settings.
    frames : Frames
        The frames to update.
    observables : h5py.Group
        The observables group from the HDF5 file.
    index : list[int]
        Indices specifying the frames to retrieve.
    """
    for grp_name in observables:
        if self.include is not None and grp_name not in self.include:
            continue
        grp = observables[grp_name]
        origin = grp.attrs.get(AttributePath.origin.value, None)
        try:
            try:
                try:
                    update_frames(
                        frames,
                        H5MDToASEMapping[grp_name].value,
                        grp["value"][index],
                        origin,
                        self.use_ase_calc,
                        variable_shape=self.variable_shape,
                    )
                except KeyError:
                    update_frames(
                        frames,
                        grp_name,
                        grp["value"][index],
                        origin,
                        self.use_ase_calc,
                        variable_shape=self.variable_shape,
                    )
            except (OSError, IndexError):
                pass  # Handle backfilling for invalid values
        except KeyError:
            raise KeyError(
                f"Key '{grp_name}' does not seem to be a valid H5MD group"
                " - missing 'value' dataset."
            )
        except Exception as err:
            raise ValueError(f"Error processing group '{grp_name}'") from err
