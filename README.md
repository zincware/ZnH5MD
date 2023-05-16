[![zincware](https://img.shields.io/badge/Powered%20by-zincware-darkcyan)](https://github.com/zincware)
[![Coverage Status](https://coveralls.io/repos/github/zincware/ZnH5MD/badge.svg?branch=main)](https://coveralls.io/github/zincware/ZnH5MD?branch=main)
[![PyPI version](https://badge.fury.io/py/znh5md.svg)](https://badge.fury.io/py/znh5md)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zincware/ZnH5MD/HEAD)

# ZnH5MD - High Performance Interface for H5MD Trajectories

ZnH5MD allows easy access to simulation results from H5MD trajectories. It
provides a Python interface and can convert existing data to H5MD files as well
as export to other formats.

```
pip install znh5md["dask"]
```

## Example

In the following example we investigate an H5MD dump from LAMMPS with 1000 atoms
and 201 configurations:

```python
import znh5md

traj = znh5md.DaskH5MD("file.h5", time_chunk_size=500, species_chunk_size=100)

print(traj.file.time_dependent_groups)
# ['edges', 'force', 'image', 'position', 'species', 'velocity']

print(traj.force)
# DaskDataSet(value=dask.array<array, shape=(201, 1000, 3), ...)

print(traj.velocity.slice_by_species(species=1))
# DaskDataSet(value=dask.array<reshape, shape=(201, 500, 3), ...)

print(traj.position.value)
# dask.array<array, shape=(201, 1000, 3), dtype=float64, chunksize=(100, 500, 3), ...>

# You can iterate through the data
for item in traj.position.batch(size=27, axis=0):
    for x in item.batch(size=17, axis=1):
        print(x.value.compute())
```

## ASE Atoms

You can use ZnH5MD to store ASE Atoms objects in the H5MD format.

> ZnH5MD does not support all features of ASE Atoms objects. It s important to
> note that unsupported parts are silently ignored and no error is raised.

> The ASEH5MD interface will not provide any time and step information.

> If you have a list of Atoms with different PBC values, you can use
> `znh5md.io.AtomsReader(atoms, use_pbc_group=True)`. This will create a `pbc`
> group in `box/` that also contains `step` and `time`. This is not an official
> H5MD specification so it can cause issues with other tools. If you don't
> specify this, the pbc of the first atoms in the list will be applied.

```python
import znh5md
import ase

atoms: list[ase.Atoms]

db = znh5md.io.DataWriter(filename="db.h5")
db.initialize_database_groups()

db.add(znh5md.io.AtomsReader(atoms)) # or znh5md.io.ChemfilesReader

data = znh5md.ASEH5MD("db.h5")
data.get_atoms_list() == atoms
```

## CLI

ZnH5MD provides a small set of CLI tools:

- `znh5md view <file.h5>` to view the File using `ase.visualize`
- `znh5md export <file.h5> <file.xyz>` to export the file to `.xyz` or any other
  supported file format
- `znh5md convert <file.xyz> <file.h5>` to save a `file.xyz` as `file.h5` in the
  H5MD standard.

## More examples

A complete documentation is still work in progress. In the meantime, I can
recommend looking at the tests, especially `test_znh5md.py` to learn more about
slicing and batching.
