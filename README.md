[![code-style](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black/)
[![Coverage Status](https://coveralls.io/repos/github/zincware/ZnH5MD/badge.svg?branch=main)](https://coveralls.io/github/zincware/ZnH5MD?branch=main)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zincware/ZnH5MD/HEAD)

# ZnH5MD - High Performance Interface for H5MD Trajectories

ZnH5MD allows easy access to simulation results from H5MD trajectories.

## Example
In the following example we investigate an H5MD dump from LAMMPS with 1000 atoms and 201 configurations:

```python
import znh5md

traj = znh5md.DaskH5MD("file.h5", time_chunk_size=500, species_chunk_size=100)

print(traj.file.time_dependent_groups)
# ['box', 'force', 'image', 'position', 'species', 'velocity']

print(traj.force)
# DaskDataSet(value=dask.array<array, shape=(201, 1000, 3), ...)

print(traj.velocity.slice_by_species(species=1))
# DaskDataSet(value=dask.array<reshape, shape=(201, 500, 3), ...)

print(traj.position.value)
# dask.array<array, shape=(201, 1000, 3), dtype=float64, chunksize=(100, 500, 3), ...>

# You can iterate through the data
for item in traj.position.batch(size=27, axis=0):
    for x in item.batch(size=17, axis=1):
        print(x.compute())
```
