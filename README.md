[![code-style](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black/)
[![Coverage Status](https://coveralls.io/repos/github/zincware/ZnH5MD/badge.svg?branch=main)](https://coveralls.io/github/zincware/ZnH5MD?branch=main)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zincware/ZnH5MD/HEAD)

# ZnH5MD - High Performance Interface for H5MD Trajectories

ZnH5MD allows easy access to simulation results from H5MD trajectories.

## Example
In the following example we investigate an H5MD dump from LAMMPS with 1000 atoms and 201 configurations:

```python
from znh5md import LammpsH5MD

traj = LammpsH5MD("npt.h5")

print(traj.position[:5].shape)
# (5, 1000, 3)

for positions in traj.position.value.get_dataset(batch_size=16):
    print(positions.shape)
    # (16, 1000, 3)

for positions in traj.position.value.get_dataset(selection=slice(500), batch_size=16):
    print(positions.shape)
    # (16, 500, 3)

for positions in traj.position.value.get_dataset(axis=1, batch_size=16):
    print(positions.shape)
    # (201, 16, 3)
```

For a better performance you can add the `prefetch` argument to `get_dataset` to prefetch more data into memory than you would access in a single iteration.
In the following example the data is read up to 3x faster with the `prefetch` argument than without it.

```python

for positions in traj.position.value.get_dataset(batch_size=1, prefetch=16):
    print(positions.shape)
    # (1,  1000, 3)
```