[![zincware](https://img.shields.io/badge/Powered%20by-zincware-darkcyan)](https://github.com/zincware)
[![Coverage Status](https://coveralls.io/repos/github/zincware/ZnH5MD/badge.svg?branch=main)](https://coveralls.io/github/zincware/ZnH5MD?branch=main)
[![PyPI version](https://badge.fury.io/py/znh5md.svg)](https://badge.fury.io/py/znh5md)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zincware/ZnH5MD/HEAD)
[![H5MD](https://img.shields.io/badge/format-H5MD-darkmagenta)](https://www.nongnu.org/h5md/)

# ZnH5MD - ASE Interface for the H5MD format.

ZnH5MD provides and interface from \[ASE\] to \[H5MD\] and vice versa. Install
via `pip install znh5md`. Similar to ASE ZnH5MD provides `read` and `write`
functionality:

```python
import znh5md
from ase.collections import s22

znh5md.write("s22.h5", list(s22))
print(list(znh5md.iread("s22.h5")))
# list[ase.Atoms]
```

Further, you can access any data from within the entire dataset through the
`znh5md.IO` class which provides a `MutableSequence`-like interface.

```python
import znh5md
from ase.collections import s22

io = znh5md.IO("s22.h5", particle_group="s22")
io.extend(list(s22))

print(io[5:10])
# list[ase.Atoms]
```

## Extended H5MD Format

ZnH5MD circumvents two current limitations of the H5MD standard.

- support `images` with varying particle counts by padding the dataset with
  `np.nan`. Using varying species counts might break the compatibility with
  other H5MD tools.
- support varying `pbc` within a single particle group by introducing
  `particles/<group>/box/pbc/value` in addition to the `particles/<group>/box`
  attributes. By default, this is enabled via `IO(pbc_group=True)`. The
  `particles/<group>/box` attribute will be set to the PBC conditions of the
  first frame. Using this feature will not typically not break ompatibility with
  other H5MD tools but can lead to unexpected behaviour.

## Current limitations

This is a not necessarily complete list of Limitations that will be fixed
eventually. Any contributions are welcome.

- Step: ZnH5MD assumes a fixed time interval of 1.
- Units: There is no automatic unit conversion through e.g. the pint package
- performance tweaks: there are many places in ZnH5MD that can be optimized for
  better performance. Currently most of the values are hard-coded, such as chunk
  size. Nevertheless, ZnH5MD outperforms most other packages w.r.t. read and
  write speed.
