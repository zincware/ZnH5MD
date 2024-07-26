
import ase.build

import znh5md


def test_h5md_time(tmp_path):
    # tmp_path = pathlib.Path("/Users/fzills/tools/ZnH5MD/tmp")
    io = znh5md.IO(tmp_path / "test_time_step.h5", store="time")
    # TODO: also add observables!
    # TODO: test inconsistent time and step
    for step in range(1, 10):
        atoms = ase.build.molecule("H2O")
        atoms.info["h5md_step"] = step
        atoms.info["h5md_time"] = step * 0.5
        io.append(atoms)

    for idx, atoms in enumerate(io[:]):
        assert atoms.info["h5md_step"] == idx + 1
        assert atoms.info["h5md_time"] == (idx + 1) * 0.5
