import znh5md
import ase.build


def test_h5md_time(tmp_path):
    io = znh5md.IO(tmp_path / "test.h5")

    for step in range(1, 10):
        atoms = ase.build.molecule("H2O")
        atoms.info["h5md_step"] = step
        atoms.info["h5md_time"] = step * 0.5
        io.append(atoms)
    
    for idx, atoms in enumerate(io):
        assert atoms.info["h5md_step"] == idx + 1
        assert atoms.info["h5md_time"] == (idx + 1) * 0.5
