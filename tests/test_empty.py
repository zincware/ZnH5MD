import znh5md
import rdkit2ase
import ase
from ase.calculators.singlepoint import SinglePointCalculator

def test_empty(tmp_path):
    atoms = ase.Atoms()

    io = znh5md.IO(tmp_path / 'frames.h5')
    io.append(atoms)
    

def test_empty_info(tmp_path):
    atoms = ase.Atoms()
    atoms.info["test"] = []

    io = znh5md.IO(tmp_path / 'frames.h5')
    io.append(atoms)

def test_empty_array(tmp_path):
    atoms = ase.Atoms()
    atoms.arrays["test"] = []

    io = znh5md.IO(tmp_path / 'frames.h5')
    io.append(atoms)

def test_empty_calc(tmp_path):
    atoms = ase.Atoms()
    atoms.calc = SinglePointCalculator(
        atoms, energy=0.0
    )
    atoms.calc.results["test"] = []
    io = znh5md.IO(tmp_path / 'frames.h5')
    io.append(atoms)
