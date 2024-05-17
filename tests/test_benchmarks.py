import os

import znh5md


def test_bench_DataWriter(tmp_path, atoms_list, benchmark):
    os.chdir(tmp_path)
    db = znh5md.io.DataWriter(filename="db.h5")
    reader = znh5md.io.AtomsReader(atoms_list)
    benchmark(db.add, reader)


def test_bench_ASEH5MD(tmp_path, atoms_list, benchmark):
    os.chdir(tmp_path)
    db = znh5md.io.DataWriter(filename="db.h5")
    reader = znh5md.io.AtomsReader(atoms_list)
    db.add(reader)

    data = znh5md.ASEH5MD("db.h5")
    benchmark(data.get_atoms_list)
