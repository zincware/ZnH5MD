import os

import znh5md
from znh5md.writing import DatabaseWriter, MockAtomsReader


def test_MockAtomsReader(atoms_list):
    reader = MockAtomsReader(atoms_list, frames_per_chunk=10)
    assert len(list(reader.yield_chunks())) == 3


def test_DatabaseWriter(tmp_path, atoms_list):
    os.chdir(tmp_path)

    db = DatabaseWriter(filename="db.h5")
    db.initialize_database_groups()

    reader = MockAtomsReader(atoms_list, frames_per_chunk=10)

    for data in reader.yield_chunks(["position", "species"]):
        db.add_chunk_data(**data)

    data = znh5md.DaskH5MD("db.h5")
    assert data.position.value.shape == (21, 2, 3)
    assert data.species.value.shape == (21, 2)
