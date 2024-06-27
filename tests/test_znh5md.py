# import ase.collections
# import znh5md
# import numpy.testing as npt

# def test_write_read(tmp_path):
#     structures = list(ase.collections.s22)
#     znh5md.write(tmp_path / 'test.h5', structures)
#     structures2 = znh5md.read(tmp_path / 'test.h5')

#     assert len(structures) == len(structures2)
#     for a, b in zip(structures, structures2):
#         npt.assert_array_equal(a.get_positions(), b.get_positions())
#         npt.assert_array_equal(a.get_atomic_numbers(), b.get_atomic_numbers())
