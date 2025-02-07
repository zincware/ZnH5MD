import warnings

import ase.build

import znh5md

# def test_h5md_time(tmp_path):
#     io = znh5md.IO(tmp_path / "test_time_step.h5", store="time")
#     for step in range(1, 10):
#         atoms = ase.build.molecule("H2O")
#         atoms.calc = SinglePointCalculator(atoms, energy=step * 0.1)
#         atoms.info["h5md_step"] = step
#         atoms.info["h5md_time"] = step * 0.5
#         io.append(atoms)

#     for idx, atoms in enumerate(io[:]):
#         assert atoms.info["h5md_step"] == idx + 1
#         assert atoms.info["h5md_time"] == (idx + 1) * 0.5
#         assert atoms.get_potential_energy() == (idx + 1) * 0.1

#     with h5py.File(tmp_path / "test_time_step.h5") as f:
#         npt.assert_array_equal(
#             f["particles/atoms/position/time"][:],
#             np.arange(1, 10) * 0.5
#         )
#         npt.assert_array_equal(f["particles/atoms/position/step"][:],
#                                 np.arange(1, 10))
#         npt.assert_array_equal(
#             f["observables/atoms/energy/time"][:],
#               np.arange(1, 10) * 0.5
#         )
#         npt.assert_array_equal(f["observables/atoms/energy/step"][:],
#                                np.arange(1, 10))
#         npt.assert_array_equal(
#             f["observables/atoms/energy/value"][:],
#               np.arange(1, 10) * 0.1
#         )


# def test_inconsistent_time(tmp_path):
#     images = [ase.build.molecule("H2O") for _ in range(10)]
#     images[5].info["h5md_time"] = 0.5

#     io = znh5md.IO(tmp_path / "test_inconsistent_time.h5", store="time")
#     with pytest.raises(ValueError):
#         io.extend(images)


# def test_inconsistent_step(tmp_path):
#     images = [ase.build.molecule("H2O") for _ in range(10)]
#     images[5].info["h5md_step"] = 5

#     io = znh5md.IO(tmp_path / "test_inconsistent_step.h5", store="time")
#     with pytest.raises(ValueError):
#         io.extend(images)


# def test_wrong_store(tmp_path):
#     io = znh5md.IO(tmp_path / "test_wrong_store.h5", store="linear")
#     atoms = ase.build.molecule("H2O")
#     atoms.info["h5md_step"] = 1
#     atoms.info["h5md_time"] = 0.1

#     with pytest.warns(UserWarning, match="time is ignored in 'linear' storage mode"):
#         io.append(atoms)


def test_no_warn_correct(tmp_path):
    io = znh5md.IO(tmp_path / "test_wrong_store.h5", store="linear")
    atoms = ase.build.molecule("H2O")

    # Ensure no warning is issued
    # https://docs.pytest.org/en/latest/how-to/capture-warnings.html#additional-use-cases-of-warnings-in-tests
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        io.append(atoms)
