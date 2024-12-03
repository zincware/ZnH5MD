from znh5md.path import get_h5md_path
from znh5md.serialization import Frames


def test_get_h5md_path(
    s22_all_properties, s22_info_arrays_calc, frames_with_residuenames
):
    frame = Frames.from_ase(
        s22_all_properties + s22_info_arrays_calc + frames_with_residuenames
    )

    assert get_h5md_path("positions", "atoms", frame) == "/particles/atoms/position"
    assert (
        get_h5md_path("energy", "atoms", frame) == "/observables/atoms/potential_energy"
    )
    assert get_h5md_path("forces", "atoms", frame) == "/particles/atoms/force"

    assert get_h5md_path("cell", "atoms", frame) == "/particles/atoms/box/edges"
    assert get_h5md_path("pbc", "atoms", frame) == "/particles/atoms/box/pbc"

    assert (
        get_h5md_path("mlip_forces", "atoms", frame) == "/particles/atoms/mlip_forces"
    )
    assert (
        get_h5md_path("mlip_forces_2", "atoms", frame)
        == "/particles/atoms/mlip_forces_2"
    )
    # assert get_h5md_path("momenta", "atoms", frame) == "/particles/atoms/momenta"
    # assert get_h5md_path("velocity", "atoms", frame) == "/particles/atoms/momenta"
    assert (
        get_h5md_path("residuenames", "atoms", frame) == "/particles/atoms/residuenames"
    )
    assert get_h5md_path("atomtypes", "atoms", frame) == "/particles/atoms/atomtypes"

    assert (
        get_h5md_path("mlip_energy", "atoms", frame) == "/observables/atoms/mlip_energy"
    )
    assert (
        get_h5md_path("mlip_energy_2", "atoms", frame)
        == "/observables/atoms/mlip_energy_2"
    )
    assert (
        get_h5md_path("mlip_stress", "atoms", frame) == "/observables/atoms/mlip_stress"
    )
    assert (
        get_h5md_path("collection", "atoms", frame) == "/observables/atoms/collection"
    )
    assert get_h5md_path("metadata", "atoms", frame) == "/observables/atoms/metadata"

    assert get_h5md_path("energies", "atoms", frame) == "/particles/atoms/energies"
    assert (
        get_h5md_path("free_energy", "atoms", frame) == "/observables/atoms/free_energy"
    )
    assert get_h5md_path("stress", "atoms", frame) == "/observables/atoms/stress"
    assert get_h5md_path("stresses", "atoms", frame) == "/particles/atoms/stresses"
    assert get_h5md_path("dipole", "atoms", frame) == "/observables/atoms/dipole"
    assert get_h5md_path("magmom", "atoms", frame) == "/observables/atoms/magmom"
    assert get_h5md_path("magmoms", "atoms", frame) == "/particles/atoms/magmoms"
    assert (
        get_h5md_path("dielectric_tensor", "atoms", frame)
        == "/observables/atoms/dielectric_tensor"
    )
    assert (
        get_h5md_path("born_effective_charges", "atoms", frame)
        == "/particles/atoms/born_effective_charges"
    )
    assert (
        get_h5md_path("polarization", "atoms", frame)
        == "/observables/atoms/polarization"
    )
