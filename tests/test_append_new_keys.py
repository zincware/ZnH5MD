import znh5md


def test_append_data_with_new_info(tmp_path, s22, s22_energy):
    io = znh5md.IO(tmp_path / "test.h5")
    io.extend(s22)
    io.extend(s22_energy)
    assert len(s22) == 22
    assert len(s22_energy) == 22
    assert len(io) == 44

    # for a, b in zip(io, s22 + s22_energy):
    #     assert a.calc == b.calc

    for a, b in zip(io[: len(s22)], s22):
        assert a.calc is None
        assert b.calc is None
    
    for a, b in zip(io[len(s22) :], s22_energy):
        assert a.calc is not None
        assert b.calc is not None
        assert a.calc.results == b.calc.results

    