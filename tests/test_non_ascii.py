import znh5md


def test_info_non_ascii(tmp_path, s22_no_ascii):
    io = znh5md.IO(tmp_path / "test.h5")
    io.extend(s22_no_ascii)

    for a, b in zip(io, s22_no_ascii):
        assert a.info["config"] == b.info["config"] == "βγ"
