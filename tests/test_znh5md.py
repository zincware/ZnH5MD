import znh5md


def test_creator(tmp_path):
    io = znh5md.IO(tmp_path / "test.h5")
    # These are the defaults
    assert io.creator == "znh5md"
    assert io.creator_version == znh5md.__version__
