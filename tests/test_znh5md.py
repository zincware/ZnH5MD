import znh5md


def test_version():
    assert znh5md.__version__ == "0.3.6"

def test_file_version(tmp_path):
    io = znh5md.IO(tmp_path / "test.h5")
    assert io.creator == "ZnH5MD"
    assert io.creator_version == znh5md.__version__

