import pathlib

import pytest

CWD = pathlib.Path(__file__).parent
DATA = CWD.parent / "data"


@pytest.fixture
def cu_file():
    return DATA / "cu.h5"
