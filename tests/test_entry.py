import json

import h5py
import numpy as np
import numpy.testing as npt

from znh5md.serialization import MISSING, Entry


def test_entry_list_int():
    e = Entry(value=[1, 2, 3], origin=None, name="test")
    assert e.ref == 1
    assert np.isnan(e.fillvalue)
    val, dtype = e.dump()
    assert isinstance(val, np.ndarray)
    npt.assert_array_equal(val, [1, 2, 3])
    assert dtype == np.float64


def test_entry_array_int():
    e = Entry(value=np.array([1, 2, 3]), origin=None, name="test")
    assert e.ref == 1
    assert np.isnan(e.fillvalue)
    val, dtype = e.dump()
    assert isinstance(val, np.ndarray)
    npt.assert_array_equal(val, [1, 2, 3])
    assert val.dtype == np.float64
    assert dtype == np.float64


def test_entry_list_int_missing():
    e = Entry(value=[1, MISSING, 3], origin=None, name="test")
    assert e.ref == 1
    assert np.isnan(e.fillvalue)
    val, dtype = e.dump()
    assert dtype == np.float64
    assert isinstance(val, np.ndarray)
    npt.assert_array_equal(val, [1, np.nan, 3])


def test_entry_array_int_missing():
    e = Entry(value=np.array([1, MISSING, 3]), origin=None, name="test")
    assert e.ref == 1
    assert np.isnan(e.fillvalue)
    val, dtype = e.dump()
    assert dtype == np.float64
    assert isinstance(val, np.ndarray)
    npt.assert_array_equal(val, [1, np.nan, 3])


def test_entry_int_missing_first():
    e = Entry(value=np.array([MISSING, 2]), origin=None, name="test")
    assert e.ref == 2
    assert np.isnan(e.fillvalue)
    val, dtype = e.dump()
    assert dtype == np.float64
    assert isinstance(val, np.ndarray)
    npt.assert_array_equal(val, [np.nan, 2])


def test_entry_list_str():
    e = Entry(value=["a", "b", "c"], origin=None, name="test")
    assert e.ref == "a"
    assert e.fillvalue == ""

    val, dtype = e.dump()
    assert isinstance(val, list)
    assert val == [json.dumps(x) for x in ["a", "b", "c"]]
    assert dtype == h5py.string_dtype()


def test_entry_array_str():
    e = Entry(value=np.array(["a", "b", "c"]), origin=None, name="test")
    assert e.ref == "a"
    assert e.fillvalue == ""

    val, dtype = e.dump()
    assert isinstance(val, list)
    assert val == [json.dumps(x) for x in ["a", "b", "c"]]
    assert dtype == h5py.string_dtype()


def test_entry_list_str_missing():
    e = Entry(value=["a", MISSING, "c"], origin=None, name="test")
    assert e.ref == "a"
    assert e.fillvalue == ""

    val, dtype = e.dump()
    assert isinstance(val, list)
    assert val == [json.dumps("a"), "", json.dumps("c")]
    assert dtype == h5py.string_dtype()


def test_entry_array_str_missing():
    e = Entry(value=np.array(["a", MISSING, "c"]), origin=None, name="test")
    assert e.ref == "a"
    assert e.fillvalue == ""

    val, dtype = e.dump()
    assert isinstance(val, list)
    assert val == [json.dumps("a"), "", json.dumps("c")]
    assert dtype == h5py.string_dtype()
