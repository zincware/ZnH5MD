import json

import h5py
import numpy as np
import numpy.testing as npt

from znh5md.misc import MISSING
from znh5md.serialization import Entry


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


def test_entry_list_dict():
    e = Entry(value=[{"a": 1}, {"b": 2}], origin=None, name="test")
    assert e.ref == {"a": 1}
    assert e.fillvalue == ""

    val, dtype = e.dump()
    assert isinstance(val, list)
    assert val == [json.dumps(x) for x in [{"a": 1}, {"b": 2}]]
    assert dtype == h5py.string_dtype()


def test_entry_array_dict():
    e = Entry(value=np.array([{"a": 1}, {"b": 2}]), origin=None, name="test")
    assert e.ref == {"a": 1}
    assert e.fillvalue == ""

    val, dtype = e.dump()
    assert isinstance(val, list)
    assert val == [json.dumps(x) for x in [{"a": 1}, {"b": 2}]]
    assert dtype == h5py.string_dtype()


def test_entry_list_dict_missing():
    e = Entry(value=[{"a": 1}, MISSING, {"b": 2}], origin=None, name="test")
    assert e.ref == {"a": 1}
    assert e.fillvalue == ""

    val, dtype = e.dump()
    assert isinstance(val, list)
    assert val == [json.dumps({"a": 1}), "", json.dumps({"b": 2})]
    assert dtype == h5py.string_dtype()


def test_entry_array_dict_missing():
    e = Entry(value=np.array([MISSING, {"a": 1}, {"b": 2}]), origin=None, name="test")
    assert e.ref == {"a": 1}
    assert e.fillvalue == ""

    val, dtype = e.dump()
    assert isinstance(val, list)
    assert val == ["", json.dumps({"a": 1}), json.dumps({"b": 2})]
    assert dtype == h5py.string_dtype()


def test_entry_list_array_int():
    e = Entry(value=[np.array([1, 2]), np.array([3, 4])], origin=None, name="test")
    npt.assert_array_equal(e.ref, np.array([1, 2]))
    assert np.isnan(e.fillvalue)

    val, dtype = e.dump()
    assert isinstance(val, np.ndarray)
    npt.assert_array_equal(val, [[1, 2], [3, 4]])
    assert dtype == np.float64


def test_entry_list_array_int_missing():
    e = Entry(
        value=[np.array([1, 2]), MISSING, np.array([3, 4])], origin=None, name="test"
    )
    npt.assert_array_equal(e.ref, np.array([1, 2]))
    assert np.isnan(e.fillvalue)

    val, dtype = e.dump()
    assert isinstance(val, np.ndarray)
    npt.assert_array_equal(val, [[1, 2], [np.nan, np.nan], [3, 4]])
    assert dtype == np.float64


def test_entry_list_array_str():
    e = Entry(
        value=[np.array(["a", "b"]), np.array(["c", "d"])], origin=None, name="test"
    )
    npt.assert_array_equal(e.ref, np.array(["a", "b"]))
    assert e.fillvalue == ""

    val, dtype = e.dump()
    assert isinstance(val, list)
    assert val == [json.dumps(["a", "b"]), json.dumps(["c", "d"])]
    assert dtype == h5py.string_dtype()


def test_entry_list_array_str_missing():
    e = Entry(
        value=[np.array(["a", "b"]), MISSING, np.array(["c", "d"])],
        origin=None,
        name="test",
    )
    npt.assert_array_equal(e.ref, np.array(["a", "b"]))
    assert e.fillvalue == ""

    val, dtype = e.dump()
    assert isinstance(val, list)
    assert val == [json.dumps(["a", "b"]), "", json.dumps(["c", "d"])]
    assert dtype == h5py.string_dtype()


def test_list_list_str_missing():
    e = Entry(value=[["Cl", "H"], ["O", "O"], MISSING], origin=None, name="test")
    npt.assert_array_equal(e.ref, np.array(["Cl", "H"]))
    assert e.fillvalue == ""

    val, dtype = e.dump()
    assert isinstance(val, list)
    assert val == [json.dumps(["Cl", "H"]), json.dumps(["O", "O"]), ""]
    assert dtype == h5py.string_dtype()


def test_list_list_str_special():
    e = Entry(value=[["Cl", "H"], ["O", "O"], None, ""], origin=None, name="test")
    npt.assert_array_equal(e.ref, np.array(["Cl", "H"]))
    assert e.fillvalue == ""

    val, dtype = e.dump()
    assert isinstance(val, list)
    assert val == [json.dumps(["Cl", "H"]), json.dumps(["O", "O"]), "null", '""']
    assert dtype == h5py.string_dtype()


def test_list_list_ndarray():
    e = Entry(value=[[np.array([1, 2]), np.array([3, 4])]], origin=None, name="test")
    npt.assert_array_equal(e.ref, np.array([np.array([1, 2]), np.array([3, 4])]))
    assert np.isnan(e.fillvalue)

    val, dtype = e.dump()
    assert isinstance(val, np.ndarray)
    npt.assert_array_equal(val[0], [[1, 2], [3, 4]])
    assert dtype == np.float64
