import numpy as np
import pytest

from znh5md.core.generators import BatchGenerator


def test_base_loop_shape():
    generator_base = BatchGenerator(
        obj=np.random.rand(100, 50, 3),
        shape=(100, 50, 3),
        axis=0,
        loop_indices=None,
        prefetch=5,
        selection=None,
    )
    assert generator_base.loop_shape == (None, 50, 3)

    generator_base.axis = 1
    assert generator_base.loop_shape == (100, None, 3)
    generator_base.axis = 2
    assert generator_base.loop_shape == (100, 50, None)


def test_batch_selection_loop_shape():
    generator = BatchGenerator(
        obj=np.random.rand(100, 50, 3),
        shape=(100, 50, 3),
        axis=0,
        loop_indices=None,
        prefetch=5,
        selection=[1, 2, 3, 4, 5],
    )

    assert generator.loop_shape == (None, 5, 3)
    generator.axis = 1
    assert generator.loop_shape == (5, None, 3)
    generator.axis = 2
    assert generator.loop_shape == (5, 50, None)


def test_batch_loop():
    generator = BatchGenerator(
        obj=np.random.rand(100, 50, 3),
        shape=(100, 50, 3),
        axis=0,
        loop_indices=None,
        prefetch=5,
        selection=None,
    )

    result = next(iter(generator.loop()))
    assert result.shape == (5, 50, 3)

    generator.axis = 1
    result = next(iter(generator.loop()))
    assert result.shape == (100, 5, 3)

    generator.axis = 2
    generator.prefetch = 1
    result = next(iter(generator.loop()))
    assert result.shape == (100, 50, 1)


def test_batch_loop_selection():
    generator = BatchGenerator(
        obj=np.random.rand(100, 50, 3),
        shape=(100, 50, 3),
        axis=0,
        loop_indices=None,
        prefetch=5,
        selection=[1, 2, 3, 4],
    )

    result = next(iter(generator.loop()))
    assert result.shape == (5, 4, 3)

    generator.axis = 1
    result = next(iter(generator.loop()))
    assert result.shape == (4, 5, 3)


def test_batch_prefetch_size():
    with pytest.raises(ValueError):
        _ = BatchGenerator(
            obj=np.random.rand(100, 50, 3),
            shape=(100, 50, 3),
            axis=2,
            loop_indices=None,
            prefetch=5,
            selection=None,
        )


def test_batch_loop_indices():
    generator = BatchGenerator(
        obj=np.random.rand(100, 50, 3),
        shape=(100, 50, 3),
        axis=0,
        loop_indices=[1, 2, 3],
        prefetch=50,
        selection=None,
    )

    result = next(iter(generator.loop()))
    assert result.shape == (3, 50, 3)

    generator.axis = 1
    result = next(iter(generator.loop()))
    assert result.shape == (100, 3, 3)
