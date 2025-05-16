import time
from dataclasses import asdict, dataclass

import numpy as np

from .abc import IOBase


@dataclass(frozen=True)
class BenchmarkMetrics:
    mean: float
    min: float
    max: float
    std: float

    def asdict(self) -> dict:
        """
        Convert the BenchmarkMetrics to a dictionary.
        """
        return asdict(self)


def benchmark_read(reader_object: IOBase, num_repeats: int = 5) -> BenchmarkMetrics:
    """
    Benchmark the read performance of a given reader object.
    Returns the mean time taken in seconds.
    """
    times = []
    for _ in range(num_repeats):
        start_time = time.time()
        _ = reader_object.read()  # Execute the read operation
        end_time = time.time()
        times.append(end_time - start_time)

    return BenchmarkMetrics(
        mean=np.mean(times), min=np.min(times), max=np.max(times), std=np.std(times)
    )
