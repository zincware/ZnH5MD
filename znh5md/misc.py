import numpy as np

def concatenate_varying_shape_arrays(values: list, fillvalue: str|int|float|bool) -> np.ndarray:
    dtype = np.array(fillvalue).dtype
    dimensions = list(values[0].shape)
    for value in values[1:]:
        for idx, (a, b) in enumerate(zip(dimensions, value.shape)):
            if b > a:
                dimensions[idx] = b
    
    dataset = np.full((len(values), *dimensions), fillvalue, dtype=dtype)
    print(dataset.shape)
    for i, value in enumerate(values):
        if len(value.shape) == 1:
            dataset[i, :value.shape[0]] = value
        elif len(value.shape) == 2:
            dataset[i, :value.shape[0], :value.shape[1]] = value
        elif len(value.shape) == 3:
            dataset[i, :value.shape[0], :value.shape[1], :value.shape[2]] = value
        elif len(value.shape) == 4:
            dataset[i, :value.shape[0], :value.shape[1], :value.shape[2], :value.shape[3]] = value
        else:
            raise ValueError(f"Unsupported shape: {value.shape}")
    return dataset