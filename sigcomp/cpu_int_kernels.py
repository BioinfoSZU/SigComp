import numpy as np
from numba import get_num_threads, njit, prange


@njit(inline="always")
def _requantize(accumulator, bias, multiplier, shift, has_relu):
    biased = np.int32(np.int64(accumulator) + np.int64(bias))
    scaled = (np.int64(biased) * multiplier) >> shift
    minimum = 0 if has_relu else -128
    if scaled < minimum:
        scaled = minimum
    if scaled > 127:
        scaled = 127
    return np.int8(scaled)


@njit(cache=True, nogil=True)
def epilogue_serial(accumulator, bias, multiplier, shift, has_relu):
    output = np.empty(accumulator.shape, dtype=np.int8)
    for row in range(accumulator.shape[0]):
        for column in range(accumulator.shape[1]):
            output[row, column] = _requantize(
                accumulator[row, column], bias[column], multiplier, shift, has_relu
            )
    return output


@njit(cache=True, nogil=True, parallel=True)
def epilogue_parallel(accumulator, bias, multiplier, shift, has_relu):
    output = np.empty(accumulator.shape, dtype=np.int8)
    for row in prange(accumulator.shape[0]):
        for column in range(accumulator.shape[1]):
            output[row, column] = _requantize(
                accumulator[row, column], bias[column], multiplier, shift, has_relu
            )
    return output


def _parallel_epilogue(rows, columns, policy, min_work):
    if policy not in {"auto", "serial", "parallel"}:
        raise ValueError("epilogue parallel policy must be auto, serial or parallel")
    if isinstance(min_work, bool) or int(min_work) != min_work or min_work < 1:
        raise ValueError("epilogue min_work must be a positive integer")
    threads = get_num_threads()
    if policy == "serial" or threads <= 1 or rows <= 1:
        return False
    if policy == "parallel":
        return True
    return rows >= threads and rows * columns >= min_work


def choose_epilogue_kernel(rows, columns, policy="auto", min_work=1 << 18):
    return (epilogue_parallel if _parallel_epilogue(rows, columns, policy, min_work)
            else epilogue_serial)


@njit(cache=True, nogil=True)
def epilogue_into_serial(accumulator, bias, multiplier, shift, has_relu, output):
    for row in range(accumulator.shape[0]):
        for column in range(accumulator.shape[1]):
            output[row, column] = _requantize(
                accumulator[row, column], bias[column], multiplier, shift, has_relu
            )


@njit(cache=True, nogil=True, parallel=True)
def epilogue_into_parallel(accumulator, bias, multiplier, shift, has_relu, output):
    for row in prange(accumulator.shape[0]):
        for column in range(accumulator.shape[1]):
            output[row, column] = _requantize(
                accumulator[row, column], bias[column], multiplier, shift, has_relu
            )


def choose_epilogue_into_kernel(rows, columns, policy="auto", min_work=1 << 18):
    return (epilogue_into_parallel
            if _parallel_epilogue(rows, columns, policy, min_work)
            else epilogue_into_serial)


@njit(inline="always")
def _embedding_row(context, weights, output, errors, row):
    errors[row] = 0
    width = weights.shape[1]
    for position in range(context.shape[1]):
        token = context[row, position]
        if token < 0 or token >= weights.shape[0]:
            errors[row] = 1
            continue  # Never perform an unchecked out-of-bounds table lookup.
        offset = position * width
        for column in range(width):
            output[row, offset + column] = weights[token, column]


@njit(cache=True, nogil=True)
def embedding_into_serial(context, weights, output, errors):
    for row in range(context.shape[0]):
        _embedding_row(context, weights, output, errors, row)


@njit(cache=True, nogil=True, parallel=True)
def embedding_into_parallel(context, weights, output, errors):
    for row in prange(context.shape[0]):
        _embedding_row(context, weights, output, errors, row)


@njit(inline="always")
def _embedding_window_row(context, weights, output, errors, row):
    signed_row = np.int64(row)
    stream = signed_row // context.shape[1]
    time = signed_row % context.shape[1]
    errors[row] = 0
    width = weights.shape[1]
    for position in range(context.shape[2]):
        token = context[stream, time, position]
        if token < 0 or token >= weights.shape[0]:
            errors[row] = 1
            continue
        offset = position * width
        for column in range(width):
            output[row, offset + column] = weights[token, column]


@njit(cache=True, nogil=True)
def embedding_windows_into_serial(context, weights, output, errors):
    for row in range(context.shape[0] * context.shape[1]):
        _embedding_window_row(context, weights, output, errors, row)


@njit(cache=True, nogil=True, parallel=True)
def embedding_windows_into_parallel(context, weights, output, errors):
    for row in prange(context.shape[0] * context.shape[1]):
        _embedding_window_row(context, weights, output, errors, row)


def choose_embedding_into_kernel(rows, columns, windows=False):
    parallel = get_num_threads() > 1 and rows >= get_num_threads() and rows * columns >= (1 << 18)
    if windows:
        return embedding_windows_into_parallel if parallel else embedding_windows_into_serial
    return embedding_into_parallel if parallel else embedding_into_serial


@njit(inline="always")
def _linear_cell(x, weight, bias, row, column, multiplier, shift, has_relu):
    accumulator = np.int32(0)
    for k in range(weight.shape[1]):
        product = np.int32(x[row, k]) * np.int32(weight[column, k])
        accumulator = np.int32(accumulator + product)
    accumulator = np.int32(accumulator + bias[column])
    scaled = (np.int64(accumulator) * np.int64(multiplier)) >> shift
    minimum = 0 if has_relu else -128
    if scaled < minimum:
        scaled = minimum
    if scaled > 127:
        scaled = 127
    return np.int8(scaled)


@njit(cache=True, nogil=True)
def linear_serial(x, weight, bias, multiplier, shift, has_relu):
    output = np.empty((x.shape[0], weight.shape[0]), dtype=np.int8)
    for row in range(x.shape[0]):
        for column in range(weight.shape[0]):
            output[row, column] = _linear_cell(
                x, weight, bias, row, column, multiplier, shift, has_relu
            )
    return output


@njit(cache=True, nogil=True, parallel=True)
def linear_parallel(x, weight, bias, multiplier, shift, has_relu):
    output = np.empty((x.shape[0], weight.shape[0]), dtype=np.int8)
    columns = weight.shape[0]
    for index in prange(x.shape[0] * columns):
        signed_index = np.int64(index)
        row = signed_index // columns
        column = signed_index % columns
        output[row, column] = _linear_cell(
            x, weight, bias, row, column, multiplier, shift, has_relu
        )
    return output
