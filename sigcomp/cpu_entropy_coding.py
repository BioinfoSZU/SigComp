from decimal import Decimal, localcontext
import math

import numpy as np


PRECISION = 24
VOCAB_SIZE = 256
_TOTAL = 1 << PRECISION
_U32_MASK = np.uint64(0xFFFFFFFF)
_U64_MAX = np.uint64(0xFFFFFFFFFFFFFFFF)
_WORD_BITS = np.uint64(32)
_PRECISION_U64 = np.uint64(PRECISION)
_RENORM_LIMIT = np.uint64(1 << 32)

_NUMBA_IMPORT_ERROR = None
try:
    from numba import get_num_threads, njit, prange
    from numba.extending import register_jitable
except Exception as exc:
    _NUMBA_IMPORT_ERROR = exc
    njit = None
    prange = range

    def get_num_threads():
        return 1

    def register_jitable(function):
        return function


NUMBA_AVAILABLE = njit is not None


def require_numba():
    if not NUMBA_AVAILABLE:
        raise ImportError(
            "CPU entropy coding requires a working Numba installation. "
            "Install compatible versions with: pip install 'numba>=0.59' "
            "'numpy>=1.26,<2'. The Python reference backend is available only "
            "when explicitly requested with backend='python'."
        ) from _NUMBA_IMPORT_ERROR


def _as_array(value, dtype):
    return np.ascontiguousarray(np.asarray(value), dtype=dtype)


def build_integer_exp_lut(output_scale, lut_precision_bits=38):
    scale = float(output_scale)
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("output_scale must be a finite nonnegative number")
    if not isinstance(lut_precision_bits, int) or not 0 <= lut_precision_bits <= 38:
        raise ValueError("lut_precision_bits must be an integer in [0, 38]")
    with localcontext() as context:
        context.prec = 50
        scale_dec = Decimal(str(scale))
        multiplier = Decimal(1 << lut_precision_bits)
        values = [max(int((Decimal(-i) * scale_dec).exp() * multiplier), 0)
                  for i in range(VOCAB_SIZE)]
    return np.asarray(values, dtype=np.int64)


def _check_lut(exp_lut):
    lut = _as_array(exp_lut, np.int64)
    if lut.shape != (VOCAB_SIZE,):
        raise ValueError("exp_lut must have shape (256,)")
    if lut[0] <= 0 or np.any(lut < 0) or np.any(lut > (1 << 38)):
        raise ValueError("exp_lut must be nonnegative, at most 2**38, and start positive")
    return lut


def int8_logits_to_cdf(int8_logits, exp_lut, precision=PRECISION):
    if precision != PRECISION:
        raise ValueError("the SigComp stream format requires 24-bit precision")
    logits = _as_array(int8_logits, np.int8)
    if logits.ndim != 2 or logits.shape[1] != VOCAB_SIZE:
        raise ValueError("int8_logits must have shape (batch, 256)")
    lut = _check_lut(exp_lut)
    shifted = logits.max(axis=1, keepdims=True).astype(np.int16) - logits.astype(np.int16)
    unnorm = lut[shifted]
    freqs = np.maximum((unnorm * _TOTAL) // unnorm.sum(axis=1, keepdims=True), 1)
    correction = freqs.sum(axis=1) - _TOTAL
    freqs[np.arange(len(freqs)), freqs.argmax(axis=1)] -= correction
    result = np.zeros((len(logits), VOCAB_SIZE + 1), dtype=np.int64)
    result[:, 1:] = np.cumsum(freqs, axis=1)
    return result


@register_jitable
def _frequencies(logits, lut, weights):
    maximum = -128
    for v in range(VOCAB_SIZE):
        maximum = max(maximum, int(logits[v]))
    total = np.int64(0)
    for v in range(VOCAB_SIZE):
        weight = lut[maximum - int(logits[v])]
        weights[v] = weight
        total += weight
    total_freq = np.int64(0)
    max_freq = np.int64(0)
    max_index = 0
    for v in range(VOCAB_SIZE):
        freq = max((weights[v] * _TOTAL) // total, 1)
        weights[v] = freq
        total_freq += freq
        if freq > max_freq:
            max_freq = freq
            max_index = v
    weights[max_index] -= total_freq - _TOTAL


@register_jitable
def _emit(output, position, word):
    if position >= len(output):
        raise OverflowError("range-coder output capacity exceeded; increase max_words")
    output[position] = np.uint32(word & _U32_MASK)
    return position + 1


@register_jitable
def _encode_one(logits, lut, symbol, lower, rnge, inverted, first, output, position, freqs):
    _frequencies(logits, lut, freqs)
    cumulative = np.uint64(0)
    for v in range(symbol):
        cumulative += np.uint64(freqs[v])
    scale = rnge >> _PRECISION_U64
    new_range = scale * np.uint64(freqs[symbol])
    new_lower = lower + scale * cumulative
    if inverted > 0 and new_lower + new_range > new_lower:
        if new_lower < lower:
            emit_first = np.uint64(first) + np.uint64(1)
            emit_rest = np.uint64(0)
        else:
            emit_first = np.uint64(first)
            emit_rest = _U32_MASK
        if position + inverted > len(output):
            return lower, rnge, inverted, first, position, 1
        output[position] = np.uint32(emit_first & _U32_MASK)
        position += 1
        for _ in range(inverted - 1):
            output[position] = np.uint32(emit_rest)
            position += 1
        inverted = 0
    lower = new_lower
    rnge = new_range
    if rnge < _RENORM_LIMIT:
        rnge = rnge << _WORD_BITS
        lower_word = lower >> _WORD_BITS
        lower = lower << _WORD_BITS
        if inverted > 0:
            inverted += 1
        elif lower + rnge > lower:
            if position >= len(output):
                return lower, rnge, inverted, first, position, 1
            output[position] = np.uint32(lower_word)
            position += 1
        else:
            inverted = 1
            first = np.uint32(lower_word)
    return lower, rnge, inverted, first, position, 0


def _encode_step(logits, lut, symbols, valid_lens, t,
                 lowers, ranges, inverted, first, output, positions, scratch, errors):
    freqs = scratch[0]
    for b in range(len(lowers)):
        errors[b] = 0
        if t < valid_lens[b]:
            values = _encode_one(logits[b], lut, int(symbols[b]), lowers[b],
                                 ranges[b], inverted[b], first[b], output[b], positions[b], freqs)
            lowers[b], ranges[b], inverted[b], first[b], positions[b], errors[b] = values


@register_jitable
def _encode_many(logits, lut, symbols, valid_lens, t_start,
                 lowers, ranges, inverted, first, output, positions, scratch, errors, partitions):
    for partition in prange(partitions):
        part = np.int64(partition)
        freqs = scratch[part]
        start = len(lowers) * part // partitions
        stop = len(lowers) * (part + 1) // partitions
        for b in range(start, stop):
            lower, rnge = lowers[b], ranges[b]
            count, word, position = inverted[b], first[b], positions[b]
            errors[b] = 0
            for t in range(logits.shape[1]):
                if t_start + t >= valid_lens[b]:
                    break
                lower, rnge, count, word, position, status = _encode_one(
                    logits[b, t], lut, int(symbols[b, t]), lower, rnge,
                    count, word, output[b], position, freqs)
                if status:
                    errors[b] = status
                    break
            lowers[b], ranges[b] = lower, rnge
            inverted[b], first[b], positions[b] = count, word, position


def _encode_many_serial(logits, lut, symbols, valid_lens, t_start,
                        lowers, ranges, inverted, first, output, positions, scratch, errors):
    _encode_many(logits, lut, symbols, valid_lens, t_start,
                 lowers, ranges, inverted, first, output, positions, scratch, errors, 1)


def _seal(lowers, ranges, inverted, first, output, positions):
    for b in range(len(lowers)):
        lower, rnge = lowers[b], ranges[b]
        position = positions[b]
        if rnge != _U64_MAX:
            point = lower + _U32_MASK
            if inverted[b] > 0:
                if point < lower:
                    emit_first = np.uint64(first[b]) + np.uint64(1)
                    emit_rest = np.uint64(0)
                else:
                    emit_first = np.uint64(first[b])
                    emit_rest = _U32_MASK
                position = _emit(output[b], position, emit_first)
                for _ in range(inverted[b] - 1):
                    position = _emit(output[b], position, emit_rest)
            point_word = point >> _WORD_BITS
            position = _emit(output[b], position, point_word)
            upper_word = (lower + rnge) >> _WORD_BITS
            if upper_word == point_word:
                position = _emit(output[b], position, np.uint64(0))
        positions[b] = position


@register_jitable
def _decode_partition(logits, lut, valid_lens, t, lowers, ranges, points,
                      words, read_positions, end_positions, symbols, freqs, errors, start, stop):
    for b in range(start, stop):
        errors[b] = 0
        if t >= valid_lens[b]:
            symbols[b] = VOCAB_SIZE
            continue
        lower, rnge, point = lowers[b], ranges[b], points[b]
        scale = rnge >> _PRECISION_U64
        if scale == 0:
            errors[b] = 1
            symbols[b] = VOCAB_SIZE
            continue
        quantile = (point - lower) // scale
        if quantile >= np.uint64(_TOTAL):
            errors[b] = 1
            symbols[b] = VOCAB_SIZE
            continue
        _frequencies(logits[b], lut, freqs)
        cumulative = np.uint64(0)
        symbol = 0
        while symbol < VOCAB_SIZE - 1 and cumulative + np.uint64(freqs[symbol]) <= quantile:
            cumulative += np.uint64(freqs[symbol])
            symbol += 1
        lower += scale * cumulative
        rnge = scale * np.uint64(freqs[symbol])
        if rnge < _RENORM_LIMIT:
            lower = lower << _WORD_BITS
            rnge = rnge << _WORD_BITS
            point = point << _WORD_BITS
            position = read_positions[b]
            if position < end_positions[b]:
                point = point | np.uint64(words[position])
            read_positions[b] = position + 1
        lowers[b], ranges[b], points[b] = lower, rnge, point
        symbols[b] = symbol


def _decode_step(logits, lut, valid_lens, t, lowers, ranges, points,
                 words, read_positions, end_positions, symbols, scratch, errors):
    _decode_partition(logits, lut, valid_lens, t, lowers, ranges, points,
                      words, read_positions, end_positions, symbols, scratch[0], errors,
                      0, len(lowers))


def _decode_step_parallel(logits, lut, valid_lens, t, lowers, ranges, points,
                          words, read_positions, end_positions, symbols, scratch, errors,
                          partitions):
    for partition in prange(partitions):
        part = np.int64(partition)
        start = len(lowers) * part // partitions
        stop = len(lowers) * (part + 1) // partitions
        _decode_partition(logits, lut, valid_lens, t, lowers, ranges, points,
                          words, read_positions, end_positions, symbols,
                          scratch[part], errors, start, stop)


if NUMBA_AVAILABLE:
    _encode_step_native = njit(cache=True, nogil=True)(_encode_step)
    _encode_many_native = njit(cache=True, nogil=True, parallel=True)(_encode_many)
    _encode_many_serial_native = njit(cache=True, nogil=True)(_encode_many_serial)
    _seal_native = njit(cache=True, nogil=True)(_seal)
    _decode_step_native = njit(cache=True, nogil=True)(_decode_step)
    _decode_step_parallel_native = njit(cache=True, nogil=True, parallel=True)(_decode_step_parallel)


class _CpuRangeBase:
    def _set_backend(self, device, backend):
        if str(device) != "cpu":
            raise ValueError("CPU range coders only accept device='cpu'")
        if backend not in ("numba", "python"):
            raise ValueError("backend must be 'numba' or 'python'")
        if backend == "numba":
            require_numba()
        self.device = "cpu"
        self.backend = backend

    def _init_workspace(self):
        self._scratch = np.empty((1, VOCAB_SIZE), dtype=np.int64)
        self._errors = np.zeros(self.B, dtype=np.uint8)
        self._failed = False

    def _set_parallel(self, parallel, parallel_min_streams):
        if parallel not in ("auto", "serial", "parallel"):
            raise ValueError("parallel must be 'auto', 'serial', or 'parallel'")
        if (isinstance(parallel_min_streams, bool)
                or not isinstance(parallel_min_streams, (int, np.integer))
                or parallel_min_streams < 1):
            raise ValueError("parallel_min_streams must be a positive integer")
        self.parallel = parallel
        self.parallel_min_streams = int(parallel_min_streams)

    def _use_parallel(self, work_items=None):
        work = self.B if work_items is None else work_items
        return (self.backend == "numba" and self.B > 1 and get_num_threads() > 1
                and (self.parallel == "parallel"
                     or (self.parallel == "auto" and work >= self.parallel_min_streams)))

    def _workspace(self, parallel):
        partitions = min(self.B, get_num_threads()) if parallel else 1
        partitions = max(partitions, 1)
        if len(self._scratch) < partitions:
            self._scratch = np.empty((partitions, VOCAB_SIZE), dtype=np.int64)
        return partitions

    def _check_failed(self):
        if self._failed:
            raise ValueError("range coder is unusable after a native coding error; create a new instance")

    def _check_errors(self, encoding=False):
        if self._errors.any():
            self._failed = True
            stream = int(np.flatnonzero(self._errors)[0])
            if encoding:
                raise OverflowError(
                    f"range-coder output capacity exceeded in stream {stream}; increase max_words")
            raise ValueError(f"invalid range stream or mismatching entropy model in stream {stream}")

    def _inputs(self, logits, exp_lut, valid_lens, t, ndim=2):
        logits = _as_array(logits, np.int8)
        expected = (self.B, VOCAB_SIZE)
        if ndim == 2 and logits.shape != expected:
            raise ValueError("int8_logits must have shape (batch, 256)")
        if ndim == 3 and (logits.ndim != 3 or logits.shape[0] != self.B
                          or logits.shape[2] != VOCAB_SIZE):
            raise ValueError("int8_logits must have shape (batch, time, 256)")
        valid_lens = _as_array(valid_lens, np.int64)
        if valid_lens.shape != (self.B,) or np.any(valid_lens < 0):
            raise ValueError("valid_lens must contain one nonnegative length per stream")
        if t < 0:
            raise ValueError("t must be nonnegative")
        return logits, _check_lut(exp_lut), valid_lens


class CpuRangeStepEncoder(_CpuRangeBase):

    def __init__(self, B, max_words, device="cpu", backend="numba", *,
                 parallel="auto", parallel_min_streams=256):
        self._set_backend(device, backend)
        self._set_parallel(parallel, parallel_min_streams)
        if B < 0 or max_words < 2:
            raise ValueError("B must be nonnegative and max_words must be at least two")
        self.B = int(B)
        self.max_words = int(max_words)
        self.state_lower = np.zeros(self.B, dtype=np.uint64)
        self.state_range = np.full(self.B, _U64_MAX, dtype=np.uint64)
        self.num_inverted = np.zeros(self.B, dtype=np.int64)
        self.first_inv_word = np.zeros(self.B, dtype=np.uint32)
        self.output_words = np.empty((self.B, self.max_words), dtype=np.uint32)
        self.write_pos = np.zeros(self.B, dtype=np.int64)
        self._init_workspace()
        self._sealed = False

    def _state(self):
        return (self.state_lower, self.state_range, self.num_inverted,
                self.first_inv_word, self.output_words, self.write_pos)

    def encode_step_int8(self, int8_logits, exp_lut, symbols, valid_lens_tensor, t):
        self._check_failed()
        if self._sealed:
            raise ValueError("cannot encode after seal()")
        logits, lut, lengths = self._inputs(int8_logits, exp_lut, valid_lens_tensor, t)
        symbols = _as_array(symbols, np.int64)
        if symbols.shape != (self.B,):
            raise ValueError("symbols must have shape (batch,)")
        active = t < lengths
        if np.any(active & ((symbols < 0) | (symbols >= VOCAB_SIZE))):
            raise ValueError("active symbols must be in [0, 255]")
        parallel = self._use_parallel()
        partitions = self._workspace(parallel)
        with np.errstate(over="ignore"):
            if parallel:
                _encode_many_native(logits[:, None, :], lut, symbols[:, None], lengths, int(t),
                                    *self._state(), self._scratch, self._errors, partitions)
            else:
                kernel = _encode_step_native if self.backend == "numba" else _encode_step
                kernel(logits, lut, symbols, lengths, int(t), *self._state(), self._scratch, self._errors)
        self._check_errors(encoding=True)

    def encode_logits(self, int8_logits, exp_lut, symbols, valid_lens, t_start=0):
        self._check_failed()
        if self._sealed:
            raise ValueError("cannot encode after seal()")
        logits, lut, lengths = self._inputs(int8_logits, exp_lut, valid_lens, t_start, ndim=3)
        symbols = _as_array(symbols, np.int64)
        if symbols.shape != logits.shape[:2]:
            raise ValueError("symbols must have shape (batch, time)")
        active = np.arange(logits.shape[1])[None, :] + t_start < lengths[:, None]
        if np.any(active & ((symbols < 0) | (symbols >= VOCAB_SIZE))):
            raise ValueError("active symbols must be in [0, 255]")
        parallel = self._use_parallel(work_items=self.B * logits.shape[1])
        partitions = self._workspace(parallel)
        with np.errstate(over="ignore"):
            if parallel:
                _encode_many_native(logits, lut, symbols, lengths, int(t_start), *self._state(),
                                    self._scratch, self._errors, partitions)
            else:
                kernel = _encode_many_serial_native if self.backend == "numba" else _encode_many_serial
                kernel(logits, lut, symbols, lengths, int(t_start), *self._state(),
                       self._scratch, self._errors)
        self._check_errors(encoding=True)

    def seal(self):
        self._check_failed()
        if not self._sealed:
            kernel = _seal_native if self.backend == "numba" else _seal
            try:
                with np.errstate(over="ignore"):
                    kernel(*self._state())
            except OverflowError:
                self._failed = True
                raise
            self._sealed = True

    def get_compressed(self):
        self.seal()
        return [self.output_words[b, :self.write_pos[b]].astype("<u4", copy=False).tobytes()
                for b in range(self.B)]


class CpuRangeDecoder(_CpuRangeBase):

    def __init__(self, compressed_bytes_list, device="cpu", backend="numba", *,
                 parallel="auto", parallel_min_streams=256):
        self._set_backend(device, backend)
        self._set_parallel(parallel, parallel_min_streams)
        self.B = len(compressed_bytes_list)
        self._init_workspace()
        self.state_lower = np.zeros(self.B, dtype=np.uint64)
        self.state_range = np.full(self.B, _U64_MAX, dtype=np.uint64)
        self.point = np.zeros(self.B, dtype=np.uint64)
        self.read_pos = np.zeros(self.B, dtype=np.int64)
        self.end_pos = np.zeros(self.B, dtype=np.int64)
        self.symbols_out = np.full(self.B, VOCAB_SIZE, dtype=np.int64)
        streams = []
        offset = 0
        for b, data in enumerate(compressed_bytes_list):
            if len(data) % 4:
                raise ValueError("compressed range stream length must be a multiple of four bytes")
            words = np.frombuffer(data, dtype="<u4").astype(np.uint32, copy=False)
            point = 0
            for i in range(min(2, len(words))):
                point = (point << 32) | int(words[i])
            if len(words) == 1:
                point <<= 32
            self.point[b] = point
            self.read_pos[b] = offset + min(2, len(words))
            offset += len(words)
            self.end_pos[b] = offset
            streams.append(words)
        self.comp_words = np.concatenate(streams) if streams else np.empty(0, dtype=np.uint32)

    def decode_step_int8(self, int8_logits, exp_lut, valid_lens_tensor, t):
        self._check_failed()
        logits, lut, lengths = self._inputs(int8_logits, exp_lut, valid_lens_tensor, t)
        parallel = self._use_parallel()
        partitions = self._workspace(parallel)
        state = (self.state_lower, self.state_range, self.point, self.comp_words,
                 self.read_pos, self.end_pos, self.symbols_out, self._scratch, self._errors)
        with np.errstate(over="ignore"):
            if parallel:
                _decode_step_parallel_native(logits, lut, lengths, int(t), *state, partitions)
            else:
                kernel = _decode_step_native if self.backend == "numba" else _decode_step
                kernel(logits, lut, lengths, int(t), *state)
        self._check_errors()
        return self.symbols_out
