from pathlib import Path
import struct

import numpy as np
import torch

from .cpu_profiling import profiled_method, scope


MODEL_DIR = Path(__file__).parent / 'pretrained_models'
DEFAULT_QUANT_LOSSLESS_MODEL_PATH = str(MODEL_DIR / 'lossless.pth')
DEFAULT_QUANT_LOSSY_MODEL_PATH = str(MODEL_DIR / 'lossy.pth')
DEFAULT_QUANT_MODEL_PATH = DEFAULT_QUANT_LOSSLESS_MODEL_PATH
DEFAULT_LSB_SETTING_FOR_LOSSY_MODEL = 3
_CHUNK_LENGTH = struct.Struct('<H')


def _symbols(values):
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError('Each token sequence must be one-dimensional')
    if values.size and (values.dtype.kind not in 'iu'
                        or np.any(values < 0) or np.any(values > 255)):
        raise ValueError('Signal tokens must be integers in [0, 255]')
    return np.ascontiguousarray(values, dtype=np.uint8)


class ReadFramingMixin:

    @profiled_method('read_framing')
    def batch_inference_reads(
            self, list_of_reads=None, list_of_compressed_reads=None,
            list_of_lengths=None, batch_size=64, use_tqdm=False, prob_quant=0):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        chunks, lengths, mapping = [], [], []
        if list_of_reads is not None:
            for read in list_of_reads:
                read = _symbols(read)
                begin = len(chunks)
                chunks.extend(read[i:i + self.chunk_len]
                              for i in range(0, len(read), self.chunk_len))
                mapping.append((begin, len(chunks)))
            encoded = self.compress_chunks(chunks, max_batch_size=batch_size,
                                           use_tqdm=use_tqdm)
            result = []
            for begin, end in mapping:
                framed = bytearray()
                for chunk in encoded[begin:end]:
                    if len(chunk) > 65535:
                        raise ValueError('Compressed chunk exceeds the SIG2 uint16 frame; '
                                         'compress with a smaller --chunk-len')
                    framed.extend(_CHUNK_LENGTH.pack(len(chunk)))
                    framed.extend(chunk)
                result.append(bytes(framed))
            return result

        if list_of_compressed_reads is None or list_of_lengths is None:
            raise ValueError('Compressed reads and token lengths are required')
        if len(list_of_compressed_reads) != len(list_of_lengths):
            raise ValueError('Compressed read and token length counts differ')
        for data, length in zip(list_of_compressed_reads, list_of_lengths):
            if int(length) != length or length < 0:
                raise ValueError('Token lengths must be nonnegative integers')
            remaining, pos, begin = int(length), 0, len(chunks)
            while remaining:
                if pos + _CHUNK_LENGTH.size > len(data):
                    raise ValueError('Truncated SIG2 chunk length')
                size, = _CHUNK_LENGTH.unpack_from(data, pos)
                pos += _CHUNK_LENGTH.size
                if size == 0 or size % 4 or pos + size > len(data):
                    raise ValueError('Invalid or truncated SIG2 range-coded chunk')
                count = min(self.chunk_len, remaining)
                chunks.append(data[pos:pos + size])
                lengths.append(count)
                pos += size
                remaining -= count
            if pos != len(data):
                raise ValueError('Trailing SIG2 chunk data disagrees with token length')
            mapping.append((begin, len(chunks)))
        decoded = self.decompress_chunks(chunks, lengths,
                                         max_batch_size=batch_size, use_tqdm=use_tqdm)
        return [np.concatenate(decoded[begin:end]) if end > begin
                else np.empty(0, dtype=np.uint8) for begin, end in mapping]


class CPUQuantMLPCompression(ReadFramingMixin, torch.nn.Module):
    DEFAULT_DIM = 256
    DEFAULT_NUM_DEC_LAYERS = 8
    DEFAULT_WINDOW_SIZE = 64

    def __init__(self, chunk_len=1024, gpu=None, model_path=None, dim=None,
                 num_dec_layers=None, window_size=None, inference_batch_size=256,
                 inference_backend='auto', *, epilogue='auto',
                 entropy_parallel='auto', entropy_min_streams=256, profiler=None,
                 epilogue_parallel='auto', epilogue_min_work=262144,
                 reuse_buffers=True):
        super().__init__()
        from .cpu_int_inference import load_cpu_decoder
        from .cpu_entropy_coding import build_integer_exp_lut, require_numba

        require_numba()
        if chunk_len < 1 or inference_batch_size < 1:
            raise ValueError('chunk_len and inference_batch_size must be positive')
        self.chunk_len = int(chunk_len)
        self.window_size = window_size or self.DEFAULT_WINDOW_SIZE
        self.inference_batch_size = int(inference_batch_size)
        self.SOS_token = 256
        self.device = 'cpu'
        if entropy_parallel not in ('auto', 'serial', 'parallel'):
            raise ValueError('entropy_parallel must be auto, serial or parallel')
        if entropy_min_streams < 1:
            raise ValueError('entropy_min_streams must be positive')
        self.entropy_parallel = entropy_parallel
        self.entropy_min_streams = int(entropy_min_streams)
        self.profiler = profiler
        self.reuse_buffers = bool(reuse_buffers)
        self.last_workspace_info = None
        self.cpu_engine = load_cpu_decoder(
            model_path or DEFAULT_QUANT_MODEL_PATH,
            dim=dim or self.DEFAULT_DIM,
            num_dec_layers=num_dec_layers or self.DEFAULT_NUM_DEC_LAYERS,
            window_size=self.window_size,
            backend=inference_backend, epilogue=epilogue, profiler=profiler,
            epilogue_parallel=epilogue_parallel, epilogue_min_work=epilogue_min_work,
        )
        self.cpu_engine.eval()
        self._exp_lut = build_integer_exp_lut(float(self.cpu_engine.output_scale))
        if profiler is not None:
            from collections import Counter
            from numba import get_num_threads
            profiler.metadata['codec'] = {
                'chunk_len': self.chunk_len, 'window_size': self.window_size,
                'inference_batch_size': self.inference_batch_size,
                'entropy_parallel': self.entropy_parallel,
                'entropy_min_work': self.entropy_min_streams,
                'torch_threads': torch.get_num_threads(), 'numba_threads': get_num_threads(),
                'inference_backends_at_load': dict(Counter(layer.backend for layer in self.cpu_engine.mlp)),
                'int_mm_epilogues': dict(Counter(layer.epilogue for layer in self.cpu_engine.mlp
                                                if layer.backend == 'int_mm')),
                'epilogue_parallel': epilogue_parallel,
                'epilogue_min_work': epilogue_min_work,
                'reuse_buffers': self.reuse_buffers,
            }

    def set_profiler(self, profiler):
        self.profiler = profiler
        self.cpu_engine.set_profiler(profiler)

    def _create_workspace(self, max_rows, operation):
        with scope(self.profiler, 'inference.workspace_setup'):
            session = self.cpu_engine.create_session(
                max_rows=max_rows, enabled=self.reuse_buffers)
        info = (session.describe() if session is not None else {
            'enabled': False,
            'reason': ('disabled by caller' if not self.reuse_buffers else
                       'workspace requires int_mm layers with Numba epilogues'),
            'max_rows': max_rows, 'buffer_bytes': 0,
        })
        self._record_workspace_info(info, operation, new_session=True)
        return session

    def _record_workspace_info(self, info, operation, *, new_session=False):
        self.last_workspace_info = info
        if self.profiler is not None:
            reports = self.profiler.metadata.setdefault('cpu_workspaces', {})
            previous = reports.get(operation, {})
            reports[operation] = {
                **info, 'sessions': previous.get('sessions', 0) + int(new_session),
                'max_capacity_rows': max(previous.get('max_capacity_rows', 0),
                                         info.get('max_rows', 0)),
                'max_buffer_bytes': max(previous.get('max_buffer_bytes', 0),
                                        info.get('buffer_bytes', 0)),
                'lifetime': 'one compress_chunks/decompress_chunks invocation',
                'descriptor_scope': 'last session; max_capacity_rows/max_buffer_bytes cover all sessions',
            }

    @torch.inference_mode()
    @profiled_method('token_compression')
    def compress_chunks(self, chunks, max_batch_size=64, use_perfect=False, use_tqdm=False):
        from .cpu_entropy_coding import CpuRangeStepEncoder

        if max_batch_size < 1:
            raise ValueError('max_batch_size must be positive')
        self.last_workspace_info = {
            'enabled': False, 'reason': 'no nonempty chunks processed',
            'max_rows': 0, 'buffer_bytes': 0,
        }
        batch_size = min(max_batch_size, self.inference_batch_size)
        starts = range(0, len(chunks), batch_size)
        if use_tqdm:
            from tqdm import tqdm
            starts = tqdm(starts, desc='Compressing chunks (CPU)')
        output = []
        profiler = self.profiler
        session = None
        session_initialized = False
        for start in starts:
            with scope(profiler, 'prepare'):
                batch = [_symbols(c) for c in chunks[start:start + batch_size]]
                valid = np.asarray([len(c) for c in batch], dtype=np.int64)
                if np.any(valid > self.chunk_len):
                    raise ValueError('A token chunk exceeds chunk_len')
                size, max_len = len(batch), int(valid.max())
                if profiler is not None:
                    profiler.add_count('tokens', valid.sum())
                    profiler.add_count('chunks', size)
                    profiler.add_count('mlp_rows', size * max_len)
            if not max_len:
                output.extend([b''] * size)
                continue
            if not session_initialized:
                session = self._create_workspace(
                    min(self.inference_batch_size,
                        min(batch_size, len(chunks)) * self.chunk_len), 'compression')
                session_initialized = True
            with scope(profiler, 'prepare'):
                if session is not None:
                    context = np.full((size, self.window_size + max_len),
                                      self.SOS_token, dtype=np.int64)
                    symbols = context[:, self.window_size:]
                    symbols.fill(0)
                    for b, chunk in enumerate(batch):
                        symbols[b, :len(chunk)] = chunk
                    windows = np.lib.stride_tricks.sliding_window_view(
                        context, self.window_size, axis=1)
                else:
                    symbols = np.zeros((size, max_len), dtype=np.int64)
                    for b, chunk in enumerate(batch):
                        symbols[b, :len(chunk)] = chunk
                    context = torch.full((size, self.window_size + max_len),
                                         self.SOS_token, dtype=torch.int64)
                    context[:, self.window_size:] = torch.from_numpy(symbols)
                    windows = context.unfold(1, self.window_size, 1)
                encoder = CpuRangeStepEncoder(size, self.chunk_len + 32,
                                              parallel=self.entropy_parallel,
                                              parallel_min_streams=self.entropy_min_streams)
            time_batch = max(1, self.inference_batch_size // size)
            for t in range(0, max_len, time_batch):
                count = min(time_batch, max_len - t)
                with scope(profiler, 'context_windows'):
                    contexts = windows[:, t:t + count]
                    if session is None:
                        contexts = contexts.reshape(-1, self.window_size)
                with scope(profiler, 'inference'):
                    logits = (session.step_windows(contexts) if session is not None
                              else self.cpu_engine._ar_step(contexts))
                    logits = logits.reshape(size, count, 256)
                with scope(profiler, 'cdf_range_encode'):
                    encoder.encode_logits(logits.numpy(), self._exp_lut,
                                          symbols[:, t:t + count], valid, t_start=t)
            with scope(profiler, 'entropy_finalize'):
                encoder.seal()
                output.extend(encoder.get_compressed())
        if session is not None:
            self._record_workspace_info(session.describe(), 'compression')
        return output

    @torch.inference_mode()
    @profiled_method('token_decompression')
    def decompress_chunks(self, compressed_bytes_list, valid_lengths,
                          max_batch_size=64, use_perfect=False, use_tqdm=False):
        from .cpu_entropy_coding import CpuRangeDecoder

        if max_batch_size < 1:
            raise ValueError('max_batch_size must be positive')
        self.last_workspace_info = {
            'enabled': False, 'reason': 'no nonempty chunks processed',
            'max_rows': 0, 'buffer_bytes': 0,
        }
        if len(compressed_bytes_list) != len(valid_lengths):
            raise ValueError('Compressed chunk and length counts differ')
        if any(int(n) != n or n < 0 or n > self.chunk_len for n in valid_lengths):
            raise ValueError('Chunk lengths must be integers in [0, chunk_len]')
        starts = range(0, len(compressed_bytes_list), max_batch_size)
        if use_tqdm:
            from tqdm import tqdm
            starts = tqdm(starts, desc='Decompressing chunks (CPU)')
        output = []
        profiler = self.profiler
        session = None
        session_initialized = False
        for start in starts:
            with scope(profiler, 'prepare'):
                batch = compressed_bytes_list[start:start + max_batch_size]
                valid = np.asarray(valid_lengths[start:start + max_batch_size], dtype=np.int64)
                size, max_len = len(batch), int(valid.max())
                for data, length in zip(batch, valid):
                    if (length and (not data or len(data) % 4)) or (not length and data):
                        raise ValueError('Invalid range-coded chunk for its token length')
                if profiler is not None:
                    profiler.add_count('tokens', valid.sum())
                    profiler.add_count('chunks', size)
                    profiler.add_count('mlp_rows', size * max_len)
            if not max_len:
                output.extend(np.empty(0, dtype=np.uint8) for _ in batch)
                continue
            if not session_initialized:
                session = self._create_workspace(
                    min(max_batch_size, len(compressed_bytes_list)), 'decompression')
                session_initialized = True
            with scope(profiler, 'prepare'):
                decoder = CpuRangeDecoder(batch, parallel=self.entropy_parallel,
                                          parallel_min_streams=self.entropy_min_streams)
                if session is not None:
                    context = np.full((size, self.window_size + max_len),
                                      self.SOS_token, dtype=np.int64)
                else:
                    context = torch.full((size, self.window_size + max_len),
                                         self.SOS_token, dtype=torch.int64)
            # Decode is autoregressive in time, but independent chunks share each
            # integer GEMM and one compiled range-decoder call.
            for t in range(max_len):
                with scope(profiler, 'inference'):
                    contexts = context[:, t:t + self.window_size]
                    logits = (session.step(contexts) if session is not None
                              else self.cpu_engine._ar_step(contexts))
                with scope(profiler, 'cdf_range_decode'):
                    symbols = decoder.decode_step_int8(logits[:, 0].numpy(), self._exp_lut,
                                                       valid, t)
                with scope(profiler, 'context_update'):
                    context[:, t + self.window_size] = (symbols if session is not None
                                                        else torch.from_numpy(symbols))
            with scope(profiler, 'output_conversion'):
                decoded = context[:, self.window_size:]
                if session is None:
                    decoded = decoded.numpy()
                output.extend(decoded[b, :length].astype(np.uint8)
                              for b, length in enumerate(valid))
        if session is not None:
            self._record_workspace_info(session.describe(), 'decompression')
        return output
