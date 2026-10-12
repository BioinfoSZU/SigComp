import math
import struct
import pickle
import os
import sys
import uuid
import datetime
import argparse
import statistics
import tempfile
import warnings
from pathlib import Path
from contextlib import ExitStack
from array import array

# Apply socket affinity and memory policy before native libraries allocate
# workspaces or start thread pools. Console-entry-point imports restart once
# through this module in main(), so they get the same early initialization.
_SOCKET_BINDING = None
if __name__ == '__main__':
    from .cpu_socket import bootstrap_socket
    _SOCKET_BINDING = bootstrap_socket(sys.argv[1:])

warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import torch
import pod5
import numcodecs
import fast_rice

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeRemainingColumn, MofNCompleteColumn, TimeElapsedColumn, ProgressColumn
)
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from .cpu_compression import (
    CPUQuantMLPCompression,
    DEFAULT_QUANT_MODEL_PATH,
    DEFAULT_QUANT_LOSSLESS_MODEL_PATH,
    DEFAULT_QUANT_LOSSY_MODEL_PATH,
    DEFAULT_LSB_SETTING_FOR_LOSSY_MODEL,
    ReadFramingMixin,
)
from .cpu_profiling import (
    CPUProfiler, current_profiler, profiled_stage, profile_iterable,
    print_profile_summary, scope,
)


MAGIC = b'SIG2'
FORMAT_VERSION = 2
IDX_MAGIC = b'SIDX'
REC_FMT = struct.Struct('<16sddBBdIIIQ')       # POD5 per-read record
REC_FMT_BLOW5 = struct.Struct('<16s')          # BLOW5 per-read record (read_id only)
SPLIT_HDR_FMT = struct.Struct('<hII')
U32 = struct.Struct('<I')
U64 = struct.Struct('<Q')
MB_UNIT = 1000000

_BLOW5_EXTS = {'.blow5', '.slow5'}


def _import_pyslow5():
    try:
        import pyslow5
        return pyslow5
    except ImportError:
        raise ImportError(
            "pyslow5 is required for BLOW5/SLOW5 support. "
            "Install with: pip install pyslow5"
        )


def _prepare_s5_header(src_header):
    if not isinstance(src_header, dict):
        return {}
    return {k: v for k, v in src_header.items() if v is not None}


@profiled_stage('output.write_blow5')
def _write_blow5_record(s5_writer, meta, raw_sig):
    record, aux = s5_writer.get_empty_record(aux=True)
    for k, v in meta.items():
        if k in record:
            record[k] = v
        if k in aux:
            aux[k] = v
    record['signal'] = raw_sig.astype(np.int16)
    record['len_raw_signal'] = len(raw_sig)
    return s5_writer.write_record(record, aux)


def _detect_format(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in _BLOW5_EXTS:
        return 'blow5'
    elif ext == '.pod5':
        return 'pod5'
    elif ext == '.bin':
        return 'bin'
    return None


def _use_pod5_reopen(args):
    return (_detect_format(args.input) == 'pod5'
            and getattr(args, 'pod5_reader_mode', 'reopen') == 'reopen')


class VRAMColumn(ProgressColumn):
    def __init__(self, device: int):
        self.device = device
        super().__init__()

    def render(self, task) -> Text:
        if self.device is None:
            return Text("CPU", style="dim cyan")
        try:
            mem_mb = torch.cuda.memory_allocated(self.device) / MB_UNIT
            return Text(f"VRAM: {mem_mb:.0f} MB", style="dim cyan")
        except Exception:
            return Text("VRAM: N/A", style="dim cyan")


def _diff_encode(signal: np.ndarray) -> np.ndarray:
    return np.diff(signal).astype(np.int16)


def _diff_decode(sig0: int, deltas: np.ndarray) -> np.ndarray:
    signal = np.empty(len(deltas) + 1, dtype=np.int16)
    signal[0] = sig0
    signal[1:] = sig0 + np.cumsum(deltas).astype(np.int16)
    return signal


def _zigzag_encode(values: np.ndarray) -> np.ndarray:
    v = values.astype(np.int32)
    return ((v << 1) ^ (v >> 31)).astype(np.uint16)


def _zigzag_decode(values: np.ndarray) -> np.ndarray:
    v = values.astype(np.uint32)
    return ((v >> 1) ^ -(v & 1)).astype(np.int16)


def _delta_zigzag_encode(signal: np.ndarray) -> np.ndarray:
    return _zigzag_encode(_diff_encode(signal))


def _delta_zigzag_decode(sig0: int, array: np.ndarray) -> np.ndarray:
    return _diff_decode(sig0, _zigzag_decode(array))


def _reduce_lsb(signal: np.ndarray, bit: int) -> np.ndarray:
    v = signal.astype(np.int32)
    step = 1 << bit
    rounding = ((v >> (bit - 1)) & 1) * step
    upper = v & (~(step - 1))
    return (upper + rounding).astype(np.int16)


def _lossy_encode(signal: np.ndarray, bit: int) -> np.ndarray:
    return (signal.astype(np.int32) >> bit).astype(np.int16)


def _lossy_decode(signal: np.ndarray, bit: int) -> np.ndarray:
    return (signal.astype(np.int32) << bit).astype(np.int16)


@profiled_stage('rice_encode')
def _rice_compress(vals, fallback_bits=32, max_quotient=31) -> bytes:
    if len(vals) == 0:
        return struct.pack('<B', 0)
    vals_u32 = np.ascontiguousarray(vals, dtype=np.uint32)
    return bytes(fast_rice.rice_encode(vals_u32, fallback_bits=fallback_bits, max_quotient=max_quotient))


@profiled_stage('rice_decode')
def _rice_decompress(buf, count, fallback_bits=32, max_quotient=31) -> np.ndarray:
    if count == 0:
        return np.array([], dtype=np.uint32)
    return fast_rice.rice_decode(bytes(buf), count, fallback_bits=fallback_bits, max_quotient=max_quotient)


@profiled_stage('model_load')
def _load_model(
        model_path: str, gpu: int, chunk_len: int,
        dim: int = None, num_dec_layers: int = None, window_size: int = None,  # type: ignore
        device: str = 'cpu', cpu_inference_batch_size: int = 4096,
        cpu_backend: str = 'int_mm', cpu_epilogue: str = 'numba',
        cpu_entropy_parallel: str = 'parallel', cpu_entropy_min_streams: int = 256,
        cpu_epilogue_parallel: str = 'parallel', cpu_epilogue_min_work: int = 262144,
        cpu_workspace: str = 'auto',
):
    profiler = current_profiler()
    if profiler is not None:
        profiler.protect_path(model_path)
    if device == 'cpu':
        return CPUQuantMLPCompression(
            model_path=model_path, chunk_len=chunk_len, dim=dim,
            num_dec_layers=num_dec_layers, window_size=window_size,
            inference_batch_size=cpu_inference_batch_size,
            inference_backend=cpu_backend,
            epilogue=cpu_epilogue, entropy_parallel=cpu_entropy_parallel,
            entropy_min_streams=cpu_entropy_min_streams, profiler=current_profiler(),
            epilogue_parallel=cpu_epilogue_parallel,
            epilogue_min_work=cpu_epilogue_min_work, reuse_buffers=cpu_workspace == 'auto',
        )
    # Import Triton only when a GPU is actually requested.
    from .quant_mlp_export import QuantMLPCompression
    class GPUCompression(ReadFramingMixin, QuantMLPCompression):
        """Reuse the original kernels with validated, empty-read-safe framing."""

    return GPUCompression(
        model_path=model_path,
        chunk_len=chunk_len,
        gpu=gpu,
        dim=dim,
        num_dec_layers=num_dec_layers,
        window_size=window_size,
    )


@profiled_stage('output.header')
def _write_header(f, global_dict):
    b_globals = pickle.dumps(global_dict, protocol=pickle.HIGHEST_PROTOCOL)
    b_globals_zstd = numcodecs.zstd.compress(b_globals, level=22)
    f.write(MAGIC)
    f.write(U32.pack(len(b_globals_zstd)))
    f.write(b_globals_zstd)


@profiled_stage('input.header')
def _read_header(f):
    magic = f.read(4)
    if magic != MAGIC:
        raise ValueError(
            f"Not a SIGCOMP file (magic={magic!r}). "
        )
    (hdr_len,) = U32.unpack(f.read(4))
    hdr_zstd = f.read(hdr_len)
    global_dict = pickle.loads(numcodecs.zstd.decompress(hdr_zstd))
    return global_dict, f.tell()


def _make_pore_key(pore_obj):
    return tuple(sorted(vars(pore_obj).items()))


def _read_ids_file(path):
    """Read UUIDs, one per nonempty line; duplicates are always errors."""
    result = []
    seen = set()
    with open(path, encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            value = line.strip()
            if not value or value.startswith('#'):
                continue
            try:
                read_id = str(uuid.UUID(value))
            except ValueError as exc:
                raise ValueError(f'{path}:{line_number}: expected one read UUID per line') from exc
            if read_id in seen:
                raise ValueError(f'{path}:{line_number}: duplicate read UUID {read_id}')
            seen.add(read_id)
            result.append(read_id)
    if not result:
        raise ValueError(f'No read UUIDs in {path}')
    return result


def _source_identity(path):
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {'path': str(resolved), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def _close_source(source):
    close = getattr(source, 'close', None)
    if close is not None:
        close()
    else:
        source.clear_readers()


@profiled_stage('input.open')
def _open_source_reads(path, source_format, read_ids=None):
    if read_ids is not None and not os.path.isfile(path):
        raise ValueError('--read-ids-file requires one source file, not a directory')
    if source_format == 'blow5':
        source = _import_pyslow5().Open(path, 'r')
        if read_ids is None:
            return source, source.seq_reads(aux='all')

        def selected_reads():
            for read_id, read in zip(read_ids, source.get_read_list(read_ids, aux='all')):
                if read is None:
                    raise ValueError(f'Read UUID not found in source: {read_id}')
                if str(uuid.UUID(read['read_id'])) != read_id:
                    raise ValueError(f'BLOW5 index returned the wrong read for {read_id}')
                yield read

        return source, selected_reads()
    if read_ids is not None:
        source = pod5.Reader(path)
        return source, source.reads(selection=read_ids, missing_ok=False)
    source = pod5.DatasetReader(path)
    return source, source.reads()


def collect_source_metadata(input_path, max_reads=-1, *, include_sample_counts=False):
    input_path = os.fspath(input_path)
    if not os.path.isfile(input_path):
        raise ValueError('Sharded compression requires one source file')
    source_format = _detect_format(input_path)
    if source_format not in ('pod5', 'blow5'):
        raise ValueError('Sharded compression requires .pod5, .blow5 or .slow5 input')
    metadata = {'source_format': source_format, 'pore_dict': {}, 'run_info_dict': {},
                'source_identity': _source_identity(input_path)}
    read_ids = []
    sample_counts = array('Q') if include_sample_counts and source_format == 'pod5' else None
    if source_format == 'blow5':
        source = _import_pyslow5().Open(input_path, 'r')
        try:
            groups = source.get_num_read_groups()
            metadata['s5_headers'] = [source.get_all_headers(read_group=rg) for rg in range(groups)]
            metadata['run_info_dict'] = {rg: rg for rg in range(groups)}
            try:
                labels = source.get_aux_enum_labels('end_reason')
            except Exception:
                labels = None
            if labels is not None:
                metadata['s5_end_reason_labels'] = labels
            indexed_ids, _ = source.get_read_ids()
            if max_reads > 0:
                indexed_ids = indexed_ids[:max_reads]
            read_ids = [str(uuid.UUID(value)) for value in indexed_ids]
        finally:
            _close_source(source)
    else:
        with pod5.Reader(input_path) as source:
            for read in source.reads():
                key = _make_pore_key(read.pore)
                if key not in metadata['pore_dict']:
                    metadata['pore_dict'][key] = len(metadata['pore_dict'])
                metadata['run_info_dict'].setdefault(int(read.run_info_index), read.run_info)
                read_ids.append(str(read.read_id))
                if sample_counts is not None:
                    sample_counts.append(int(read.num_samples))
                if 0 < max_reads <= len(read_ids):
                    break
    if len(set(read_ids)) != len(read_ids):
        raise ValueError('Duplicate source read UUIDs cannot be assigned to independent shards')
    if metadata['source_identity'] != _source_identity(input_path):
        raise ValueError('Source file changed while collecting shard metadata')
    metadata['num_reads'] = len(read_ids)
    if include_sample_counts:
        return metadata, read_ids, sample_counts
    return metadata, read_ids


def _canonical_compression_header(args, source_format, selected_ids):
    path = getattr(args, 'metadata_header', None)
    if path is None:
        return None
    if selected_ids is None:
        raise ValueError('--metadata-header requires --read-ids-file')
    with open(path, 'rb') as source:
        header, data_start = _read_header(source)
        if source.read(1):
            raise ValueError('--metadata-header must contain only a SIG2 header')
    if not isinstance(header, dict):
        raise ValueError('Shared metadata must be a dictionary')
    if header.get('source_identity') != _source_identity(args.input):
        raise ValueError('Shared metadata source identity does not match the input file')
    expected = {
        'format_version': FORMAT_VERSION, 'source_format': source_format,
        'chunk_len': args.chunk_len, 'model_dim': args.dim,
        'model_num_dec_layers': args.num_dec_layers, 'model_window_size': args.window_size,
        'lsb_eliminate': args.lsb_eliminate, 'model_embed': args.model_embed,
    }
    for name, value in expected.items():
        if header.get(name) != value:
            raise ValueError(f'Shared metadata disagrees with compression option {name}')
    for name in ('pore_dict', 'run_info_dict', 'num_reads'):
        if name not in header:
            raise ValueError(f'Shared metadata is missing {name}')
    if type(header['num_reads']) is not int or header['num_reads'] < 0:
        raise ValueError('Shared metadata num_reads must be a nonnegative integer')
    for name in ('pore_dict', 'run_info_dict'):
        if not isinstance(header[name], dict):
            raise ValueError(f'Shared metadata {name} must be a dictionary')
    pore_indices = list(header['pore_dict'].values())
    if (any(type(index) is not int or not 0 <= index <= 0xFFFFFFFF for index in pore_indices)
            or len(set(pore_indices)) != len(pore_indices)):
        raise ValueError('Shared metadata pore indices must be distinct uint32 integers')
    if any(type(index) is not int or not 0 <= index <= 0xFFFFFFFF
           for index in header['run_info_dict']):
        raise ValueError('Shared metadata run-info indices must be uint32 integers')
    if source_format == 'blow5':
        headers = header.get('s5_headers')
        if not isinstance(headers, list) or not headers or any(not isinstance(item, dict) for item in headers):
            raise ValueError('Shared BLOW5 metadata requires its original read-group headers')
        if set(header['run_info_dict']) != set(range(len(headers))):
            raise ValueError('Shared BLOW5 read-group headers and run-info indices disagree')
    if len(selected_ids) > header['num_reads']:
        raise ValueError('Selection is larger than the canonical source inventory')
    if 'model_identity' in header and header['model_identity'] != _source_identity(args.model_path):
        raise ValueError('Shared metadata model identity does not match the checkpoint')
    if args.model_embed == 'path':
        if os.path.abspath(header.get('model_data') or '') != os.path.abspath(args.model_path):
            raise ValueError('Shared metadata refers to a different model path')
    elif args.model_embed == 'embed':
        with open(args.model_path, 'rb') as model_file:
            if model_file.read() != header.get('model_data'):
                raise ValueError('Shared metadata embeds different model weights')
    elif header.get('model_data') is not None:
        raise ValueError('Shared metadata with model_embed=none cannot contain model data')
    return header


def _checked_index_header(index, archive_reads):
    prefix = index.read(8)
    if len(prefix) != 8 or prefix[:4] != IDX_MAGIC:
        raise ValueError('Not a valid SIDX index')
    count = U32.unpack(prefix[4:])[0]
    if count != archive_reads or os.fstat(index.fileno()).st_size != 8 + 24 * count:
        raise ValueError('Index count/size does not match the archive; rebuild the index')
    return count


def _index_entry(index, number, data_start, file_size):
    index.seek(8 + 24 * number)
    entry = index.read(24)
    if len(entry) != 24:
        raise ValueError('Truncated SIDX entry')
    offset = U64.unpack(entry[16:])[0]
    if not data_start <= offset < file_size:
        raise ValueError('SIDX entry points outside archive records; rebuild the index')
    return entry[:16], offset


def _decompression_selection(args, num_reads, data_start):
    record_start = getattr(args, 'record_start', 0)
    record_count = getattr(args, 'record_count', None)
    read_ids_file = getattr(args, 'read_ids_file', None)
    index_path = getattr(args, 'index_file', None) or (args.input + '.idx')
    file_size = os.path.getsize(args.input)
    if read_ids_file:
        if record_start or record_count is not None:
            raise ValueError('--read-ids-file cannot be combined with --record-start/--record-count')
        read_ids = _read_ids_file(read_ids_file)
        if args.max_reads > 0:
            read_ids = read_ids[:args.max_reads]
        wanted = {uuid.UUID(value).bytes for value in read_ids}
        offsets = {}
        with open(index_path, 'rb') as index:
            count = _checked_index_header(index, num_reads)
            for _ in range(count):
                entry = index.read(24)
                if entry[:16] in wanted:
                    if entry[:16] in offsets:
                        raise ValueError('Duplicate read UUID in index; rebuild the index')
                    offset = U64.unpack(entry[16:])[0]
                    if not data_start <= offset < file_size:
                        raise ValueError('SIDX entry points outside archive records')
                    offsets[entry[:16]] = offset
        missing = wanted.difference(offsets)
        if missing:
            raise ValueError(f'{len(missing)} requested read UUID(s) are not in the index')
        records = [(uuid.UUID(value).bytes, offsets[uuid.UUID(value).bytes]) for value in read_ids]
        return {'count': len(records), 'records': records, 'start_offset': data_start,
                'end_offset': None, 'expected_first_id': None}
    if record_start < 0 or record_start > num_reads:
        raise ValueError('--record-start is outside the archive read range')
    remaining = num_reads - record_start
    if record_count is not None and not 0 <= record_count <= remaining:
        raise ValueError('--record-count exceeds the remaining archive reads')
    count = remaining if record_count is None else record_count
    if args.max_reads > 0:
        count = min(count, args.max_reads)
    first_id = None
    start_offset = data_start
    end_offset = file_size if record_start + count == num_reads else None
    if record_start or getattr(args, 'index_file', None):
        with open(index_path, 'rb') as index:
            _checked_index_header(index, num_reads)
            if record_start < num_reads:
                first_id, start_offset = _index_entry(index, record_start, data_start, file_size)
            else:
                start_offset = file_size
            if record_start + count < num_reads:
                _, end_offset = _index_entry(index, record_start + count, data_start, file_size)
            if start_offset > end_offset or (count and start_offset == end_offset):
                raise ValueError('SIDX interval offsets are inconsistent; rebuild the index')
    return {'count': count, 'records': None, 'start_offset': start_offset,
            'end_offset': end_offset, 'expected_first_id': first_id}


@profiled_stage('delta_split')
def _split_signal(sig: np.ndarray):
    zz = _delta_zigzag_encode(sig)
    is_large = zz >= 256
    small_vals = zz[~is_large].astype(np.uint8)
    large_vals = zz[is_large].astype(np.uint16)
    large_pos = np.where(is_large)[0].astype(np.uint32)

    if len(large_pos) > 0:
        large_pos_deltas = np.empty_like(large_pos)
        large_pos_deltas[0] = large_pos[0]
        if len(large_pos) > 1:
            large_pos_deltas[1:] = np.diff(large_pos) - 1
    else:
        large_pos_deltas = np.array([], dtype=np.uint32)

    l_vals_adjusted = (large_vals - 256).astype(np.uint16)
    return int(sig[0]), small_vals, l_vals_adjusted, large_pos_deltas


@profiled_stage('signal_reconstruction')
def _reassemble_signal(sig0, N, small_vals, c_large, l_bytes, p_bytes):
    large_vals = (
        _rice_decompress(l_bytes, c_large, fallback_bits=16).astype(np.uint16) + 256
    )
    large_pos_deltas = _rice_decompress(p_bytes, c_large, fallback_bits=32)
    if len(large_pos_deltas) > 1:
        large_pos_deltas[1:] += 1
    large_pos = (
        np.cumsum(large_pos_deltas).astype(np.int64)
        if c_large > 0
        else np.array([], dtype=np.int64)
    )

    zz = np.zeros(N - 1, dtype=np.uint16)
    small_mask = np.ones(N - 1, dtype=bool)
    if len(large_pos) > 0:
        idx = large_pos.astype(np.intp)
        zz[idx] = large_vals
        small_mask[idx] = False
    zz[small_mask] = small_vals

    return _delta_zigzag_decode(sig0, zz)


def _write_split_blobs(f, sig0, N, c_small, s_bytes, l_bytes, p_bytes):
    f.write(SPLIT_HDR_FMT.pack(sig0, N, c_small))
    f.write(U32.pack(len(s_bytes)))
    f.write(s_bytes)
    f.write(U32.pack(len(l_bytes)))
    f.write(l_bytes)
    f.write(U32.pack(len(p_bytes)))
    f.write(p_bytes)


def _read_split_blobs(f):
    split_hdr = f.read(SPLIT_HDR_FMT.size)
    sig0, N, c_small = SPLIT_HDR_FMT.unpack(split_hdr)
    c_large = N - 1 - c_small
    (s_len,) = U32.unpack(f.read(4))
    s_bytes = f.read(s_len)
    (l_len,) = U32.unpack(f.read(4))
    l_bytes = f.read(l_len)
    (p_len,) = U32.unpack(f.read(4))
    p_bytes = f.read(p_len)
    return {
        'sig0': sig0, 'N': N, 'c_small': c_small, 'c_large': c_large,
        's_bytes': s_bytes, 'l_bytes': l_bytes, 'p_bytes': p_bytes,
    }


@profiled_stage('output.archive_record')
def _write_read_payload(f, rec_bytes, sig0, N, c_small, s_bytes, l_bytes, p_bytes):
    f.write(rec_bytes)
    _write_split_blobs(f, sig0, N, c_small, s_bytes, l_bytes, p_bytes)


@profiled_stage('input.archive_record')
def _parse_read_payload(f):
    rec_data = f.read(REC_FMT.size)
    if not rec_data or len(rec_data) < REC_FMT.size:
        return None
    rec_info = REC_FMT.unpack(rec_data)
    return rec_info, _read_split_blobs(f)


@profiled_stage('output.archive_record')
def _write_read_payload_blow5(f, read_id_bytes, meta_bytes,
                              sig0, N, c_small, s_bytes, l_bytes, p_bytes):
    f.write(REC_FMT_BLOW5.pack(read_id_bytes))
    f.write(U32.pack(len(meta_bytes)))
    f.write(meta_bytes)
    _write_split_blobs(f, sig0, N, c_small, s_bytes, l_bytes, p_bytes)


@profiled_stage('input.archive_record')
def _parse_read_payload_blow5(f):
    rec_data = f.read(REC_FMT_BLOW5.size)
    if not rec_data or len(rec_data) < REC_FMT_BLOW5.size:
        return None
    (read_id_bytes,) = REC_FMT_BLOW5.unpack(rec_data)
    (m_len,) = U32.unpack(f.read(4))
    meta = pickle.loads(f.read(m_len)) if m_len else {}
    return read_id_bytes, meta, _read_split_blobs(f)


def _resolve_model_path(global_dict, cli_model_path, console):
    if cli_model_path is not None:
        return cli_model_path, None  # (path, temp_file_to_clean_up)

    model_embed = global_dict.get('model_embed', 'path')

    if model_embed == 'path':
        path = global_dict.get('model_data')
        if path is None:
            console.print(
                "[bold red]✗ No model path in header and --model-path not given.[/]"
            )
            sys.exit(1)
        return path, None

    elif model_embed == 'embed':
        model_bytes = global_dict.get('model_data')
        if model_bytes is None:
            console.print("[bold red]✗ model_embed=embed but no weight data in header.[/]")
            sys.exit(1)
        tmp = tempfile.NamedTemporaryFile(suffix='.pth', delete=False)
        tmp.write(model_bytes)
        tmp.close()
        console.print(f"[dim]Extracted embedded model → {tmp.name}[/]")
        return tmp.name, tmp.name

    else:  # 'none'
        console.print(
            "[bold red]✗ model_embed=none in header; you must specify --model-path.[/]"
        )
        sys.exit(1)


def cmd_compress(args, console):
    with ExitStack() as resources:
        return _cmd_compress(args, console, resources)


def _cmd_compress(args, console, resources):
    profiler = current_profiler()
    if args.model_path is None:
        if args.lsb_eliminate == 0:
            args.model_path = DEFAULT_QUANT_LOSSLESS_MODEL_PATH
        elif args.lsb_eliminate == DEFAULT_LSB_SETTING_FOR_LOSSY_MODEL:
            args.model_path = DEFAULT_QUANT_LOSSY_MODEL_PATH
        else:
            console.print(
                f"[bold red]✗ No pretrained model for --lsb-eliminate "
                f"{args.lsb_eliminate}.[/] Supported values are 0 (lossless) and "
                f"{DEFAULT_LSB_SETTING_FOR_LOSSY_MODEL} (lossy); pass --model-path "
                f"to use a custom model."
            )
            sys.exit(1)
        console.print(
            f"[dim]Auto-selected model for --lsb-eliminate {args.lsb_eliminate}: "
            f"{args.model_path}[/]"
        )

    input_fmt = _detect_format(args.input)
    if input_fmt is None:
        input_fmt = 'pod5'  # default fallback
    selected_ids = (_read_ids_file(args.read_ids_file)
                    if getattr(args, 'read_ids_file', None) else None)
    if selected_ids is not None and args.max_reads > 0:
        selected_ids = selected_ids[:args.max_reads]
    if selected_ids is not None and not os.path.isfile(args.input):
        raise ValueError('--read-ids-file requires one source file')
    canonical_header = _canonical_compression_header(args, input_fmt, selected_ids)
    pod5_plan = None
    if _use_pod5_reopen(args):
        if not os.path.isfile(args.input):
            raise ValueError('--pod5-reader-mode reopen requires one POD5 input file')
        from .pod5_reopen import Pod5ReadPlan
        pod5_plan = resources.enter_context(Pod5ReadPlan(
            args.input, directory=Path(args.output).absolute().parent))

    console.print(Panel.fit(
        f"[bold cyan]SigComp — Compress[/]\n"
        f"  Input   [bold]{args.input}[/]  ([bold]{input_fmt.upper()}[/])\n"
        f"  Output  [bold]{args.output}[/]\n"
        f"  Device [bold]{args.device_label}[/]   Batch [bold]{args.effective_batch_size}[/]   "
        f"ChunkLen [bold]{args.chunk_len}[/]   "
        f"ModelEmbed [bold]{args.model_embed}[/]"
        + (f"   LSB-Eliminate [bold]{args.lsb_eliminate}[/]" if args.lsb_eliminate > 0 else ""),
        border_style="cyan",
    ))

    pore_dict = {}
    run_info_dict = {}
    pore_idx_counter = 0
    s5_headers = None
    s5_end_reason_labels = None

    read_count = 0
    if canonical_header is not None:
        console.print("[dim]Phase 1 — Reusing canonical shard metadata…[/]")
        pore_dict = canonical_header['pore_dict']
        run_info_dict = canonical_header['run_info_dict']
        s5_headers = canonical_header.get('s5_headers')
        s5_end_reason_labels = canonical_header.get('s5_end_reason_labels')
        read_count = len(selected_ids)
        if pod5_plan is not None:
            console.print('[dim]Planning POD5 flushes from selected read IDs and sample counts…[/]')
            pod5_plan.scan(selection=selected_ids)
            if pod5_plan.count != read_count:
                raise ValueError('POD5 shard read plan does not match the selected read count')
    elif input_fmt == 'blow5':
        console.print("[dim]Phase 1 — Pre-scanning metadata…[/]")
        pyslow5 = _import_pyslow5()
        s5 = pyslow5.Open(args.input, 'r')
        try:
            num_rg = s5.get_num_read_groups()
            s5_headers = [s5.get_all_headers(read_group=rg) for rg in range(num_rg)]
            try:
                s5_end_reason_labels = s5.get_aux_enum_labels('end_reason')
            except Exception:
                s5_end_reason_labels = None
        finally:
            _close_source(s5)
        metadata_source, metadata_reads = _open_source_reads(args.input, input_fmt, selected_ids)
        try:
            for read in profile_iterable(metadata_reads, 'metadata.read_records'):
                ch = read.get('channel_number', '0')
                mux = read.get('start_mux', 0)
                pore_key = (('channel_number', str(ch)), ('start_mux', int(mux)))
                if pore_key not in pore_dict:
                    pore_dict[pore_key] = pore_idx_counter
                    pore_idx_counter += 1
                rg = read.get('read_group', 0)
                if rg not in run_info_dict:
                    run_info_dict[rg] = rg
                read_count += 1
                if 0 < args.max_reads <= read_count:
                    break
        finally:
            _close_source(metadata_source)
    elif pod5_plan is not None:
        console.print('[dim]Phase 1 — Pre-scanning metadata and planning POD5 flushes…[/]')

        def collect_pod5_metadata(read):
            pore_key = _make_pore_key(read.pore)
            if pore_key not in pore_dict:
                pore_dict[pore_key] = len(pore_dict)
            ri = read.run_info_index
            if ri not in run_info_dict:
                run_info_dict[ri] = read.run_info

        pod5_plan.scan(selection=selected_ids, max_reads=args.max_reads, on_read=collect_pod5_metadata)
        read_count = pod5_plan.count
    else:
        console.print("[dim]Phase 1 — Pre-scanning metadata…[/]")
        metadata_source, metadata_reads = _open_source_reads(args.input, input_fmt, selected_ids)
        try:
            for read in profile_iterable(metadata_reads, 'metadata.read_records'):
                pore_key = _make_pore_key(read.pore)
                if pore_key not in pore_dict:
                    pore_dict[pore_key] = pore_idx_counter
                    pore_idx_counter += 1
                ri = read.run_info_index
                if ri not in run_info_dict:
                    run_info_dict[ri] = read.run_info
                read_count += 1
                if 0 < args.max_reads <= read_count:
                    break
        finally:
            _close_source(metadata_source)
    if selected_ids is not None and read_count != len(selected_ids):
        raise ValueError('Not all selected source reads were found exactly once')

    console.print(
        f"  [bold]{read_count:,}[/] reads, "
        f"[bold]{len(pore_dict)}[/] pore configs, "
        f"[bold]{len(run_info_dict)}[/] run infos"
    )

    model_embed_data = None
    if canonical_header is not None:
        model_embed_data = canonical_header.get('model_data')
    elif args.model_embed == 'path':
        model_embed_data = args.model_path
    elif args.model_embed == 'embed':
        console.print(f"[dim]Reading model weights for embedding…[/]")
        with open(args.model_path, 'rb') as mf:
            model_embed_data = mf.read()
        console.print(
            f"  Model size: [bold]{len(model_embed_data) / MB_UNIT:.1f} MB[/]"
        )

    global_dict = {
        'format_version': FORMAT_VERSION,
        'source_format': input_fmt,
        'run_info_dict': run_info_dict,
        'pore_dict': pore_dict,
        'num_reads': read_count,
        'chunk_len': args.chunk_len,
        'model_embed': args.model_embed,
        'model_data': model_embed_data,
        'model_dim': args.dim,
        'model_num_dec_layers': args.num_dec_layers,
        'model_window_size': args.window_size,
        'lsb_eliminate': args.lsb_eliminate,
        'command': ' '.join(sys.argv),
        'datetime': datetime.datetime.now().isoformat(),
    }
    if s5_headers is not None:
        global_dict['s5_headers'] = s5_headers
    if s5_end_reason_labels is not None:
        global_dict['s5_end_reason_labels'] = s5_end_reason_labels
    if canonical_header is not None:
        global_dict = dict(canonical_header)
        global_dict.update(num_reads=read_count, command=' '.join(sys.argv),
                           datetime=datetime.datetime.now().isoformat())

    out_f = resources.enter_context(open(args.output, 'wb'))
    _write_header(out_f, global_dict)  # type: ignore

    console.print(f"[dim]Phase 2 — Loading model onto {args.device_label}…[/]")
    model = _load_model(
        args.model_path, args.gpu, args.chunk_len,
        dim=args.dim, num_dec_layers=args.num_dec_layers, window_size=args.window_size,
        device=args.device, cpu_inference_batch_size=args.cpu_inference_batch_size,
        cpu_backend=args.cpu_backend,
        cpu_epilogue=args.cpu_epilogue, cpu_entropy_parallel=args.cpu_entropy_parallel,
        cpu_entropy_min_streams=args.cpu_entropy_min_streams,
        cpu_epilogue_parallel=args.cpu_epilogue_parallel,
        cpu_epilogue_min_work=args.cpu_epilogue_min_work, cpu_workspace=args.cpu_workspace,
    )

    total_compressed_bytes = 0
    total_samples = 0
    reads_buffer = []
    accumulated_chunks = 0
    processed = 0

    @profiled_stage('compress_batch')
    def flush_compress_batch(batch=None):
        nonlocal total_compressed_bytes, total_samples
        if batch is None:
            batch = reads_buffer
        if not batch:
            return

        list_of_small_vals = []
        cached = {}
        lsb = args.lsb_eliminate
        for bi, r in enumerate(batch):
            with scope(profiler, 'input.signal'):
                if input_fmt == 'blow5':
                    sig = np.asarray(r['signal'], dtype=np.int16)
                else:
                    sig = r.signal
            total_samples += len(sig)
            if lsb > 0:
                with scope(profiler, 'lossy_transform'):
                    sig = _lossy_encode(_reduce_lsb(sig, lsb), lsb)
            sig0, small_vals, l_adj, pos_deltas = _split_signal(sig)
            cached[bi] = (len(sig), sig0, len(small_vals), l_adj, pos_deltas)
            list_of_small_vals.append(small_vals)
            del sig

        small_vals = l_adj = pos_deltas = None
        batch_samples = sum(value[0] for value in cached.values())

        compressed_small = model.batch_inference_reads(
            list_of_reads=list_of_small_vals,
            batch_size=args.effective_batch_size,
            use_tqdm=False,
        )
        list_of_small_vals.clear()

        for i, r in enumerate(batch):
            n_samples, sig0, n_small, l_adj, pos_deltas = cached.pop(i)

            s_bytes = compressed_small[i]
            l_bytes = _rice_compress(l_adj, fallback_bits=16)
            p_bytes = _rice_compress(pos_deltas, fallback_bits=32)

            if input_fmt == 'blow5':
                rid_bytes = uuid.UUID(r['read_id']).bytes
                meta = {
                    k: v for k, v in r.items()
                    if k not in ('signal', 'len_raw_signal')
                }
                meta_bytes = pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL)

                _write_read_payload_blow5(
                    out_f, rid_bytes, meta_bytes,
                    sig0, n_samples, n_small, s_bytes, l_bytes, p_bytes,
                )
                total_compressed_bytes += (
                    REC_FMT_BLOW5.size + 4 + len(meta_bytes)
                    + SPLIT_HDR_FMT.size + 12
                    + len(s_bytes) + len(l_bytes) + len(p_bytes)
                )
            else:
                rec_bytes = REC_FMT.pack(
                    r.read_id.bytes if hasattr(r.read_id, 'bytes')
                    else uuid.UUID(str(r.read_id)).bytes,
                    float(r.calibration.offset),
                    float(r.calibration.scale),
                    int(r.end_reason.forced),
                    int(r.end_reason_index),
                    float(r.median_before),
                    int(pore_dict[_make_pore_key(r.pore)]),
                    int(r.read_number),
                    int(r.run_info_index),
                    int(r.start_sample),
                )

                _write_read_payload(
                    out_f, rec_bytes, sig0, n_samples, n_small,
                    s_bytes, l_bytes, p_bytes,
                )
                total_compressed_bytes += (
                    REC_FMT.size + SPLIT_HDR_FMT.size + 12
                    + len(s_bytes) + len(l_bytes) + len(p_bytes)
                )

        if profiler is not None:
            profiler.add_count('raw_signal_samples', batch_samples)
            profiler.add_count('reads', len(batch))
        batch.clear()

    console.print("[dim]Phase 3 — Compressing…[/]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("•"),
        VRAMColumn(args.gpu if args.device == 'cuda' else None),
        console=console,
    ) as progress:
        task = progress.add_task("Compressing", total=read_count)

        use_dynamic_flush = (args.chunk_size <= 0)

        selected_seen = set()
        if pod5_plan is not None:
            for group in pod5_plan.groups(
                    chunk_len=args.chunk_len,
                    chunk_capacity=args.effective_batch_size * args.effective_batch_number,
                    chunk_size=args.chunk_size):
                count = pod5_plan.process_group(group, flush_compress_batch)
                processed += count
                progress.update(task, advance=count)
            pod5_plan.assert_unchanged()
        else:
            input_source, read_iter = _open_source_reads(args.input, input_fmt, selected_ids)
            try:
                for read in profile_iterable(read_iter, 'input.read_records'):
                    if selected_ids is not None:
                        read_id = str(uuid.UUID(read['read_id'])) if input_fmt == 'blow5' else str(read.read_id)
                        if read_id in selected_seen:
                            raise ValueError(f'Source returned duplicate selected read {read_id}')
                        selected_seen.add(read_id)
                    reads_buffer.append(read)
                    processed += 1
                    progress.update(task, advance=1)

                    if use_dynamic_flush:
                        if input_fmt == 'blow5':
                            est_tokens = max(1, read['len_raw_signal'] - 1)
                        else:
                            est_tokens = max(1, read.num_samples - 1)
                        accumulated_chunks += max(1, math.ceil(est_tokens / args.chunk_len))
                        should_flush = accumulated_chunks >= (args.effective_batch_size * args.effective_batch_number)
                    else:
                        should_flush = len(reads_buffer) >= args.chunk_size

                    if should_flush:
                        flush_compress_batch()
                        accumulated_chunks = 0

                    if processed >= read_count:
                        break

                flush_compress_batch()
            finally:
                reads_buffer.clear()
                read = None
                try:
                    if hasattr(read_iter, 'close'):
                        read_iter.close()
                finally:
                    read_iter = None
                    _close_source(input_source)

    if processed != read_count or (pod5_plan is None and selected_ids is not None
                                  and selected_seen != set(selected_ids)):
        out_f.close()
        raise ValueError('Source read inventory changed or selected reads are missing; shard is incomplete')
    if canonical_header is not None and canonical_header['source_identity'] != _source_identity(args.input):
        out_f.close()
        raise ValueError('Source file changed during shard compression')
    if (canonical_header is not None and 'model_identity' in canonical_header
            and canonical_header['model_identity'] != _source_identity(args.model_path)):
        out_f.close()
        raise ValueError('Model checkpoint changed during shard compression')

    with scope(profiler, 'output.close'):
        out_f.close()
    if pod5_plan is not None:
        console.print(pod5_plan.summary(), markup=False)

    file_size = os.path.getsize(args.output)
    if profiler is not None:
        profiler.counters['archive_bytes'] = file_size
    uncompressed_est = total_samples * 2 + read_count * REC_FMT.size
    ratio = uncompressed_est / max(1, file_size)
    bps = (file_size * 8) / max(1, total_samples)

    tbl = Table(
        title="✅ Compression Summary",
        box=box.ROUNDED, border_style="green", title_style="bold green",
    )
    tbl.add_column("Metric", style="bold")
    tbl.add_column("Value", justify="right")
    tbl.add_row("Reads", f"{read_count:,}")
    tbl.add_row("Total Samples", f"{total_samples:,}")
    tbl.add_row("Uncompressed (est.)", f"{uncompressed_est / MB_UNIT:.2f} MB")
    tbl.add_row("Compressed", f"{file_size / MB_UNIT:.2f} MB")
    tbl.add_row("Ratio", f"{ratio:.3f}×")
    tbl.add_row("Bits / Sample", f"{bps:.3f}")
    console.print(tbl)
    console.print(f"[bold green]✓ Saved to {args.output}[/]")


def cmd_decompress(args, console):
    profiler = current_profiler()
    with open(args.input, 'rb') as f:
        global_dict, data_start = _read_header(f)  # type: ignore

    num_reads = global_dict['num_reads']
    chunk_len = global_dict['chunk_len']
    model_dim = global_dict.get('model_dim', None)
    model_layers = global_dict.get('model_num_dec_layers', None)
    model_ws = global_dict.get('model_window_size', None)
    lsb_eliminate = global_dict.get('lsb_eliminate', 0)
    pore_dict = global_dict['pore_dict']
    source_format = global_dict.get('source_format', 'pod5')
    selection = _decompression_selection(args, num_reads, data_start)

    output_fmt = _detect_format(args.output)
    if output_fmt is None:
        output_fmt = 'pod5'

    if output_fmt != 'bin' and output_fmt != source_format:
        console.print(
            f"[bold red]✗ Unsupported conversion:[/] the source format is "
            f"[bold]{source_format.upper()}[/], so the output must be "
            f"[bold]{source_format.upper()}[/] (or raw [bold].bin[/]), "
            f"not [bold]{output_fmt.upper()}[/].\n"
            f"  SigComp only decompresses back to the original source format."
        )
        sys.exit(1)

    console.print(Panel.fit(
        f"[bold cyan]SigComp v2 — Decompress[/]\n"
        f"  Input   [bold]{args.input}[/]  (source: [bold]{source_format.upper()}[/])\n"
        f"  Output  [bold]{args.output}[/]  ([bold]{output_fmt.upper()}[/])\n"
        f"  Device [bold]{args.device_label}[/]   Batch [bold]{args.effective_batch_size}[/]"
        + (f"   LSB-Eliminate [bold]{lsb_eliminate}[/] (lossy)" if lsb_eliminate > 0 else ""),
        border_style="cyan",
    ))
    run_info_dict = global_dict['run_info_dict']
    index_to_pore = {v: dict(k) for k, v in pore_dict.items()}

    model_path, tmp_model = _resolve_model_path(
        global_dict, args.model_path, console,
    )
    console.print(f"[dim]Loading model from {model_path} onto {args.device_label}…[/]")
    model = _load_model(model_path, args.gpu, chunk_len,
                        dim=model_dim, num_dec_layers=model_layers, window_size=model_ws,
                        device=args.device, cpu_inference_batch_size=args.cpu_inference_batch_size,
                        cpu_backend=args.cpu_backend, cpu_epilogue=args.cpu_epilogue,
                        cpu_entropy_parallel=args.cpu_entropy_parallel,
                        cpu_entropy_min_streams=args.cpu_entropy_min_streams,
                        cpu_epilogue_parallel=args.cpu_epilogue_parallel,
                        cpu_epilogue_min_work=args.cpu_epilogue_min_work,
                        cpu_workspace=args.cpu_workspace)
    if profiler is not None:
        profiler.metadata.update(chunk_len=chunk_len, dim=model_dim,
                                 num_dec_layers=model_layers, window_size=model_ws,
                                 lsb_eliminate=lsb_eliminate)

    total_to_process = selection['count']

    f_in = open(args.input, 'rb')
    f_in.seek(selection['start_offset'])
    reader = f_in

    f_out = None
    writer = None
    s5_writer = None

    if output_fmt == 'blow5':
        pyslow5 = _import_pyslow5()
        s5_writer = pyslow5.Open(
            args.output, 'w',
            rec_press=getattr(args, 'rec_press', 'zlib'),
            sig_press=getattr(args, 'sig_press', 'svb-zd'),
        )

        s5_end_reason_labels = global_dict.get('s5_end_reason_labels', None)
        s5_headers = global_dict.get('s5_headers', None)
        if s5_headers:
            for rg, hdr in enumerate(s5_headers):
                s5_writer.write_header(
                    _prepare_s5_header(hdr),
                    read_group=rg,
                    end_reason_labels=s5_end_reason_labels,
                )
        else:
            header = s5_writer.get_empty_header()
            s5_writer.write_header(
                header, read_group=0,
                end_reason_labels=s5_end_reason_labels,
            )
    elif output_fmt == 'bin':
        f_out = open(args.output, 'wb')
    else:
        writer = pod5.Writer(args.output)

    batch_payloads = []
    accumulated_chunks = 0
    reads_processed = 0

    @profiled_stage('decompress_batch')
    def flush_decompress_batch():
        if not batch_payloads:
            return

        small_bytes_list = [p[1]['s_bytes'] for p in batch_payloads]
        small_lens_list = [p[1]['c_small'] for p in batch_payloads]

        decoded_small = model.batch_inference_reads(
            list_of_compressed_reads=small_bytes_list,
            list_of_lengths=small_lens_list,
            batch_size=args.effective_batch_size,
            use_tqdm=False,
        )

        for i, (rec, split) in enumerate(batch_payloads):
            raw_sig = _reassemble_signal(
                split['sig0'], split['N'], decoded_small[i],
                split['c_large'], split['l_bytes'], split['p_bytes'],
            )
            if lsb_eliminate > 0:
                with scope(profiler, 'lossy_transform'):
                    raw_sig = _lossy_decode(raw_sig, lsb_eliminate)

            if output_fmt == 'blow5':
                _write_blow5_record(s5_writer, rec, raw_sig)
            elif output_fmt == 'bin':
                with scope(profiler, 'output.write_bin'):
                    sig_i16 = raw_sig.astype(np.int16)
                    f_out.write(struct.pack('<I', len(sig_i16)))
                    f_out.write(sig_i16.tobytes())
            else:
                uid_bytes, offset, scale, forced, er_idx, median, \
                    pore_idx, read_num, run_info_idx, start_samp = rec
                output_read = pod5.Read(
                    read_id=uuid.UUID(bytes=uid_bytes),
                    pore=pod5.Pore(**index_to_pore[pore_idx]),
                    calibration=pod5.Calibration(offset=offset, scale=scale),
                    read_number=read_num,
                    start_sample=start_samp,
                    median_before=median,
                    end_reason=pod5.EndReason(
                        reason=pod5.EndReasonEnum(er_idx), forced=bool(forced),
                    ),
                    run_info=run_info_dict[run_info_idx],
                    signal=raw_sig,
                )
                with scope(profiler, 'output.write_pod5'):
                    writer.add_read(output_read)
            if profiler is not None:
                profiler.add_count('raw_signal_samples', len(raw_sig))
                profiler.add_count('reads', 1)

        batch_payloads.clear()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("•"),
        VRAMColumn(args.gpu if args.device == 'cuda' else None),
        console=console,
    ) as progress:
        task = progress.add_task("Decompressing", total=total_to_process)

        use_dynamic_flush = (args.chunk_size <= 0)

        while reads_processed < total_to_process:
            expected_id = None
            if selection['records'] is not None:
                expected_id, record_offset = selection['records'][reads_processed]
                reader.seek(record_offset)
            elif reads_processed == 0:
                expected_id = selection['expected_first_id']
            if source_format == 'blow5':
                result = _parse_read_payload_blow5(reader)
                if result is None:
                    raise EOFError('Archive ended before all selected records were read')
                rid_bytes, rec, split_state = result
            else:
                result = _parse_read_payload(reader)
                if result is None:
                    raise EOFError('Archive ended before all selected records were read')
                rec, split_state = result
                rid_bytes = rec[0]
            if expected_id is not None and rid_bytes != expected_id:
                raise ValueError('Index UUID does not match archive record; rebuild the index')
            if selection['end_offset'] is not None and reader.tell() > selection['end_offset']:
                raise ValueError('Record payload extends beyond the indexed interval')
            batch_payloads.append((rec, split_state))
            reads_processed += 1
            progress.update(task, advance=1)

            if use_dynamic_flush:
                accumulated_chunks += max(
                    1, math.ceil(split_state['c_small'] / chunk_len)
                )
                should_flush = (accumulated_chunks >= args.effective_batch_size * args.effective_batch_number)
            else:
                should_flush = len(batch_payloads) >= args.chunk_size

            if should_flush:
                flush_decompress_batch()
                accumulated_chunks = 0

        flush_decompress_batch()

    if selection['end_offset'] is not None and reader.tell() != selection['end_offset']:
        raise ValueError('Selected record count does not end at the indexed archive boundary')

    with scope(profiler, 'output.close'):
        if writer is not None:
            writer.close()
        if s5_writer is not None:
            s5_writer.close()
        if f_out is not None:
            f_out.close()
        f_in.close()

    if tmp_model is not None and os.path.exists(tmp_model):
        os.remove(tmp_model)

    out_labels = {'pod5': 'POD5', 'blow5': 'BLOW5', 'bin': 'raw signal binary'}
    out_label = out_labels.get(output_fmt, output_fmt.upper())
    console.print(
        f"[bold green]✓ Decompressed {reads_processed:,} reads → {args.output} ({out_label})[/]"
    )


def cmd_inspect(args, console):
    with open(args.input, 'rb') as f:
        global_dict, data_start = _read_header(f)

    file_size = os.path.getsize(args.input)
    num_reads = global_dict.get('num_reads', 0)

    model_embed = global_dict.get('model_embed', 'path')
    model_info = ""
    if model_embed == 'path':
        model_info = f"Path → {global_dict.get('model_data', 'N/A')}"
    elif model_embed == 'embed':
        md = global_dict.get('model_data', b'')
        model_info = f"Embedded ({len(md) / MB_UNIT:.1f} MB)"
    else:
        model_info = "None (must supply at decompression)"

    source_fmt = global_dict.get('source_format', 'pod5').upper()

    console.print(Panel(
        f"[bold]File       [/] {args.input}\n"
        f"[bold]Format     [/] SIGCOMP v{global_dict.get('format_version', '?')}\n"
        f"[bold]Source     [/] {source_fmt}\n"
        f"[bold]Created    [/] {global_dict.get('datetime', 'N/A')}\n"
        f"[bold]Command    [/] {global_dict.get('command', 'N/A')}\n"
        f"[bold]Model      [/] {model_info}\n"
        f"[bold]LSB Elim.  [/] {global_dict.get('lsb_eliminate', 0)}"
        f"{' (lossy)' if global_dict.get('lsb_eliminate', 0) > 0 else ' (lossless)'}\n",
        title="[bold cyan]SIGCOMP File Info[/]",
        border_style="cyan",
    ))

    tbl = Table(box=box.SIMPLE_HEAVY, border_style="blue")
    tbl.add_column("Field", style="bold")
    tbl.add_column("Value", justify="right")
    tbl.add_row("Reads", f"{num_reads:,}")
    tbl.add_row("Chunk Length", f"{global_dict.get('chunk_len', '?')}")
    tbl.add_row("Pore Configs", f"{len(global_dict.get('pore_dict', {}))}")
    tbl.add_row("Run Infos", f"{len(global_dict.get('run_info_dict', {}))}")
    tbl.add_row("Header Size", f"{data_start:,} bytes")
    tbl.add_row("File Size", f"{file_size / MB_UNIT:.2f} MB")
    console.print(tbl)
    if args.no_scan:
        return

    console.print("[dim]Scanning read payloads…[/]")
    total_samples = 0
    read_sizes = []
    total_small_bytes = 0
    total_large_bytes = 0
    total_pos_bytes = 0
    total_small_count = 0
    total_large_count = 0

    with open(args.input, 'rb') as f:
        f.seek(data_start)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            is_blow5 = global_dict.get('source_format', 'pod5') == 'blow5'
            task = progress.add_task("Scanning", total=num_reads)
            for _ in range(num_reads):
                pos_before = f.tell()
                if is_blow5:
                    rec_data = f.read(REC_FMT_BLOW5.size)
                    if not rec_data or len(rec_data) < REC_FMT_BLOW5.size:
                        break
                    (m_len,) = U32.unpack(f.read(4)); f.seek(m_len, 1)
                else:
                    rec_data = f.read(REC_FMT.size)
                    if not rec_data or len(rec_data) < REC_FMT.size:
                        break
                split_raw = f.read(SPLIT_HDR_FMT.size)
                sig0, N, c_small = SPLIT_HDR_FMT.unpack(split_raw)
                c_large = N - 1 - c_small
                total_samples += N
                total_small_count += c_small
                total_large_count += c_large

                (s_len,) = U32.unpack(f.read(4)); f.seek(s_len, 1)
                (l_len,) = U32.unpack(f.read(4)); f.seek(l_len, 1)
                (p_len,) = U32.unpack(f.read(4)); f.seek(p_len, 1)
                total_small_bytes += s_len
                total_large_bytes += l_len
                total_pos_bytes += p_len
                read_sizes.append(f.tell() - pos_before)
                progress.update(task, advance=1)

    if not read_sizes:
        return

    uncompressed_est = total_samples * 2 + len(read_sizes) * REC_FMT.size
    total_deltas = total_small_count + total_large_count  # = total_samples - num_reads

    st = Table(
        title="📊 Read Statistics",
        box=box.ROUNDED, border_style="green", title_style="bold green",
    )
    st.add_column("Metric", style="bold")
    st.add_column("Value", justify="right")
    st.add_row("Total Samples", f"{total_samples:,}")
    st.add_row("Uncompressed (est.)", f"{uncompressed_est / MB_UNIT:.2f} MB")
    st.add_row("Compressed (file)", f"{file_size / MB_UNIT:.2f} MB")
    st.add_row("Compressed (file, bytes)", f"{file_size} Bytes")
    st.add_row("Ratio", f"{uncompressed_est / max(1, file_size):.3f}×")
    st.add_row("Bits / Sample", f"{file_size * 8 / max(1, total_samples):.3f}")
    st.add_row("Avg Read Payload", f"{statistics.mean(read_sizes):,.0f} bytes")
    st.add_row("Median Read Payload", f"{statistics.median(read_sizes):,.0f} bytes")
    console.print(st)

    small_pct = 100.0 * total_small_count / max(1, total_deltas)
    large_pct = 100.0 * total_large_count / max(1, total_deltas)
    small_uncomp = total_small_count
    large_uncomp = total_large_count * 2
    pos_uncomp = total_large_count * 4

    ss = Table(
        title="📊 Per-Stream Breakdown",
        box=box.ROUNDED, border_style="magenta", title_style="bold magenta",
    )
    ss.add_column("Stream", style="bold")
    ss.add_column("Count", justify="right")
    ss.add_column("Proportion", justify="right")
    ss.add_column("Uncompressed", justify="right")
    ss.add_column("Compressed", justify="right")
    ss.add_column("Ratio", justify="right")
    ss.add_column("Bits / Symbol", justify="right")

    ss.add_row(
        "Small values (<256)",
        f"{total_small_count:,}",
        f"{small_pct:.2f}%",
        f"{small_uncomp / MB_UNIT:.2f} MB",
        f"{total_small_bytes / MB_UNIT:.2f} MB",
        f"{small_uncomp / max(1, total_small_bytes):.3f}×",
        f"{total_small_bytes * 8 / max(1, total_small_count):.3f}",
    )
    ss.add_row(
        "Large values (≥256)",
        f"{total_large_count:,}",
        f"{large_pct:.2f}%",
        f"{large_uncomp / MB_UNIT:.2f} MB",
        f"{total_large_bytes / MB_UNIT:.2f} MB",
        f"{large_uncomp / max(1, total_large_bytes):.3f}×",
        f"{total_large_bytes * 8 / max(1, total_large_count):.3f}",
    )
    ss.add_row(
        "Large positions",
        f"{total_large_count:,}",
        "—",
        f"{pos_uncomp / MB_UNIT:.2f} MB",
        f"{total_pos_bytes / MB_UNIT:.2f} MB",
        f"{pos_uncomp / max(1, total_pos_bytes):.3f}×",
        f"{total_pos_bytes * 8 / max(1, total_large_count):.3f}",
    )

    overhead_bytes = len(read_sizes) * (REC_FMT.size + SPLIT_HDR_FMT.size + 12)
    ss.add_row(
        "Metadata overhead",
        f"{len(read_sizes):,} reads",
        "—",
        "—",
        f"{overhead_bytes / MB_UNIT:.2f} MB",
        "—",
        "—",
    )
    console.print(ss)


def cmd_index(args, console):
    idx_path = args.input + '.idx'

    with open(args.input, 'rb') as f:
        global_dict, data_start = _read_header(f)  # type: ignore
        num_reads = global_dict.get('num_reads', 0)

        entries = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            is_blow5 = global_dict.get('source_format', 'pod5') == 'blow5'
            task = progress.add_task("Building index", total=num_reads)

            for _ in range(num_reads):
                file_offset = f.tell()
                if is_blow5:
                    rec_data = f.read(REC_FMT_BLOW5.size)
                    if not rec_data or len(rec_data) < REC_FMT_BLOW5.size:
                        break
                    read_id_bytes = rec_data[:16]
                    (m_len,) = U32.unpack(f.read(4)); f.seek(m_len, 1)
                else:
                    rec_data = f.read(REC_FMT.size)
                    if not rec_data or len(rec_data) < REC_FMT.size:
                        break
                    read_id_bytes = rec_data[:16]

                split_hdr = f.read(SPLIT_HDR_FMT.size)
                sig0, N, c_small = SPLIT_HDR_FMT.unpack(split_hdr)

                (s_len,) = U32.unpack(f.read(4)); f.seek(s_len, 1)
                (l_len,) = U32.unpack(f.read(4)); f.seek(l_len, 1)
                (p_len,) = U32.unpack(f.read(4)); f.seek(p_len, 1)

                entries.append((read_id_bytes, file_offset))
                progress.update(task, advance=1)

    with open(idx_path, 'wb') as f:
        f.write(IDX_MAGIC)
        f.write(U32.pack(len(entries)))
        for read_id_bytes, file_offset in entries:
            f.write(read_id_bytes)
            f.write(U64.pack(file_offset))

    console.print(f"[bold green]✓ Index built → {idx_path}[/]")
    console.print(f"  Indexed [bold]{len(entries):,}[/] reads")


def _load_index(idx_path):
    index_dict = {}
    index_list = []
    with open(idx_path, 'rb') as f:
        magic = f.read(4)
        if magic != IDX_MAGIC:
            raise ValueError(f"Not a valid index file (magic={magic!r})")
        (num_reads,) = U32.unpack(f.read(4))
        for _ in range(num_reads):
            rid_bytes = f.read(16)
            (offset,) = U64.unpack(f.read(8))
            rid = str(uuid.UUID(bytes=rid_bytes))
            index_dict[rid] = offset
            index_list.append((rid, offset))
    return index_dict, index_list


def cmd_view(args, console):
    idx_path = args.input + '.idx'
    if not os.path.exists(idx_path):
        console.print(
            "[bold red]✗ Index file not found.[/]  "
            "Run [bold]index[/] first:  "
            f"[dim]python main.py index -i {args.input}[/]"
        )
        return

    index_dict, index_list = _load_index(idx_path)
    total_reads = len(index_list)

    if args.read_ids:
        query_ids = [r.strip() for r in args.read_ids.split(',')]
        missing = [r for r in query_ids if r not in index_dict]
        if missing:
            console.print(f"[yellow]⚠ {len(missing)} read(s) not found in index[/]")
        selected = [(r, index_dict[r]) for r in query_ids if r in index_dict]
    elif args.range:
        parts = args.range.split(',')
        start, end = max(0, int(parts[0])), min(total_reads - 1, int(parts[1]))
        selected = index_list[start:end + 1]
        console.print(f"[dim]Viewing reads [{start}, {end}] ({end - start + 1} reads)[/]")
    elif args.tail:
        n = min(args.tail, total_reads)
        selected = index_list[-n:]
    else:
        n = min(args.head, total_reads)
        selected = index_list[:n]

    if not selected:
        console.print("[yellow]No reads selected.[/]")
        return

    with open(args.input, 'rb') as f:
        global_dict, _ = _read_header(f)  # type: ignore
        run_info_dict = global_dict['run_info_dict']
        pore_dict = global_dict['pore_dict']
        index_to_pore = {v: dict(k) for k, v in pore_dict.items()}
        chunk_len = global_dict.get('chunk_len', 2048)
        model_dim = global_dict.get('model_dim', None)
        model_layers = global_dict.get('model_num_dec_layers', None)
        model_ws = global_dict.get('model_window_size', None)
        lsb_eliminate = global_dict.get('lsb_eliminate', 0)
        source_format = global_dict.get('source_format', 'pod5')

        model_path, tmp_model = _resolve_model_path(
            global_dict, args.model_path, console,
        )
        model = _load_model(model_path, args.gpu, chunk_len,
                            dim=model_dim, num_dec_layers=model_layers, window_size=model_ws,
                            device=args.device, cpu_inference_batch_size=args.cpu_inference_batch_size,
                            cpu_backend=args.cpu_backend, cpu_epilogue=args.cpu_epilogue,
                            cpu_entropy_parallel=args.cpu_entropy_parallel,
                            cpu_entropy_min_streams=args.cpu_entropy_min_streams,
                            cpu_epilogue_parallel=args.cpu_epilogue_parallel,
                            cpu_epilogue_min_work=args.cpu_epilogue_min_work,
                            cpu_workspace=args.cpu_workspace)

        # `payload` is the POD5 rec_info tuple, or the BLOW5 source metadata dict.
        parsed_payloads = []
        for rid, file_offset in selected:
            f.seek(file_offset)
            if source_format == 'blow5':
                result = _parse_read_payload_blow5(f)
                if result is None:
                    continue
                _rid_bytes, payload, split = result
            else:
                result = _parse_read_payload(f)
                if result is None:
                    continue
                payload, split = result
            parsed_payloads.append((rid, payload, split))

        if not parsed_payloads:
            if tmp_model is not None and os.path.exists(tmp_model):
                os.remove(tmp_model)
            return

        small_bytes_list = [p[2]['s_bytes'] for p in parsed_payloads]
        small_lens_list = [p[2]['c_small'] for p in parsed_payloads]

        decoded_small = model.batch_inference_reads(
            list_of_compressed_reads=small_bytes_list,
            list_of_lengths=small_lens_list,
            batch_size=args.effective_batch_size,
            use_tqdm=False,
        )

        decoded_reads = []
        for i, (rid, payload, split) in enumerate(parsed_payloads):
            raw_sig = _reassemble_signal(
                split['sig0'], split['N'], decoded_small[i],
                split['c_large'], split['l_bytes'], split['p_bytes'],
            )
            if lsb_eliminate > 0:
                raw_sig = _lossy_decode(raw_sig, lsb_eliminate)
            decoded_reads.append((rid, payload, raw_sig))

    if args.output:
        ext = os.path.splitext(args.output)[1].lower()
        out_fmt = _detect_format(args.output)
        if out_fmt in ('pod5', 'blow5') and out_fmt != source_format:
            console.print(
                f"[bold red]✗ Unsupported export:[/] source is "
                f"[bold]{source_format.upper()}[/]; export to "
                f"[bold]{out_fmt.upper()}[/] is not supported "
                f"(use a matching format, a text file, or omit -o)."
            )
            if tmp_model is not None and os.path.exists(tmp_model):
                os.remove(tmp_model)
            return

        if ext == '.pod5':
            w = pod5.Writer(args.output)
            for rid, payload, raw_sig in decoded_reads:
                uid_b, off, sc, forced, er, med, pi, rn, ri, ss = payload
                w.add_read(pod5.Read(
                    read_id=uuid.UUID(bytes=uid_b),
                    pore=pod5.Pore(**index_to_pore[pi]),
                    calibration=pod5.Calibration(offset=off, scale=sc),
                    read_number=rn, start_sample=ss, median_before=med,
                    end_reason=pod5.EndReason(
                        reason=pod5.EndReasonEnum(er), forced=bool(forced),
                    ),
                    run_info=run_info_dict[ri],
                    signal=raw_sig,
                ))
            w.close()
            console.print(
                f"[bold green]✓ Wrote {len(decoded_reads)} read(s) → {args.output}[/]"
            )
        elif ext in _BLOW5_EXTS:
            pyslow5 = _import_pyslow5()
            s5w = pyslow5.Open(args.output, 'w')
            s5_end_reason_labels = global_dict.get('s5_end_reason_labels', None)
            s5_headers = global_dict.get('s5_headers', None)
            if s5_headers:
                for rg_i, hdr in enumerate(s5_headers):
                    s5w.write_header(
                        _prepare_s5_header(hdr),
                        read_group=rg_i,
                        end_reason_labels=s5_end_reason_labels,
                    )
            else:
                header = s5w.get_empty_header()
                s5w.write_header(
                    header, read_group=0,
                    end_reason_labels=s5_end_reason_labels,
                )
            for rid, payload, raw_sig in decoded_reads:
                _write_blow5_record(s5w, payload, raw_sig)
            s5w.close()
            console.print(
                f"[bold green]✓ Wrote {len(decoded_reads)} read(s) → {args.output} (BLOW5)[/]"
            )
        else:
            with open(args.output, 'w') as out_f:
                for rid, payload, raw_sig in decoded_reads:
                    out_f.write(f"{rid}\t{','.join(str(v) for v in raw_sig)}\n")
            console.print(
                f"[bold green]✓ Wrote {len(decoded_reads)} read(s) → {args.output}[/]"
            )
    else:
        for rid, payload, raw_sig in decoded_reads:
            sig_head = ','.join(str(v) for v in raw_sig[:8])
            sig_tail = ','.join(str(v) for v in raw_sig[-4:])
            if source_format == 'blow5':
                m = payload
                rn = m.get('read_number', 0)
                ss = m.get('start_sample', 0)
                off = float(m.get('offset', 0.0))
                digit = float(m.get('digitisation', 0.0))
                rng = float(m.get('range', 0.0))
                sc = (rng / digit) if digit else 0.0
                med = float(m.get('median_before', 0.0))
                end_reason_str = str(m.get('end_reason', 0))
            else:
                uid_b, off, sc, forced, er, med, pi, rn, ri, ss = payload
                end_reason_str = f"{pod5.EndReasonEnum(er).name} (forced={bool(forced)})"
            console.print(Panel(
                f"[bold]Read ID      [/] {rid}\n"
                f"[bold]Read Number  [/] {rn}\n"
                f"[bold]Start Sample [/] {ss}\n"
                f"[bold]Calibration  [/] offset={off:.4f}  scale={sc:.6f}\n"
                f"[bold]End Reason   [/] {end_reason_str}\n"
                f"[bold]Median Before[/] {med:.2f}\n"
                f"[bold]Signal Len   [/] {len(raw_sig):,}\n"
                f"[bold]Signal       [/] [{sig_head}, …, {sig_tail}]",
                title=f"[bold blue]Read[/]",
                border_style="blue",
            ))

    if tmp_model is not None and os.path.exists(tmp_model):
        os.remove(tmp_model)


class DefaultAndRawFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def _validate_cpu_placement(args, parser):
    if args.cpu_socket_id is not None or args.cpu_memory_policy is not None:
        if args.device != 'cpu':
            parser.error('--cpu-socket-id and --cpu-memory-policy require --device cpu')
        if args.cpu_socket_id is None:
            parser.error('--cpu-memory-policy requires --cpu-socket-id')
        if args.cpu_socket_id < 0:
            parser.error('--cpu-socket-id must be nonnegative')
        if args.cpu_memory_policy is None:
            args.cpu_memory_policy = 'bind'


def _configure_device(args, parser):
    import importlib.util

    _validate_cpu_placement(args, parser)
    if args.cpu_socket_id is not None:
        if _SOCKET_BINDING is None or _SOCKET_BINDING['socket_id'] != args.cpu_socket_id:
            parser.error('Socket placement must be initialized at CLI startup; run python -m sigcomp.main')
        args.cpu_socket_placement = _SOCKET_BINDING
        if args.cpu_threads is None:
            args.cpu_threads = _SOCKET_BINDING['cpu_threads']
        if args.cpu_numba_threads is None:
            args.cpu_numba_threads = _SOCKET_BINDING['cpu_numba_threads']
        if (args.cpu_threads != _SOCKET_BINDING['cpu_threads']
                or args.cpu_numba_threads != _SOCKET_BINDING['cpu_numba_threads']
                or args.cpu_memory_policy != _SOCKET_BINDING['memory_policy']):
            parser.error('CPU socket settings differ from the startup placement; restart the CLI')
    if args.device == 'gpu':
        args.device = 'cuda'
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            parser.error('CUDA is unavailable; use --device cpu')
        if importlib.util.find_spec('triton') is None:
            parser.error('GPU mode requires Triton; install it or use --device cpu')
        if not 0 <= args.gpu < torch.cuda.device_count():
            parser.error('--gpu must identify an available CUDA device')
    if args.device == 'cpu':
        if args.cpu_threads is None:
            from .cpu_runtime import get_physical_cpu_count
            try:
                args.cpu_threads = get_physical_cpu_count()
            except ValueError as exc:
                parser.error(str(exc))
        if args.cpu_threads < 1:
            parser.error('--cpu-threads must be positive')
        if args.cpu_numba_threads is None:
            args.cpu_numba_threads = args.cpu_threads
        if args.cpu_numba_threads < 1:
            parser.error('--cpu-numba-threads must be positive')
        try:
            import numba
        except Exception as exc:
            parser.error(f'CPU coding requires a working Numba installation: {exc}')
        if args.cpu_numba_threads > numba.config.NUMBA_NUM_THREADS:
            parser.error('--cpu-numba-threads (defaults to --cpu-threads) exceeds '
                         f'Numba maximum {numba.config.NUMBA_NUM_THREADS}; set '
                         'NUMBA_NUM_THREADS before starting Python to raise the '
                         'maximum, or explicitly select a lower worker count')
        torch.set_num_threads(args.cpu_threads)
        numba.set_num_threads(args.cpu_numba_threads)
        if args.cpu_socket_id is not None:
            # Detect a native runtime overriding the inherited mask. A worker
            # may use a subset, but it must never widen onto another socket/SMT.
            allowed = set(_SOCKET_BINDING['cpu_ids'])
            for task in Path('/proc/self/task').iterdir():
                try:
                    if not set(os.sched_getaffinity(int(task.name))) <= allowed:
                        parser.error(f'Native thread {task.name} widened the requested socket affinity')
                except ProcessLookupError:
                    pass  # A transient worker finished during inspection.
    args.device_label = f'cuda:{args.gpu}' if args.device == 'cuda' else 'cpu'
    # Keep the two public option families independent in saved configurations.
    # Only these effective values are consumed by coding and IO-flush logic.
    args.effective_batch_size = (args.cpu_batch_size if args.device == 'cpu'
                                 else getattr(args, 'batch_size', 10240))
    args.effective_batch_number = (getattr(args, 'cpu_batch_number', 1)
                                   if args.device == 'cpu' else getattr(args, 'batch_number', 1))
    for name in ('batch_size', 'batch_number', 'cpu_batch_size', 'cpu_batch_number',
                 'cpu_inference_batch_size',
                 'cpu_entropy_min_streams', 'cpu_epilogue_min_work'):
        if getattr(args, name, 1) < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.command == 'compress':
        if args.chunk_len < 1:
            parser.error('--chunk-len must be positive')
        if not 0 <= args.lsb_eliminate <= 15:
            parser.error('--lsb-eliminate must be in [0, 15]')
        if args.dim < 2 or args.num_dec_layers < 1 or args.window_size < 1:
            parser.error('model dimensions, layer count and window size must be positive')
        if args.metadata_header and not args.read_ids_file:
            parser.error('--metadata-header requires --read-ids-file')
        if _use_pod5_reopen(args) and not os.path.isfile(args.input):
            parser.error('--pod5-reader-mode reopen requires one POD5 input file')
    elif args.command == 'decompress':
        if args.record_start < 0 or (args.record_count is not None and args.record_count < 0):
            parser.error('--record-start and --record-count must be nonnegative')
        if args.read_ids_file and (args.record_start or args.record_count is not None):
            parser.error('--read-ids-file cannot be combined with --record-start/--record-count')


def _build_parser():
    parser = argparse.ArgumentParser(
        description="SigComp — CPU/GPU lossless and lossy nanopore signal compressor",
        formatter_class=DefaultAndRawFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s compress   -i reads.pod5  -o reads.sigcomp --device cpu\n"
            "  %(prog)s compress   -i reads.blow5 -o reads.sigcomp --device cuda\n"
            "  %(prog)s decompress -i reads.sigcomp -o reads.decompressed.pod5 --device cpu\n"
            "  %(prog)s decompress -i reads.sigcomp -o reads.decompressed.blow5 --device cuda\n"
            "  %(prog)s decompress -i reads.sigcomp -o reads.decompressed.bin --device cpu\n"
            "  %(prog)s inspect    -i reads.sigcomp\n"
            "  %(prog)s index      -i reads.sigcomp\n"
            "  %(prog)s view       -i reads.sigcomp --head 5 --device cpu\n"
            "  %(prog)s view       -i reads.sigcomp -o subset.pod5 --head 10 --device cuda\n"
        ),
    )
    sub = parser.add_subparsers(dest='command', required=True)

    pc = sub.add_parser('compress', help='Compress POD5/BLOW5 → SIGCOMP', allow_abbrev=False,
                        formatter_class=DefaultAndRawFormatter)
    pc.add_argument('-i', '--input', required=True,
                    help='Input signal file (.pod5 or .blow5/.slow5)')
    pc.add_argument('-o', '--output', required=True, help='Output .sigcomp file')
    pc.add_argument('-m', '--model-path', default=None,
                    help='Path to quantized MLP model weights (.pth). '
                         'If omitted, the model is auto-selected from --lsb-eliminate: '
                         f'0 → lossless model, {DEFAULT_LSB_SETTING_FOR_LOSSY_MODEL} → lossy model.')
    pc.add_argument('-g', '--gpu', type=int, default=0, help='GPU device ID')
    pc.add_argument('-b', '--batch-size', type=int, default=20480,
                    help='GPU batch size for neural coding; CPU uses --cpu-batch-size')
    pc.add_argument('-B', '--batch-number', type=int, default=64,
                    help='GPU batches per IO flush; CPU uses --cpu-batch-number')
    pc.add_argument('-L', '--chunk-len', type=int, default=2048,
                    help='Token chunk length for neural coding')
    pc.add_argument('-c', '--chunk-size', type=int, default=0,
                    help='Number of reads per IO flush batch. '
                         '0 = auto-tune from signal lengths and batch_size. '
                         'Positive value = manual override.')
    pc.add_argument('-n', '--max-reads', type=int, default=-1,
                    help='Max reads to compress (-1 = all)')
    pc.add_argument('--read-ids-file', default=None,
                    help='One UUID per line for indexed selection from one source file; '
                         'POD5 uses source order, BLOW5 uses list order. --max-reads '
                         'limits the UUID list before selection.')
    pc.add_argument('--metadata-header', default=None,
                    help='Canonical SIG2 header-only file for shard workers; requires '
                         '--read-ids-file and matching source/model/configuration')
    pc.add_argument('--pod5-reader-mode', choices=('persistent', 'reopen'), default='reopen',
                    help='POD5 input only: reopen a reader for each IO flush using read IDs. '
                         'Releases reader-owned caches; adds metadata/selection overhead. '
                         'Requires one POD5 file; ignored for other input formats. The model stays loaded.')
    pc.add_argument('--model-embed', choices=['path', 'embed', 'none'],
                    default='path',
                    help='How to store model reference in compressed file')
    pc.add_argument('--dim', type=int, default=CPUQuantMLPCompression.DEFAULT_DIM,
                    help='MLP hidden dimension')
    pc.add_argument('--num-dec-layers', type=int, default=CPUQuantMLPCompression.DEFAULT_NUM_DEC_LAYERS,
                    help='Number of MLP decoder layers')
    pc.add_argument('--window-size', type=int, default=CPUQuantMLPCompression.DEFAULT_WINDOW_SIZE,
                    help='Context window size (number of previous tokens)')
    pc.add_argument('--lsb-eliminate', type=int, default=0,
                    help='Lossy LSB reduction bits (0=lossless; 3=lossy)')

    pd = sub.add_parser('decompress', help='Decompress SIGCOMP → POD5/BLOW5/BIN', allow_abbrev=False,
                        formatter_class=DefaultAndRawFormatter)
    pd.add_argument('-i', '--input', required=True, help='Input .sigcomp file')
    pd.add_argument('-o', '--output', required=True,
                    help='Output file. Format detected by extension: '
                         '.pod5 → POD5, .blow5/.slow5 → BLOW5, '
                         '.bin → raw binary [uint32 len][int16[] signal]')
    pd.add_argument('-m', '--model-path', default=None,
                    help='Override model path (uses header info by default)')
    pd.add_argument('-g', '--gpu', type=int, default=0, help='GPU device ID')
    pd.add_argument('-b', '--batch-size', type=int, default=20480,
                    help='GPU batch size for neural decoding; CPU uses --cpu-batch-size')
    pd.add_argument('-B', '--batch-number', type=int, default=8,
                    help='GPU batches per IO flush; CPU uses --cpu-batch-number')
    pd.add_argument('-c', '--chunk-size', type=int, default=0,
                    help='Number of reads per IO flush batch. '
                         '0 = auto-tune from signal lengths and batch_size. '
                         'Positive value = manual override.')
    pd.add_argument('-n', '--max-reads', type=int, default=-1,
                    help='Max reads to decompress (-1 = all)')
    pd.add_argument('--read-ids-file', default=None,
                    help='One UUID per line; decompress selected indexed records in list order')
    pd.add_argument('--record-start', type=int, default=0,
                    help='First record number (0-based); nonzero start requires a current SIDX index')
    pd.add_argument('--record-count', type=int, default=None,
                    help='Number of contiguous records from --record-start (default: remainder)')
    pd.add_argument('--index-file', default=None,
                    help='SIDX index path for record/UUID selection (default: INPUT.idx); '
                         'range workers read only the boundary entries')
    pd.add_argument('--rec-press', default='zlib',
                    choices=['none', 'zlib', 'zstd'],
                    help='BLOW5 record compression (only for .blow5 output)')
    pd.add_argument('--sig-press', default='svb-zd',
                    choices=['none', 'svb-zd', 'ex-zd'],
                    help='BLOW5 signal compression (only for .blow5 output)')

    pi = sub.add_parser('inspect', help='Show file metadata & statistics',
                        formatter_class=DefaultAndRawFormatter)
    pi.add_argument('-i', '--input', required=True, help='Input .sigcomp file')
    pi.add_argument('--no-scan', action='store_true',
                    help='Skip scanning reads (show header info only)')

    px = sub.add_parser('index', help='Build random-access index (.idx)',
                        formatter_class=DefaultAndRawFormatter)
    px.add_argument('-i', '--input', required=True, help='Input .sigcomp file')

    pv = sub.add_parser('view', help='View specific reads (requires index)', allow_abbrev=False,
                        formatter_class=DefaultAndRawFormatter)
    pv.add_argument('-i', '--input', required=True, help='Input .sigcomp file')
    pv.add_argument('-o', '--output', default=None,
                    help='Output file (.pod5, .blow5/.slow5, or text; omit for console summary)')
    pv.add_argument('-m', '--model-path', default=None,
                    help='Override model path')
    pv.add_argument('-g', '--gpu', type=int, default=0, help='GPU device ID')
    pv.add_argument('-b', '--batch-size', type=int, default=10240,
                    help='GPU batch size; CPU uses --cpu-batch-size')
    pv.add_argument('--read-ids', help='Comma-separated read UUIDs')
    pv.add_argument('--head', type=int, default=1,
                    help='View first N reads')
    pv.add_argument('--tail', type=int, default=None,
                    help='View last N reads')
    pv.add_argument('--range', type=str, default=None,
                    help='Range start,end (0-indexed, inclusive)')

    for batch_parser in (pc, pd):
        batch_parser.add_argument('--cpu-batch-number', type=int, default=64,
                                  help='CPU batches per IO flush; GPU uses -B/--batch-number')

    for device_parser in (pc, pd, pv):
        device_parser.add_argument('--cpu-batch-size', type=int,
                                   default=1024 if device_parser is pc else 4096,
                                   help='CPU concurrent chunks; GPU uses -b/--batch-size')
        device_parser.add_argument('--device', choices=['cpu', 'cuda', 'gpu'], required=True,
                                   help='Required inference and entropy device; gpu is an alias for cuda')
        device_parser.add_argument('--cpu-socket-id', type=int, default=None,
                                   help='CPU only: Linux physical socket ID; bind to one allowed CPU per core before imports')
        device_parser.add_argument('--cpu-memory-policy', choices=['bind', 'inherit'], default=None,
                                   help='Requires --device cpu and --cpu-socket-id; default bind uses allowed socket NUMA nodes (libnuma); inherit sets CPU affinity only')
        device_parser.add_argument('--cpu-threads', type=int, default=None,
                                   help='CPU PyTorch workers; default physical cores available on selected socket, or all allowed cores without socket selection')
        device_parser.add_argument('--cpu-numba-threads', type=int,
                                   help='Numba workers for entropy and integer kernels; defaults to resolved --cpu-threads')
        device_parser.add_argument('--cpu-inference-batch-size', type=int, default=4096,
                                   help='Maximum context rows per CPU encoding inference call; decoding uses --cpu-batch-size')
        device_parser.add_argument('--cpu-backend', choices=['auto', 'int_mm', 'numba'],
                                   default='int_mm', help='CPU integer MLP implementation; auto enables the Numba fallback')
        device_parser.add_argument('--cpu-epilogue', choices=['auto', 'numba', 'torch'],
                                   default='numba', help='INT8 GEMM post-processing: auto uses fused Numba; torch retains eager reference path')
        device_parser.add_argument('--cpu-epilogue-parallel', choices=['auto', 'serial', 'parallel'],
                                   default='parallel', help='Parallel policy for fused Numba INT8 post-processing')
        device_parser.add_argument('--cpu-epilogue-min-work', type=int, default=262144,
                                   help='Minimum output elements (rows * columns) for automatic parallel Numba post-processing')
        device_parser.add_argument('--cpu-workspace', choices=['auto', 'off'], default='auto',
                                   help='Reuse CPU inference buffers to avoid repeated allocations; off retains allocating inference')
        device_parser.add_argument('--cpu-entropy-parallel', choices=['auto', 'serial', 'parallel'],
                                   default='parallel', help='Native CPU entropy stream parallelism (encode and decode)')
        device_parser.add_argument('--cpu-entropy-min-work', '--cpu-entropy-min-streams',
                                   dest='cpu_entropy_min_streams', type=int, default=256,
                                   help='Auto parallel threshold: stream/time pairs for encode, streams per step for decode')

    for profile_parser in (pc, pd):
        profile_parser.add_argument('--cpu-profile', type=Path,
                                    help='Write hierarchical CPU stage timings to this JSON file (adds instrumentation overhead)')
        profile_parser.add_argument('--cpu-profile-label', default=None,
                                    help='Label for the CPU profiling report')

    return parser


def main(argv=None):
    console = Console()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command in ('compress', 'decompress', 'view'):
        _validate_cpu_placement(args, parser)
    if getattr(args, 'cpu_socket_id', None) is not None and _SOCKET_BINDING is None:
        command = list(sys.argv[1:] if argv is None else argv)
        os.execv(sys.executable, [sys.executable, '-m', 'sigcomp.main', *command])
        raise RuntimeError('Failed to restart the CPU socket CLI')
    if args.command in ('compress', 'decompress', 'view'):
        _configure_device(args, parser)

    dispatch = {
        'compress': cmd_compress,
        'decompress': cmd_decompress,
        'inspect': cmd_inspect,
        'index': cmd_index,
        'view': cmd_view,
    }
    profile_path = getattr(args, 'cpu_profile', None)
    if profile_path is None:
        dispatch[args.command](args, console)
        return
    if args.device != 'cpu':
        parser.error('--cpu-profile requires --device cpu')
    destination = profile_path.expanduser().resolve()
    if destination.suffix.lower() != '.json':
        parser.error('--cpu-profile requires a separate .json report path')
    protected = [getattr(args, name, None) for name in
                 ('input', 'output', 'model_path', 'read_ids_file', 'metadata_header', 'index_file')]
    protected += [DEFAULT_QUANT_LOSSLESS_MODEL_PATH, DEFAULT_QUANT_LOSSY_MODEL_PATH]
    protected.append(str(args.input) + '.idx')
    if any(destination == Path(path).expanduser().resolve() for path in protected if path):
        parser.error('--cpu-profile must differ from input, output, model and selection/index files')
    from .cpu_runtime import get_cpu_runtime_info
    profiler = CPUProfiler(args.cpu_profile_label, metadata={
        'operation': 'compression' if args.command == 'compress' else 'decompression',
        'measurement_scope': 'CLI after argument/device setup, including loading, first-use JIT, metadata, signal I/O and codec stages; excludes imports and report writing',
        'configuration': {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
        'cpu_runtime': get_cpu_runtime_info(),
    })
    for path in protected:
        profiler.protect_path(path)
    try:
        with profiler.session(profiler.metadata['operation']):
            dispatch[args.command](args, console)
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            profiler.write(destination)
            print_profile_summary(profiler, sys.stderr)
            print(f'CPU profile saved to {destination}', file=sys.stderr)
        except Exception as exc:
            if not active_error:
                raise
            print(f'Could not save partial CPU profile: {exc}', file=sys.stderr)


if __name__ == "__main__":
    main()

