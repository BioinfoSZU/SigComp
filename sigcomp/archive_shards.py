from bisect import bisect_left
import argparse
from dataclasses import dataclass
import errno
import itertools
import os
from pathlib import Path
import pickle
import struct
import tempfile
import uuid

import numcodecs


MAGIC = b"SIG2"
IDX_MAGIC = b"SIDX"
REC_FMT = struct.Struct("<16sddBBdIIIQ")
REC_FMT_BLOW5 = struct.Struct("<16s")
SPLIT_HDR_FMT = struct.Struct("<hII")
U32 = struct.Struct("<I")
INDEX_ENTRY = struct.Struct("<16sQ")
_COPY_BUFFER_SIZE = 8 * 1024 * 1024
_COPY_SYSCALL_SIZE = 1024 * 1024 * 1024
_NON_SEMANTIC_FIELDS = frozenset(("num_reads", "command", "datetime"))


@dataclass(frozen=True, slots=True)
class RecordSpan:
    read_id: bytes
    start: int
    end: int
    num_samples: int


@dataclass(frozen=True, slots=True)
class ArchiveScan:
    path: str
    header: dict
    data_start: int
    file_size: int
    mtime_ns: int
    records: tuple

    @property
    def num_reads(self):
        return len(self.records)

    @property
    def num_samples(self):
        return sum(record.num_samples for record in self.records)


@dataclass(frozen=True, slots=True)
class ShardPlan:
    start_read: int
    stop_read: int
    start_offset: int
    end_offset: int
    read_ids: tuple
    num_samples: int

    @property
    def num_reads(self):
        return self.stop_read - self.start_read


@dataclass(frozen=True, slots=True)
class MergeResult:
    path: str
    num_reads: int
    num_samples: int
    file_size: int
    data_start: int
    index_path: str | None


def _read_exact(stream, size, description):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(f"Truncated SIG2 {description}")
    return data


def _validate_header(header):
    if not isinstance(header, dict):
        raise ValueError("SIG2 global metadata must be a dictionary")
    if header.get("format_version", 2) != 2:
        raise ValueError("Only SIG2 format version 2 can be sharded")
    if header.get("source_format", "pod5") not in ("pod5", "blow5"):
        raise ValueError("Unsupported SIG2 source format")
    count = header.get("num_reads")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("SIG2 num_reads must be a nonnegative integer")


def _read_header(stream):
    if _read_exact(stream, 4, "magic") != MAGIC:
        raise ValueError("Not a SIG2 archive")
    size, = U32.unpack(_read_exact(stream, U32.size, "header length"))
    file_size = os.fstat(stream.fileno()).st_size
    if size > file_size - stream.tell():
        raise ValueError("SIG2 header extends beyond end of file")
    payload = _read_exact(stream, size, "header")
    header = pickle.loads(numcodecs.zstd.decompress(payload))
    _validate_header(header)
    return header, stream.tell()


def read_archive_header(path):
    with open(path, "rb") as stream:
        return _read_header(stream)


def write_archive_header(stream, header):
    _validate_header(header)
    payload = numcodecs.zstd.compress(
        pickle.dumps(header, protocol=pickle.HIGHEST_PROTOCOL), level=22)
    if len(payload) > 0xFFFFFFFF:
        raise ValueError("SIG2 metadata exceeds the uint32 header limit")
    _write_all(stream, MAGIC)
    _write_all(stream, U32.pack(len(payload)))
    _write_all(stream, payload)
    return stream.tell()


write_header = write_archive_header


def _skip_blob(stream, file_size, description):
    size, = U32.unpack(_read_exact(stream, U32.size, description + " length"))
    position = stream.tell() + size
    if position > file_size:
        raise ValueError(f"SIG2 {description} extends beyond end of file")
    stream.seek(position)


def _assert_unchanged(scan):
    current = os.stat(scan.path)
    if current.st_size != scan.file_size or current.st_mtime_ns != scan.mtime_ns:
        raise ValueError(f"Archive changed after its shard plan was built: {scan.path}")


def scan_archive(path):
    path = str(Path(path).resolve())
    records = []
    with open(path, "rb") as stream:
        before = os.fstat(stream.fileno())
        header, data_start = _read_header(stream)
        is_blow5 = header.get("source_format", "pod5") == "blow5"
        pores = set(header.get("pore_dict", {}).values())
        run_infos = header.get("run_info_dict", {})
        for _ in range(header["num_reads"]):
            start = stream.tell()
            if is_blow5:
                read_id = _read_exact(stream, REC_FMT_BLOW5.size, "BLOW5 read record")
                _skip_blob(stream, before.st_size, "BLOW5 read metadata")
            else:
                record = REC_FMT.unpack(_read_exact(stream, REC_FMT.size, "POD5 read record"))
                read_id = record[0]
                if record[6] not in pores or record[8] not in run_infos:
                    raise ValueError("SIG2 record references a missing pore or run-info entry")
            _, num_samples, num_small = SPLIT_HDR_FMT.unpack(
                _read_exact(stream, SPLIT_HDR_FMT.size, "split header"))
            if num_samples < 1 or num_small > num_samples - 1:
                raise ValueError("Invalid sample or token count in SIG2 read record")
            for description in ("small-symbol payload", "large-value payload", "position payload"):
                _skip_blob(stream, before.st_size, description)
            records.append(RecordSpan(read_id, start, stream.tell(), num_samples))
        if stream.tell() != before.st_size:
            raise ValueError("SIG2 read count disagrees with trailing file data")
        after = os.fstat(stream.fileno())
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            raise ValueError("Archive changed during metadata scanning")
    return ArchiveScan(path, header, data_start, before.st_size,
                       before.st_mtime_ns, tuple(records))


def plan_archive_shards(scan_or_path, workers, max_reads=0):
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not isinstance(max_reads, int):
        raise ValueError("max_reads must be an integer")
    scan = scan_or_path if isinstance(scan_or_path, ArchiveScan) else scan_archive(scan_or_path)
    count = min(scan.num_reads, max_reads) if max_reads > 0 else scan.num_reads
    if count == 0:
        return ()
    parts = min(workers, count)
    prefix = [0]
    for record in scan.records[:count]:
        prefix.append(prefix[-1] + record.num_samples)
    boundaries = [0]
    for i in range(1, parts):
        earliest = boundaries[-1] + 1
        latest = count - (parts - i)
        target = (prefix[-1] * i + parts - 1) // parts
        boundary = bisect_left(prefix, target, earliest, latest + 1)
        boundaries.append(min(boundary, latest))
    boundaries.append(count)
    return tuple(
        ShardPlan(start, end, scan.records[start].start, scan.records[end - 1].end,
                  tuple(r.read_id for r in scan.records[start:end]), prefix[end] - prefix[start])
        for start, end in zip(boundaries, boundaries[1:])
    )


def _write_all(stream, data):
    remaining = memoryview(data)
    while remaining:
        written = stream.write(remaining)
        if written is None or written <= 0:
            raise OSError("Unable to write archive data")
        remaining = remaining[written:]


def copy_range(source, destination, start, length, buffer_size=_COPY_BUFFER_SIZE):
    if start < 0 or length < 0 or buffer_size < 1:
        raise ValueError("Invalid copy range or buffer size")
    if start + length > os.fstat(source.fileno()).st_size:
        raise ValueError("Copy range extends beyond source file")
    destination.flush()
    target_start = destination.tell()
    copied = 0
    kernel_copy = getattr(os, "copy_file_range", None)
    fallback_errors = {errno.EXDEV, errno.EINVAL, errno.ENOSYS, errno.EPERM,
                       getattr(errno, "EOPNOTSUPP", errno.EINVAL),
                       getattr(errno, "ENOTSUP", errno.EINVAL)}
    while kernel_copy is not None and copied < length:
        try:
            transferred = kernel_copy(
                source.fileno(), destination.fileno(),
                min(length - copied, _COPY_SYSCALL_SIZE),
                offset_src=start + copied, offset_dst=target_start + copied)
        except InterruptedError:
            continue
        except OSError as exc:
            if exc.errno not in fallback_errors:
                raise
            break
        if transferred == 0:
            break
        copied += transferred
    source.seek(start + copied)
    destination.seek(target_start + copied)
    if copied < length:
        buffer = bytearray(min(buffer_size, length - copied))
        view = memoryview(buffer)
        while copied < length:
            count = source.readinto(view[:min(len(buffer), length - copied)])
            if not count:
                raise ValueError("Source archive was truncated during payload copying")
            _write_all(destination, view[:count])
            copied += count
    return copied


def _ensure_absent(path):
    if os.path.lexists(path):
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def _make_staging(path):
    path = Path(path)
    fd, staged = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    return os.fdopen(fd, "wb"), staged


def _publish_no_overwrite(staged, destination):
    try:
        os.link(staged, destination)
    except OSError as exc:
        if isinstance(exc, FileExistsError) or os.name != "nt":
            raise
        os.rename(staged, destination)
    else:
        os.unlink(staged)


def _discard_staging(path):
    if path is not None:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def _finish_staging(stream):
    stream.flush()
    os.fsync(stream.fileno())


def _write_index_entries(stream, count, entries):
    if count > 0xFFFFFFFF:
        raise ValueError("SIDX supports at most 2**32-1 reads")
    _write_all(stream, IDX_MAGIC + U32.pack(count))
    written = 0
    for read_id, offset in entries:
        _write_all(stream, INDEX_ENTRY.pack(read_id, offset))
        written += 1
    if written != count:
        raise ValueError("Index entry count disagrees with archive read count")


def write_index(scan_or_path, index_path):
    scan = scan_or_path if isinstance(scan_or_path, ArchiveScan) else scan_archive(scan_or_path)
    _assert_unchanged(scan)
    destination = os.path.abspath(os.fspath(index_path))
    _ensure_absent(destination)
    stream, staged = _make_staging(destination)
    try:
        with stream:
            _write_index_entries(stream, scan.num_reads,
                                 ((record.read_id, record.start) for record in scan.records))
            _finish_staging(stream)
        _assert_unchanged(scan)
        _publish_no_overwrite(staged, destination)
    finally:
        _discard_staging(staged)
    return destination


def _metadata_signature(header):
    metadata = tuple((key, header[key]) for key in sorted(header)
                     if key not in _NON_SEMANTIC_FIELDS)
    return pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL)


def _read_id_bytes(value):
    if isinstance(value, bytes):
        if len(value) != 16:
            raise ValueError("A read ID must contain exactly 16 bytes")
        return value
    if isinstance(value, uuid.UUID):
        return value.bytes
    return uuid.UUID(str(value)).bytes


def merge_archives(paths, output, global_dict=None, write_index=False,
                   expected_read_ids=None, index_path=None):
    scans = tuple(path if isinstance(path, ArchiveScan) else scan_archive(path) for path in paths)
    if not scans and global_dict is None:
        raise ValueError("An empty merge needs canonical global metadata")
    if global_dict is None:
        header = dict(scans[0].header)
    elif isinstance(global_dict, (str, os.PathLike)):
        header = dict(read_archive_header(global_dict)[0])
    else:
        header = dict(global_dict)
    _validate_header(header)
    total_reads = sum(scan.num_reads for scan in scans)
    total_samples = sum(scan.num_samples for scan in scans)
    if global_dict is not None and header["num_reads"] != total_reads:
        raise ValueError("Merged shard read count disagrees with canonical metadata")
    signature = _metadata_signature(header)
    seen = set()
    for scan in scans:
        if _metadata_signature(scan.header) != signature:
            raise ValueError(f"Shard metadata differs from canonical tables/model settings: {scan.path}")
        _assert_unchanged(scan)
        for record in scan.records:
            if record.read_id in seen:
                raise ValueError("Duplicate read ID across merged shards")
            seen.add(record.read_id)
    if expected_read_ids is not None:
        actual = (record.read_id for scan in scans for record in scan.records)
        expected = (_read_id_bytes(value) for value in expected_read_ids)
        missing = object()
        for observed, wanted in itertools.zip_longest(actual, expected, fillvalue=missing):
            if observed != wanted:
                raise ValueError("Shard read IDs/count/order differ from the requested input selection")
    header["num_reads"] = total_reads
    destination = os.path.abspath(os.fspath(output))
    _ensure_absent(destination)
    index_destination = None
    if write_index or index_path is not None:
        index_destination = os.path.abspath(os.fspath(index_path or (destination + ".idx")))
        if index_destination == destination:
            raise ValueError("Archive and index output paths must differ")
        _ensure_absent(index_destination)
    archive_stream, staged_archive = _make_staging(destination)
    staged_index = None
    index_stream = None
    try:
        if index_destination is not None:
            index_stream, staged_index = _make_staging(index_destination)
        with archive_stream:
            data_start = write_archive_header(archive_stream, header)
            if index_stream is not None:
                if total_reads > 0xFFFFFFFF:
                    raise ValueError("SIDX supports at most 2**32-1 reads")
                _write_all(index_stream, IDX_MAGIC + U32.pack(total_reads))
            for scan in scans:
                _assert_unchanged(scan)
                target_start = archive_stream.tell()
                if index_stream is not None:
                    for record in scan.records:
                        offset = target_start + record.start - scan.data_start
                        _write_all(index_stream, INDEX_ENTRY.pack(record.read_id, offset))
                with open(scan.path, "rb") as source:
                    copy_range(source, archive_stream, scan.data_start, scan.file_size - scan.data_start)
                _assert_unchanged(scan)
            file_size = archive_stream.tell()
            _finish_staging(archive_stream)
        if index_stream is not None:
            _finish_staging(index_stream)
            index_stream.close()
            index_stream = None
        _publish_no_overwrite(staged_archive, destination)
        if staged_index is not None:
            try:
                _publish_no_overwrite(staged_index, index_destination)
            except OSError as exc:
                raise OSError(f"Archive is complete at {destination}, but index publication failed: {exc}") from exc
    finally:
        if not archive_stream.closed:
            archive_stream.close()
        if index_stream is not None:
            index_stream.close()
        _discard_staging(staged_archive)
        _discard_staging(staged_index)
    return MergeResult(destination, total_reads, total_samples, file_size, data_start, index_destination)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Merge ordered SigComp worker archives without decoding their payloads")
    commands = parser.add_subparsers(dest="command", required=True)
    merge = commands.add_parser("merge", help="Create one ordinary SIG2 archive")
    merge.add_argument("-o", "--output", required=True, help="New output archive (never overwritten)")
    merge.add_argument("--write-index", action="store_true", help="Also create OUTPUT.idx")
    merge.add_argument("--metadata-header", help="Optional shared canonical SIG2 header-only file")
    merge.add_argument("shards", nargs="+", help="Worker archives in the required output read order")
    args = parser.parse_args(argv)
    try:
        result = merge_archives(args.shards, args.output,
                                global_dict=args.metadata_header, write_index=args.write_index)
    except (OSError, ValueError, pickle.PickleError) as exc:
        parser.exit(1, f"Merge failed: {exc}\n")
    print(f"Merged {result.num_reads:,} reads ({result.file_size:,} bytes) into {result.path}")
    if result.index_path is not None:
        print(f"Index: {result.index_path}")


if __name__ == "__main__":
    main()
