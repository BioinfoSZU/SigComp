from contextlib import contextmanager
from pathlib import Path
import struct
import tempfile
import time
import uuid

from .cpu_profiling import current_profiler, scope


_ENTRY = struct.Struct('<16sQ')


def _uuid_bytes(value):
    return value.bytes if isinstance(value, uuid.UUID) else uuid.UUID(str(value)).bytes


def _close_iterator(iterator):
    close = getattr(iterator, 'close', None)
    if close is not None:
        close()


class Pod5ReadPlan:

    def __init__(self, path, *, directory=None, reader_factory=None):
        self.path = Path(path).resolve(strict=True)
        if not self.path.is_file():
            raise ValueError('POD5 reopening requires one regular input file')
        self._identity = self._signature()
        self._reader_factory = reader_factory
        self._spool = tempfile.TemporaryFile(mode='w+b', prefix='sigcomp-pod5-plan-', dir=directory)
        self._ready = False
        self._scan_started = False
        self.count = 0
        self.stats = {'metadata_scan_seconds': 0.0, 'reader_open_seconds': 0.0,
                      'group_selection_seconds': 0.0, 'reader_close_seconds': 0.0,
                      'reader_opens': 0, 'completed_groups': 0}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._spool.close()

    def _signature(self):
        stat = self.path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def assert_unchanged(self):
        if self._signature() != self._identity:
            raise ValueError('POD5 input changed after read planning; compression is incomplete')

    @contextmanager
    def _timed(self, stage, key):
        started = time.perf_counter()
        try:
            with scope(current_profiler(), stage):
                yield
        finally:
            self.stats[key] += time.perf_counter() - started

    @contextmanager
    def _reader(self):
        self.assert_unchanged()
        factory = self._reader_factory
        if factory is None:
            from pod5 import Reader
            factory = Reader
        with self._timed('input.pod5.open', 'reader_open_seconds'):
            reader = factory(str(self.path))
        self.stats['reader_opens'] += 1
        profiler = current_profiler()
        if profiler is not None:
            profiler.add_count('pod5_reader_opens', 1)
        try:
            yield reader
        finally:
            with self._timed('input.pod5.close', 'reader_close_seconds'):
                reader.close()

    def scan(self, *, selection=None, max_reads=-1, on_read=None):
        if self._scan_started:
            raise RuntimeError('A POD5 read plan can only be scanned once')
        self._scan_started = True
        if max_reads != -1 and max_reads < 1:
            raise ValueError('max_reads must be -1 or positive')
        if selection is not None and max_reads > 0:
            selection = selection[:max_reads]
        pending = None if selection is None else {_uuid_bytes(value) for value in selection}
        if pending is not None and len(pending) != len(selection):
            raise ValueError('Duplicate UUID in POD5 selection')
        with self._timed('metadata.pod5_plan', 'metadata_scan_seconds'):
            with self._reader() as reader:
                iterator = None
                read = None
                try:
                    iterator = iter(reader.reads(selection=selection, missing_ok=False))
                    for read in iterator:
                        rid = _uuid_bytes(read.read_id)
                        if pending is not None:
                            if rid not in pending:
                                raise ValueError('Duplicate or unexpected selected POD5 read')
                            pending.remove(rid)
                        samples = int(read.num_samples)
                        if samples < 0:
                            raise ValueError('Negative POD5 sample count')
                        if on_read is not None:
                            on_read(read)
                        self._spool.write(_ENTRY.pack(rid, samples))
                        self.count += 1
                        # Exhaust selected iteration to catch duplicate/extra IDs.
                        if selection is None and 0 < max_reads <= self.count:
                            break
                finally:
                    read = None
                    if iterator is not None:
                        _close_iterator(iterator)
                    iterator = None
            if pending:
                raise ValueError('Missing selected POD5 reads during planning')
            self.assert_unchanged()
            self._spool.flush()
        self._ready = True
        profiler = current_profiler()
        if profiler is not None:
            profiler.add_count('pod5_planned_reads', self.count)
            profiler.add_count('pod5_plan_bytes', self.count * _ENTRY.size)

    def groups(self, *, chunk_len, chunk_capacity, chunk_size=0):
        if not self._ready:
            raise RuntimeError('Scan the POD5 read plan before reading groups')
        if chunk_len < 1 or chunk_capacity < 1 or chunk_size < 0:
            raise ValueError('Invalid POD5 flush geometry')
        self._spool.seek(0)
        group, chunks = [], 0
        for _ in range(self.count):
            raw = self._spool.read(_ENTRY.size)
            if len(raw) != _ENTRY.size:
                raise ValueError('Truncated POD5 read plan')
            rid, samples = _ENTRY.unpack(raw)
            group.append((rid, samples))
            tokens = max(1, samples - 1)
            chunks += (tokens + chunk_len - 1) // chunk_len
            if (len(group) >= chunk_size if chunk_size > 0 else chunks >= chunk_capacity):
                yield group
                group, chunks = [], 0
        if group:
            yield group

    def process_group(self, group, consumer):
        if not self._ready:
            raise RuntimeError('Scan the POD5 read plan before processing groups')
        if not group:
            return 0
        ids = [str(uuid.UUID(bytes=rid)) for rid, _ in group]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate POD5 read ID in a flush group')
        records, iterator, read = [], None, None
        with self._reader() as reader:
            try:
                with self._timed('input.pod5.select', 'group_selection_seconds'):
                    iterator = iter(reader.reads(selection=ids, missing_ok=False))
                    for read in iterator:
                        index = len(records)
                        if index >= len(group) or (_uuid_bytes(read.read_id), int(read.num_samples)) != group[index]:
                            raise ValueError('POD5 selection order, UUID or sample count changed')
                        records.append(read)
                    if len(records) != len(group):
                        raise ValueError('Missing POD5 reads in a flush group')
                    read = None
                    _close_iterator(iterator)
                    iterator = None
                consumer(records)
            finally:
                read = None
                records.clear()
                if iterator is not None:
                    _close_iterator(iterator)
                iterator = None
        self.assert_unchanged()
        self.stats['completed_groups'] += 1
        return len(group)

    def summary(self):
        return (f"POD5 reopen: {self.stats['completed_groups']:,} flushes; "
                f"{self.stats['reader_opens']:,} reader opens (including metadata); "
                f"metadata scan {self.stats['metadata_scan_seconds']:.3f}s; "
                f"open {self.stats['reader_open_seconds']:.3f}s, "
                f"group selection {self.stats['group_selection_seconds']:.3f}s, "
                f"close {self.stats['reader_close_seconds']:.3f}s; "
                f"temporary plan {self.count * _ENTRY.size:,} bytes. "
                "Scan overlaps open/close totals; timings exclude signal decoding/compression.")
