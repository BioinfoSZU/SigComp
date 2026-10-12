from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter


_CURRENT = ContextVar('sigcomp_cpu_profiler', default=None)


class _NoScope:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


_NO_SCOPE = _NoScope()


def current_profiler():
    return _CURRENT.get()


def scope(profiler, name):
    return _NO_SCOPE if profiler is None else profiler.measure(name)


def profiled_stage(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            profiler = _CURRENT.get()
            if profiler is None:
                return function(*args, **kwargs)
            with profiler.measure(name):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def profiled_method(name):
    def decorate(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            profiler = getattr(self, 'profiler', None)
            if profiler is None:
                return function(self, *args, **kwargs)
            with profiler.measure(name):
                return function(self, *args, **kwargs)
        return wrapped
    return decorate


def profile_iterable(iterable, name):
    profiler = _CURRENT.get()
    if profiler is None:
        return iterable

    def measured():
        iterator = iter(iterable)
        while True:
            with profiler.measure(name):
                try:
                    item = next(iterator)
                except StopIteration:
                    return
            yield item
    return measured()


class CPUProfiler:
    def __init__(self, label=None, metadata=None):
        self.label = label or 'SigComp CPU profile'
        self.metadata = dict(metadata or {})
        self.counters = {}
        self._stages = {}
        self._stack = []
        self._protected_paths = set()
        self.created_utc = datetime.now(timezone.utc).isoformat()
        self._started = None
        self._finished = None
        self.status = 'pending'
        self.error = None

    def add_count(self, name, value):
        self.counters[name] = self.counters.get(name, 0) + int(value)

    def protect_path(self, path):
        if path is not None:
            self._protected_paths.add(Path(path).expanduser().resolve())

    @contextmanager
    def measure(self, name):
        if not name or '/' in name:
            raise ValueError('Stage names must be nonempty and must not contain /')
        parent = self._stack[-1] if self._stack else None
        path = parent['path'] + '/' + name if parent else name
        frame = {'path': path, 'children': 0.0}
        self._stack.append(frame)
        started = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - started
            self._stack.pop()
            if parent is not None:
                parent['children'] += elapsed
            stage = self._stages.setdefault(path, {
                'path': path, 'name': name, 'calls': 0,
                'inclusive_seconds': 0.0, 'exclusive_seconds': 0.0,
                'min_seconds': elapsed, 'max_seconds': elapsed,
            })
            stage['calls'] += 1
            stage['inclusive_seconds'] += elapsed
            stage['exclusive_seconds'] += max(0.0, elapsed - frame['children'])
            stage['min_seconds'] = min(stage['min_seconds'], elapsed)
            stage['max_seconds'] = max(stage['max_seconds'], elapsed)

    @contextmanager
    def session(self, name):
        if self._started is not None:
            raise RuntimeError('Use a new CPUProfiler for each session')
        self._started = perf_counter()
        token = _CURRENT.set(self)
        self.status = 'running'
        try:
            with self.measure(name):
                yield self
        except BaseException as exc:
            self.status = 'failed'
            self.error = {'type': type(exc).__name__, 'message': str(exc)}
            raise
        else:
            self.status = 'completed'
        finally:
            self._finished = perf_counter()
            _CURRENT.reset(token)

    def report(self):
        elapsed = (0.0 if self._started is None else
                   (self._finished if self._finished is not None else perf_counter()) - self._started)
        return {
            'schema_version': 1, 'report_type': 'sigcomp_cpu_profile',
            'label': self.label, 'created_utc': self.created_utc,
            'status': self.status, 'error': self.error, 'wall_seconds': elapsed,
            'metadata': self.metadata, 'counters': dict(self.counters),
            'stages': [dict(self._stages[key]) for key in sorted(self._stages)],
            'notes': [
                'Synchronous wall-clock spans, not the sum of native worker CPU times.',
                'Parent inclusive times overlap children. Sum exclusive times only.',
                'Parent exclusive time includes orchestration and profiler overhead.',
                'CLI profiles include first-use JIT, loading and I/O after argument/device setup; Python imports and JSON writing are excluded.',
                'No ISA dispatch or AMX utilization is inferred from these timings.',
                'Profiling changes runtime; compare throughput with profiling disabled.',
            ],
        }

    def write(self, path):
        path = Path(path).expanduser()
        if path.resolve() in self._protected_paths:
            raise ValueError(f'Profile destination would overwrite a protected input: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                             dir=path.parent, prefix=path.name + '.',
                                             suffix='.tmp', delete=False) as handle:
                temporary = handle.name
                json.dump(self.report(), handle, indent=2, allow_nan=False)
                handle.write('\n')
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)


def print_profile_summary(profiler, stream, limit=12):
    report = profiler.report()
    wall = report['wall_seconds']
    print(f"\nCPU profile: {report['status']}, {wall:.3f} s (instrumented wall time)", file=stream)
    print(f"{'Exclusive s':>12} {'Wall %':>8} {'Calls':>9} Stage", file=stream)
    stages = sorted(report['stages'], key=lambda row: row['exclusive_seconds'], reverse=True)
    for row in stages[:limit]:
        seconds = row['exclusive_seconds']
        percent = 100.0 * seconds / wall if wall else 0.0
        print(f"{seconds:12.4f} {percent:8.2f} {row['calls']:9d} {row['path']}", file=stream)
