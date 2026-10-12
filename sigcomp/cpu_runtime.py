import importlib
import os
from pathlib import Path
import platform


_ENV_KEYS = (
    'ONEDNN_MAX_CPU_ISA', 'DNNL_MAX_CPU_ISA', 'ONEDNN_CPU_ISA_HINTS',
    'ONEDNN_VERBOSE', 'DNNL_VERBOSE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMBA_NUM_THREADS', 'OMP_PROC_BIND', 'OMP_PLACES', 'KMP_AFFINITY',
)


def _safe_call(function, errors, label):
    if function is None:
        return None
    try:
        return function()
    except Exception as exc:
        errors[label] = f'{type(exc).__name__}: {exc}'
        return None


def _cpu_list(text):
    """Parse Linux CPU lists such as '0-15,64-79'."""
    result = set()
    for part in text.strip().split(','):
        bounds = part.split('-')
        first = int(bounds[0])
        last = int(bounds[-1])
        if len(bounds) > 2 or first < 0 or last < first:
            raise ValueError('Invalid CPU list')
        result.update(range(first, last + 1))
    return result


def get_physical_cpu_count():
    affinity = None
    if hasattr(os, 'sched_getaffinity'):
        try:
            affinity = set(os.sched_getaffinity(0))
        except OSError:
            pass
    try:
        cpus = affinity if affinity is not None else _cpu_list(
            Path('/sys/devices/system/cpu/online').read_text())
        siblings = {
            frozenset(_cpu_list(Path(
                f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list'
            ).read_text())) for cpu in cpus
        }
        if siblings and all(siblings):
            return len(siblings)
    except (OSError, ValueError):
        pass

    try:
        cores = {}
        for record in Path('/proc/cpuinfo').read_text().strip().split('\n\n'):
            fields = dict(line.split(':', 1) for line in record.splitlines() if ':' in line)
            fields = {key.strip(): value.strip() for key, value in fields.items()}
            cpu = int(fields['processor'])
            if affinity is None or cpu in affinity:
                cores[cpu] = (int(fields['physical id']), int(fields['core id']))
        if cores and (affinity is None or set(cores) == affinity):
            return len(set(cores.values()))
    except (OSError, ValueError, KeyError):
        pass

    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
        if physical and (affinity is None or (logical and len(affinity) == logical)):
            return int(physical)
    except (ImportError, OSError, NotImplementedError):
        pass
    raise ValueError('Cannot determine physical CPU cores from OS topology; '
                     'set --cpu-threads explicitly (or install psutil for '
                     'system-wide physical-core detection)')


def get_cpu_runtime_info():
    errors = {}
    model, flags = platform.processor(), []
    try:
        first_processor = Path('/proc/cpuinfo').read_text().split('\n\n', 1)[0]
        fields = dict(line.split(':', 1) for line in first_processor.splitlines() if ':' in line)
        fields = {k.strip(): v.strip() for k, v in fields.items()}
        model = fields.get('model name', model)
        flags = sorted(set(fields.get('flags', fields.get('Features', '')).split()))
    except OSError:
        pass
    affinity = _safe_call(lambda: sorted(os.sched_getaffinity(0)), errors, 'affinity') if hasattr(os, 'sched_getaffinity') else None
    report = {
        'cpu_model': model, 'architecture': platform.machine(),
        'kernel': platform.release(), 'logical_cpu_count': os.cpu_count(),
        'affinity_cpu_ids': affinity, 'affinity_cpu_count': len(affinity) if affinity is not None else None,
        'linux_first_processor_flags': flags,
        'linux_amx_tile_flag': 'amx_tile' in flags if flags else None,
        'linux_amx_int8_flag': 'amx_int8' in flags if flags else None,
        'environment': {key: os.environ.get(key) for key in _ENV_KEYS},
        'amx_dispatch_verified': None,
        'dispatch_note': 'Capabilities are eligibility hints. GEMM dispatch requires execution evidence; no GEMM is run while collecting metadata.',
        'errors': errors,
    }
    try:
        torch = importlib.import_module('torch')
    except Exception as exc:
        errors['torch_import'] = f'{type(exc).__name__}: {exc}'
        return report
    report['torch_version'] = str(torch.__version__)
    report['torch_build'] = _safe_call(torch.__config__.show, errors, 'torch_build')
    report['mkldnn_available'] = bool(torch.backends.mkldnn.is_available())
    report['mkldnn_enabled'] = bool(torch.backends.mkldnn.enabled)
    report['torch_threads'] = torch.get_num_threads()
    cpu = getattr(torch, 'cpu', None)
    for key, method in (('torch_vnni_supported', '_is_vnni_supported'),
                        ('torch_amx_tile_supported', '_is_amx_tile_supported')):
        report[key] = _safe_call(getattr(cpu, method, None), errors, key)
    caps = _safe_call(getattr(cpu, 'get_capabilities', None), errors, 'torch_cpu_capabilities')
    report['torch_cpu_capabilities'] = dict(caps) if caps is not None else None
    return report
