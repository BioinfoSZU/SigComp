import argparse
import ctypes
import json
import os
from pathlib import Path
import sys


def parse_cpu_list(text):
    result = set()
    for part in text.strip().split(','):
        bounds = part.split('-')
        first, last = int(bounds[0]), int(bounds[-1])
        if len(bounds) > 2 or first < 0 or last < first:
            raise ValueError('Invalid CPU/node list')
        result.update(range(first, last + 1))
    return result


def format_cpu_list(values):
    values = sorted(set(values))
    ranges = []
    for value in values:
        if ranges and value == ranges[-1][1] + 1:
            ranges[-1][1] = value
        else:
            ranges.append([value, value])
    return ','.join(str(start) if start == end else f'{start}-{end}'
                    for start, end in ranges)


def _require_linux():
    if not sys.platform.startswith('linux') or not hasattr(os, 'sched_setaffinity'):
        raise ValueError('--cpu-socket-id requires Linux CPU/NUMA topology and affinity support')


def discover_sockets():
    _require_linux()
    allowed = set(os.sched_getaffinity(0))
    groups = {}
    for cpu in sorted(allowed):
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}')
        try:
            socket = int((base / 'topology/physical_package_id').read_text().strip())
            core = int((base / 'topology/core_id').read_text().strip())
            if socket < 0 or core < 0:
                raise ValueError('Unknown socket/core ID')
            nodes = [int(path.name[4:]) for path in base.glob('node[0-9]*')
                     if path.name[4:].isdigit()]
            if len(nodes) != 1:
                raise ValueError('Expected one NUMA node for this CPU')
            node = nodes[0]
            siblings = parse_cpu_list((base / 'topology/thread_siblings_list').read_text())
            if cpu not in siblings:
                raise ValueError('CPU missing from its own sibling list')
        except (OSError, ValueError) as exc:
            raise ValueError(f'Cannot read complete topology for allowed CPU {cpu}: {exc}') from exc
        key = (socket, tuple(sorted(siblings)))
        if key not in groups:
            groups[key] = {'socket': socket, 'node': node, 'core': core, 'cpu': cpu}
        elif groups[key]['node'] != node:
            raise ValueError(f'Inconsistent NUMA nodes for SMT siblings of CPU {cpu}')

    sockets = {}
    for record in groups.values():
        item = sockets.setdefault(record['socket'], {'cpu_ids': [], 'nodes': {}})
        item['cpu_ids'].append(record['cpu'])
        item['nodes'].setdefault(record['node'], []).append(record['cpu'])
    for item in sockets.values():
        item['cpu_ids'].sort()
        item['cpu_list'] = format_cpu_list(item['cpu_ids'])
        item['physical_core_count'] = len(item['cpu_ids'])
        item['numa_nodes'] = sorted(item['nodes'])
        item['nodes'] = {str(node): sorted(cpus) for node, cpus in sorted(item['nodes'].items())}
    if not sockets:
        raise ValueError('No physical CPU cores are available in process affinity')
    return sockets


def _allowed_memory_nodes():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('Mems_allowed_list:'):
            return parse_cpu_list(line.split(':', 1)[1])
    raise ValueError('Cannot determine the process Mems_allowed_list')


def plan_socket(socket_id, *, memory_policy='bind', cpu_threads=None, numba_threads=None,
                sockets=None):
    sockets = discover_sockets() if sockets is None else sockets
    if socket_id not in sockets:
        raise ValueError(f'Socket {socket_id} is not available in process affinity; '
                         f'available socket IDs: {sorted(sockets)}')
    if memory_policy not in ('bind', 'inherit'):
        raise ValueError('CPU memory policy must be bind or inherit')
    plan = dict(sockets[socket_id], socket_id=socket_id, memory_policy=memory_policy)
    count = plan['physical_core_count']
    plan['cpu_threads'] = count if cpu_threads is None else cpu_threads
    plan['cpu_numba_threads'] = plan['cpu_threads'] if numba_threads is None else numba_threads
    for key in ('cpu_threads', 'cpu_numba_threads'):
        if not 1 <= plan[key] <= count:
            raise ValueError(f'--{key.replace("_", "-")} must be between 1 and {count} '
                             f'for socket {socket_id}; omitting it uses available physical cores')
    if memory_policy == 'bind':
        nodes = set(plan['numa_nodes']) & _allowed_memory_nodes()
        # Exclude memoryless nodes when the kernel exposes that information.
        memory_nodes = Path('/sys/devices/system/node/has_memory')
        if memory_nodes.exists():
            nodes &= parse_cpu_list(memory_nodes.read_text())
        if not nodes:
            raise ValueError(f'No allowed memory-bearing NUMA nodes on socket {socket_id}')
        plan['memory_nodes'] = sorted(nodes)
    else:
        plan['memory_nodes'] = None
    return plan


class _MemoryPolicy:
    MPOL_BIND = 2

    def __init__(self, nodes):
        try:
            self.lib = ctypes.CDLL('libnuma.so.1', use_errno=True)
        except OSError as exc:
            raise RuntimeError('NUMA memory binding requires libnuma.so.1 '
                               '(Ubuntu/Debian: libnuma1; RHEL/Rocky: numactl-libs). '
                               'Use --cpu-memory-policy inherit explicitly for CPU affinity only.') from exc
        # libnuma requires this initialization check before any other API.
        self.lib.numa_available.argtypes = []
        self.lib.numa_available.restype = ctypes.c_int
        if self.lib.numa_available() < 0:
            raise RuntimeError('NUMA policy support is unavailable or prohibited by the kernel/container. '
                               'Use --cpu-memory-policy inherit explicitly for CPU affinity only.')
        self.lib.numa_num_possible_nodes.argtypes = []
        self.lib.numa_num_possible_nodes.restype = ctypes.c_int
        possible_nodes = self.lib.numa_num_possible_nodes()
        if not 1 <= possible_nodes <= 1048576 or not nodes:
            raise RuntimeError('Invalid libnuma node-mask size')
        self.maxnode = max(possible_nodes, max(nodes) + 1)
        if self.maxnode > 1048576 or min(nodes) < 0:
            raise RuntimeError('Invalid requested NUMA node IDs')
        self.word_bits = ctypes.sizeof(ctypes.c_ulong) * 8
        self.mask_type = ctypes.c_ulong * ((self.maxnode + self.word_bits - 1) // self.word_bits)
        self.lib.set_mempolicy.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong]
        self.lib.set_mempolicy.restype = ctypes.c_int
        self.lib.get_mempolicy.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_ulong),
                                          ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong]
        self.lib.get_mempolicy.restype = ctypes.c_int

    @staticmethod
    def _check(result, operation):
        if result != 0:
            code = ctypes.get_errno()
            raise RuntimeError(f'{operation} failed: {os.strerror(code)} (errno {code}). '
                               'The kernel/container may prohibit NUMA policy calls; '
                               'use --cpu-memory-policy inherit explicitly for CPU affinity only.')

    def get(self):
        mode, mask = ctypes.c_int(), self.mask_type()
        self._check(self.lib.get_mempolicy(ctypes.byref(mode), mask, self.maxnode, None, 0),
                    'get_mempolicy')
        return mode.value, mask

    def set(self, mode, mask):
        # Default/local policies require an empty nodemask and maxnode=0.
        if not any(mask):
            mask, count = None, 0
        else:
            count = self.maxnode
        self._check(self.lib.set_mempolicy(mode, mask, count), 'set_mempolicy')

    def bind(self, nodes):
        mask = self.mask_type()
        for node in nodes:
            mask[node // self.word_bits] |= 1 << (node % self.word_bits)
        self.set(self.MPOL_BIND, mask)
        mode, actual = self.get()
        if mode != self.MPOL_BIND or list(actual) != list(mask):
            raise RuntimeError('Kernel NUMA policy did not match the requested socket memory binding')


def apply_socket(plan):
    _require_linux()
    if len(list(Path('/proc/self/task').iterdir())) != 1:
        raise RuntimeError('Socket binding requires a fresh single-threaded Python startup; '
                           'a startup hook has already created worker threads')
    previous_cpus = set(os.sched_getaffinity(0))
    cpus = set(plan['cpu_ids'])
    if not cpus or not cpus <= previous_cpus:
        raise RuntimeError('CPU allocation changed while preparing socket binding; restart the command')
    memory = _MemoryPolicy(plan['memory_nodes']) if plan['memory_policy'] == 'bind' else None
    previous_memory = memory.get() if memory is not None else None
    try:
        os.sched_setaffinity(0, cpus)
        if set(os.sched_getaffinity(0)) != cpus:
            raise RuntimeError('Kernel CPU affinity did not match the selected physical cores')
        if memory is not None:
            memory.bind(plan['memory_nodes'])
    except Exception as exc:
        restoration_errors = []
        if previous_memory is not None:
            try:
                memory.set(*previous_memory)
            except Exception as restore_exc:
                restoration_errors.append(str(restore_exc))
        try:
            os.sched_setaffinity(0, previous_cpus)
        except OSError as restore_exc:
            restoration_errors.append(str(restore_exc))
        if restoration_errors:
            raise RuntimeError(f'{exc}; policy restoration failed: {restoration_errors}') from exc
        raise
    return dict(plan, affinity_applied=True, memory_binding_applied=memory is not None,
                application_stage='before_numpy_torch_numba_imports',
                cpu_affinity_scope='one allowed logical CPU per physical core; workers may migrate within mask')


def socket_environment(plan):
    changes = {key: str(plan['cpu_threads']) for key in
               ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS')}
    changes['NUMBA_NUM_THREADS'] = str(plan['cpu_numba_threads'])
    changes['OMP_PROC_BIND'] = 'false'
    # These inherited settings can cause runtimes to replace the socket mask.
    removed = {key: os.environ.pop(key) for key in
               ('OMP_PLACES', 'KMP_AFFINITY', 'GOMP_CPU_AFFINITY', 'KMP_HW_SUBSET') if key in os.environ}
    previous = {key: os.environ.get(key) for key in changes}
    os.environ.update(changes)
    return {'set': changes, 'previous': previous, 'removed': removed}


def bootstrap_socket(argv):
    if not argv or argv[0] not in ('compress', 'decompress', 'view'):
        return None
    if '--help' in argv or '-h' in argv:
        return None
    if not any(value == '--cpu-socket-id' or value.startswith('--cpu-socket-id=') for value in argv):
        return None
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--cpu-socket-id', type=int, required=True,
                        help='Physical Linux socket ID; select one allowed logical CPU per physical core before native libraries are imported')
    parser.add_argument('--cpu-memory-policy', choices=['bind', 'inherit'], default='bind',
                        help='bind restricts new allocations to allowed NUMA nodes of the selected socket (requires libnuma); inherit applies CPU affinity only')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'gpu'], required=True,
                        help='Codec device; socket placement requires cpu (gpu is an alias for cuda)')
    parser.add_argument('--cpu-threads', type=int,
                        help="PyTorch thread count; defaults to the selected socket's available physical cores")
    parser.add_argument('--cpu-numba-threads', type=int,
                        help='Numba thread count for integer and entropy kernels; defaults to the resolved CPU thread count')
    args, _ = parser.parse_known_args(argv[1:])
    if args.device != 'cpu':
        parser.error('--cpu-socket-id requires --device cpu')
    try:
        plan = plan_socket(args.cpu_socket_id, memory_policy=args.cpu_memory_policy,
                           cpu_threads=args.cpu_threads, numba_threads=args.cpu_numba_threads)
        result = apply_socket(plan)
        result['thread_environment'] = socket_environment(plan)
    except (OSError, ValueError, RuntimeError, AttributeError) as exc:
        parser.error(str(exc))
    memory_text = ('bind ' + format_cpu_list(result['memory_nodes'])
                   if result['memory_binding_applied'] else 'inherited (CPU affinity only)')
    print(f"CPU socket {args.cpu_socket_id}: CPUs {result['cpu_list']}; "
          f"{result['physical_core_count']} physical cores; memory {memory_text}; "
          f"Torch/Numba workers {result['cpu_threads']}/{result['cpu_numba_threads']}", file=sys.stderr)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='List available Linux socket/NUMA physical CPU topology')
    parser.add_argument('--list', action='store_true', required=True, help='Print topology JSON; does not bind or import the codec')
    parser.parse_args(argv)
    try:
        print(json.dumps(discover_sockets(), indent=2))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
