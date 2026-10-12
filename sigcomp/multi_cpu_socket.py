import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

from . import multi_gpu as files
from .cpu_socket import discover_sockets, plan_socket


_CPU_OPTION_HELP = {
    'cpu_inference_batch_size': 'Maximum context rows per CPU encoding inference call per socket; decoding uses --cpu-batch-size',
    'cpu_backend': 'Integer MLP backend: int_mm uses PyTorch INT8 matrix multiplication, numba uses native integer kernels, auto allows fallback',
    'cpu_epilogue': 'Integer matrix-product post-processing: numba uses fused kernels, torch uses the eager reference path, auto prefers Numba with a Torch fallback',
    'cpu_epilogue_parallel': 'Parallel policy for fused Numba post-processing: auto uses the work threshold, serial disables parallelism, parallel requests it',
    'cpu_epilogue_min_work': 'Minimum output elements (rows * columns) for automatic parallel Numba post-processing',
    'cpu_workspace': 'auto reuses CPU inference buffers to reduce allocations; off uses allocating inference',
    'cpu_entropy_parallel': 'Entropy-coder parallel policy per socket: auto uses the work threshold, serial disables parallelism, parallel requests it',
}


_CPU_OPTIONS = (
    ('cpu_inference_batch_size', 4096, None),
    ('cpu_backend', 'int_mm', ('auto', 'int_mm', 'numba')),
    ('cpu_epilogue', 'numba', ('auto', 'numba', 'torch')),
    ('cpu_epilogue_parallel', 'parallel', ('auto', 'serial', 'parallel')),
    ('cpu_epilogue_min_work', 262144, None),
    ('cpu_workspace', 'auto', ('auto', 'off')),
    ('cpu_entropy_parallel', 'parallel', ('auto', 'serial', 'parallel')),
)


def _socket_ids(value):
    try:
        result = [int(part.strip()) for part in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('use comma-separated physical socket IDs, e.g. 0,1') from exc
    if not result or min(result) < 0 or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError('socket IDs must be distinct and nonnegative')
    return result


def _select_sockets(requested, topology):
    selected = sorted(topology) if requested is None else list(requested)
    if not selected or any(socket not in topology for socket in selected):
        raise ValueError(f'--sockets must select available physical socket IDs: {sorted(topology)}')
    return selected


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('compress', 'decompress'):
        sub = commands.add_parser(name, allow_abbrev=False,
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  help=('Compress reads into one SIG2 archive' if name == 'compress' else 'Decompress into independent files in OUTPUT.parts/'))
        sub.add_argument('-i', '--input', type=Path, required=True,
                         help='One input POD5/BLOW5/SLOW5 file for compression, or SIG2 archive for decompression; directories are not accepted')
        sub.add_argument('-o', '--output', type=Path, required=True,
                         help='Compression: new SIG2 archive path. Decompression: output name with .pod5/.blow5/.slow5/.bin extension; files are retained in OUTPUT.parts/')
        sub.add_argument('--sockets', type=_socket_ids, help='Physical socket IDs; omit for all available sockets')
        sub.add_argument('--cpu-memory-policy', choices=('bind', 'inherit'), default='bind',
                         help='bind restricts new allocations to allowed NUMA nodes on each selected socket (requires libnuma); inherit leaves the existing memory policy unchanged')
        sub.add_argument('--cpu-threads', type=files._positive, help='Per socket; default its available physical cores')
        sub.add_argument('--cpu-numba-threads', type=files._positive, help='Per socket; default its resolved CPU thread count')
        sub.add_argument('--cpu-batch-size', type=files._positive, default=1024 if name == 'compress' else 4096,
                         help='Concurrent neural chunks per socket worker; independent of the number of selected sockets')
        sub.add_argument('--cpu-batch-number', type=files._positive, default=64,
                         help='I/O buffer multiplier per socket worker; target chunks per flush = cpu-batch-size * cpu-batch-number, unless --chunk-size is set')
        for key, default, choices in _CPU_OPTIONS:
            kwargs = {'default': default}
            if choices is None:
                kwargs['type'] = files._positive
            else:
                kwargs['choices'] = choices
            sub.add_argument('--' + key.replace('_', '-'), **kwargs,
                             help=_CPU_OPTION_HELP[key])
        sub.add_argument('--cpu-entropy-min-work', '--cpu-entropy-min-streams',
                         dest='cpu_entropy_min_work', type=files._positive, default=256,
                         help='Automatic entropy parallelism threshold per socket: stream/time pairs for encoding, streams per step for decoding')
        sub.add_argument('--cpu-profile-dir', type=Path,
                         help='Save one stage profile per active socket here; new report paths required')
        sub.add_argument('--temp-dir', type=Path, help='Existing scratch parent; defaults to output parent')
        sub.add_argument('--keep-temp', action='store_true',
                         help='Retain scratch shards, read-ID lists, job manifest and worker logs after success; decompressed output parts are always retained')
        sub.add_argument('--report', type=Path, help='Permanent job/timing JSON at a new path')
        sub.add_argument('-m', '--model-path', type=Path,
                         help='Quantized .pth checkpoint; compression selects the bundled model if omitted, decompression uses the archive model reference')
        sub.add_argument('-c', '--chunk-size', type=int, default=0, help='Reads per worker flush; 0 uses CPU b*B')
        sub.add_argument('-n', '--max-reads', type=int, default=-1, help='TOTAL reads before splitting; -1 means all')
        if name == 'compress':
            sub.add_argument('--pod5-reader-mode', choices=('persistent', 'reopen'), default='reopen',
                             help='Per socket: reopen POD5 by read IDs after each IO flush; ignored for BLOW5/SLOW5')
            sub.add_argument('-L', '--chunk-len', type=files._positive, default=2048,
                             help='Maximum small-delta tokens per neural chunk; stored in the archive for decompression')
            sub.add_argument('--dim', type=files._positive, default=256,
                             help='MLP decoder dimension; must match the model checkpoint')
            sub.add_argument('--num-dec-layers', type=files._positive, default=8,
                             help='Number of hidden MLP decoder layers; must match the model checkpoint')
            sub.add_argument('--window-size', type=files._positive, default=64,
                             help='Context length in preceding tokens; must match the model checkpoint')
            sub.add_argument('--model-embed', choices=('path', 'embed', 'none'), default='path',
                             help='Model storage in the archive: path records the checkpoint path, embed includes its bytes, none requires --model-path when decoding')
            sub.add_argument('--lsb-eliminate', type=int, choices=range(16), default=0,
                             help='Signal LSB reduction in bits: 0 selects the bundled lossless model, 3 the bundled lossy model; other values require a matching custom model')
            sub.add_argument('--write-index', action='store_true',
                             help='Also create OUTPUT.idx for the final compressed archive')
        else:
            sub.add_argument('--no-merge', action='store_true', default=True,
                             help='Default behavior: retain decoded files in OUTPUT.parts/ with manifest.json; merge manually if needed')
            sub.add_argument('--rec-press', choices=('none', 'zlib', 'zstd'), default='zlib',
                             help='Record compression for .blow5 output only; zstd requires a Zstandard-enabled pyslow5 build')
            sub.add_argument('--sig-press', choices=('none', 'svb-zd', 'ex-zd'), default='svb-zd',
                             help='Signal compression for .blow5 output only; none stores uncompressed signals')
    args = parser.parse_args(argv)
    if not sys.platform.startswith('linux'):
        parser.error('CPU socket placement requires Linux')
    args = files._validate_file_args(args, parser)
    if args.cpu_profile_dir:
        args.cpu_profile_dir = args.cpu_profile_dir.expanduser().resolve()
        if args.cpu_profile_dir.exists() and not args.cpu_profile_dir.is_dir():
            parser.error('--cpu-profile-dir must be a directory')
    return args


def _profile_path(args, socket_id):
    return (args.cpu_profile_dir / f'socket-{socket_id}.{args.command}.json'
            if args.cpu_profile_dir else None)


def _base_worker(args, socket_id, output):
    placement = args.socket_plans[socket_id]
    command = [sys.executable, '-m', 'sigcomp.main', args.command,
               '-i', str(args.input), '-o', str(output), '--device', 'cpu',
               '--cpu-socket-id', str(socket_id), '--cpu-memory-policy', args.cpu_memory_policy,
               '--cpu-threads', str(placement['cpu_threads']),
               '--cpu-numba-threads', str(placement['cpu_numba_threads']),
               '--cpu-batch-size', str(args.cpu_batch_size), '--cpu-batch-number', str(args.cpu_batch_number),
               '-c', str(args.chunk_size)]
    for key, _, _ in _CPU_OPTIONS:
        command.extend(['--' + key.replace('_', '-'), str(getattr(args, key))])
    command.extend(['--cpu-entropy-min-work', str(args.cpu_entropy_min_work)])
    profile = _profile_path(args, socket_id)
    if profile:
        command.extend(['--cpu-profile', str(profile), '--cpu-profile-label', f'CPU socket {socket_id}'])
    return command


def _worker_environment():
    env = os.environ.copy()
    root = str(Path(__file__).resolve().parent.parent)
    env['PYTHONPATH'] = root + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['PYTHONUNBUFFERED'] = '1'
    return env


def main(argv=None):
    args = _parse_args(argv)
    launch_affinity = set(os.sched_getaffinity(0))
    topology = discover_sockets()
    args.sockets = _select_sockets(args.sockets, topology)
    args.socket_plans = {socket: plan_socket(socket, memory_policy=args.cpu_memory_policy,
                         cpu_threads=args.cpu_threads, numba_threads=args.cpu_numba_threads, sockets=topology)
                         for socket in args.sockets}
    if args.cpu_profile_dir:
        protected = {args.input, args.output, args.model_path, args.report, Path(str(args.output) + '.idx')}
        for socket in args.sockets:
            path = _profile_path(args, socket)
            if path in protected or os.path.lexists(path):
                raise ValueError(f'CPU profile must be a new path distinct from job files: {path}')
        args.cpu_profile_dir.mkdir(parents=True, exist_ok=True)
    single = len(args.sockets) == 1
    job_dir = Path(tempfile.mkdtemp(prefix='sigcomp-mcpu-', dir=args.temp_dir)).resolve()
    files._message(f'Job directory: {job_dir}')
    for socket, plan in args.socket_plans.items():
        files._message(f"Socket {socket}: CPUs {plan['cpu_list']}; "
                       f"Torch/Numba workers {plan['cpu_threads']}/{plan['cpu_numba_threads']}; "
                       f"memory {plan['memory_policy']} {plan['memory_nodes']}")
    started = time.perf_counter()
    report = {'schema_version': 2, 'launcher': 'multi_cpu_socket', 'operation': args.command,
              'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'preparing',
              'input': str(args.input), 'output': str(args.output), 'job_directory': str(job_dir),
              'source_identity': files._signature(args.input), 'selected_sockets': args.sockets,
              'planned_socket_placement': args.socket_plans, 'single_socket_fallback': single,
              'execution_mode': 'workers', 'output_mode': 'single_file',
              'no_merge': getattr(args, 'no_merge', False),
              'coordinator_initial_cpu_ids': sorted(launch_affinity),
              'cpu_batch_size_per_socket': args.cpu_batch_size, 'cpu_batch_number_per_socket': args.cpu_batch_number,
              'cpu_profile_directory': str(args.cpu_profile_dir) if args.cpu_profile_dir else None,
              'workers': []}
    if args.command == 'compress':
        report['pod5_reader_mode'] = args.pod5_reader_mode
    files._save_manifest(job_dir, report)
    if single:
        return files._run_single_job(args, job_dir, report, args.sockets[0],
                                     device_key='socket_id', base_worker=_base_worker,
                                     environment=_worker_environment())
    previous_term = signal.getsignal(signal.SIGTERM)

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt('termination requested')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        files._start_shard_outputs(args, report)
        prepare = files._prepare_compression if args.command == 'compress' else files._prepare_decompression
        workers, header, model_identity = prepare(args, job_dir, worker_ids=args.sockets,
                                                 worker_key='socket_id', base_worker=_base_worker,
                                                 **({'report': report} if args.command == 'compress' else {}))
        for worker in workers:
            worker['planned_placement'] = args.socket_plans[worker['socket_id']]
            profile = _profile_path(args, worker['socket_id'])
            worker['cpu_profile'] = str(profile) if profile else None
        report['workers'] = workers
        report['num_reads'] = (sum(worker['read_count'] for worker in workers)
                               if all(worker['read_count'] is not None for worker in workers) else None)
        report['preparation_seconds'] = time.perf_counter() - started
        report['model_identity'] = model_identity
        files._assert_unchanged(report['source_identity'])
        if model_identity:
            files._assert_unchanged(model_identity)
        os.sched_setaffinity(0, launch_affinity)
        selected_cpus = {cpu for plan in args.socket_plans.values() for cpu in plan['cpu_ids']}
        if not selected_cpus <= set(os.sched_getaffinity(0)):
            raise RuntimeError('CPU allocation changed during preparation; restart the launcher')
        report['status'] = 'running'
        report['worker_wall_seconds'] = files._run_workers(args, job_dir, report, environment=_worker_environment())
        files._assert_unchanged(report['source_identity'])
        if model_identity:
            files._assert_unchanged(model_identity)
        report['status'] = 'publishing' if getattr(args, 'no_merge', False) else 'merging'
        files._save_manifest(job_dir, report)
        merge_started = time.perf_counter()
        if getattr(args, 'no_merge', False):
            report['merge'] = files._retain_decoded_shards(args, report)
        else:
            from .archive_shards import merge_archives
            result = merge_archives([worker['output'] for worker in workers], args.output,
                                    global_dict=header, write_index=args.write_index,
                                    expected_read_ids=files._read_ids(worker['read_ids_file'] for worker in workers))
            report['merge'] = {'output_bytes': args.output.stat().st_size, 'details': str(result)}
        report['merge_seconds'] = (0.0 if getattr(args, 'no_merge', False)
                                   else time.perf_counter() - merge_started)
        report['total_seconds'] = time.perf_counter() - started
        report['status'] = 'complete'
        files._save_manifest(job_dir, report)
        if args.report:
            with args.report.open('x', encoding='utf-8') as stream:
                json.dump(report, stream, indent=2)
                stream.write('\n')
        files._message(f"Saved {report.get('output_directory', args.output)}: {report['num_reads']:,} reads. "
                       f"Preparation {report['preparation_seconds']:.2f} s; "
                       f"workers {report['worker_wall_seconds']:.2f} s; merge/publication {report['merge_seconds']:.2f} s.")
    except BaseException as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        report['total_seconds'] = time.perf_counter() - started
        files._save_manifest(job_dir, report)
        files._message(f'Job files retained for recovery: {job_dir}')
        for key in ('single_staging_directory', 'output_directory'):
            if report.get(key):
                files._message(f"Retained staging path: {report[key]}")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    files._cleanup_job(args, job_dir)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        files._message('Interrupted; active workers were stopped.')
        raise SystemExit(130)
    except Exception as exc:
        files._message(f'Error: {exc}')
        raise SystemExit(1)
