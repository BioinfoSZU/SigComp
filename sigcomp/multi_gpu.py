import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def _gpu_ids(value):
    try:
        values = [int(part.strip()) for part in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('use comma-separated logical GPU numbers, e.g. 0,1,2,3') from exc
    if not values or min(values) < 0 or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError('GPU numbers must be distinct and nonnegative')
    return values


def _message(text):
    print(text, file=sys.stderr, flush=True)


def _signature(path):
    stat = path.stat()
    return {'path': str(path.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def _assert_unchanged(signature):
    if _signature(Path(signature['path'])) != signature:
        raise RuntimeError(f"Input/model changed while the job was running: {signature['path']}")


def _save_manifest(job_dir, report):
    temporary = job_dir / 'job.json.new'
    temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, job_dir / 'job.json')
    if report.get('output_mode') == 'shards' and report.get('output_directory'):
        directory = Path(report['output_directory'])
        temporary = directory / 'manifest.json.new'
        temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, directory / 'manifest.json')


def _write_ids(path, read_ids):
    with path.open('x', encoding='ascii', newline='\n') as stream:
        for read_id in read_ids:
            stream.write(str(uuid.UUID(bytes=read_id) if isinstance(read_id, bytes)
                             else uuid.UUID(str(read_id))) + '\n')


def _read_ids(paths):
    for path in paths:
        with Path(path).open(encoding='ascii') as stream:
            for line in stream:
                if line.strip():
                    yield uuid.UUID(line.strip()).bytes


def _ranges(count, workers):
    workers = min(workers, count)
    quotient, remainder = divmod(count, workers)
    start = 0
    for index in range(workers):
        stop = start + quotient + (index < remainder)
        yield start, stop
        start = stop


def _base_worker(args, gpu, output):
    return [sys.executable, '-m', 'sigcomp.main', args.command,
            '-i', str(args.input), '-o', str(output), '--device', 'cuda',
            '-g', str(gpu), '-b', str(args.batch_size), '-B', str(args.batch_number),
            '-c', str(args.chunk_size)]


def _single_command(args, device_id, output, base_worker):
    command = base_worker(args, device_id, output) + ['-n', str(args.max_reads)]
    if args.model_path:
        command.extend(['-m', str(args.model_path)])
    if args.command == 'compress':
        command.extend(['--model-embed', args.model_embed, '-L', str(args.chunk_len),
                        '--dim', str(args.dim), '--num-dec-layers', str(args.num_dec_layers),
                        '--window-size', str(args.window_size), '--lsb-eliminate', str(args.lsb_eliminate),
                        '--pod5-reader-mode', getattr(args, 'pod5_reader_mode', 'reopen')])
    else:
        command.extend(['--rec-press', args.rec_press, '--sig-press', args.sig_press])
    return command


def _invoke_codec(command, environment):
    previous_argv, previous_environment = sys.argv, os.environ.copy()
    try:
        sys.argv = ['sigcomp.main', *command[3:]]
        os.environ.update(environment)
        try:
            runpy.run_module('sigcomp.main', run_name='__main__', alter_sys=True)
        except SystemExit as error:
            if error.code not in (None, 0):
                raise
    finally:
        sys.argv = previous_argv
        os.environ.clear()
        os.environ.update(previous_environment)


def _start_shard_outputs(args, report):
    if not getattr(args, 'no_merge', False):
        return
    directory = Path(str(args.output) + '.parts')
    directory.mkdir()  # Never reuse or overwrite somebody else's shards.
    args.shard_output_dir = directory
    report['output_mode'] = 'shards'
    report['output_directory'] = str(directory)
    _message(f'Decompressed shards will remain in {directory}; no merged file will be created.')


def _decoded_output(args, job_dir, index):
    directory = getattr(args, 'shard_output_dir', None) or job_dir
    return directory / f'part-{index:04d}{args.output.suffix.lower()}'


def _retain_decoded_shards(args, report):
    entries = report['workers'] or [report['single_run']]
    outputs = []
    for entry in entries:
        path = Path(entry['output'])
        if path.parent != args.shard_output_dir or not path.is_file():
            raise RuntimeError(f'Missing or misplaced decoded shard: {path}')
        outputs.append({'path': str(path), 'bytes': path.stat().st_size,
                        'start_read': entry.get('start_read', 0), 'read_count': entry['read_count'],
                        'num_samples': entry.get('num_samples')})
    report['output_files'] = outputs
    return {'skipped': True, 'reason': 'separate_decoded_outputs',
            'output_bytes': sum(item['bytes'] for item in outputs),
            'note': 'Independent decoded files; see output_directory/manifest.json. No payload copy or merge.'}


def _publish_single_result(args, report):
    from .archive_shards import read_archive_header, scan_archive, write_index

    entry = report['single_run']
    staged = Path(entry['output'])
    if not staged.is_file():
        raise RuntimeError(f'Codec completed without creating {staged}')
    header, _ = read_archive_header(staged if args.command == 'compress' else args.input)
    count = header['num_reads']
    if args.command == 'decompress' and args.max_reads > 0:
        count = min(count, args.max_reads)
    entry['read_count'] = report['num_reads'] = count
    if args.command == 'decompress' and args.output.suffix.lower() == '.pod5':
        import pod5
        with pod5.Reader(staged) as reader:
            if reader.num_reads != count:
                raise ValueError('Single-device POD5 output read count differs from selection')
    if getattr(args, 'no_merge', False):
        return _retain_decoded_shards(args, report)

    staged_index = None
    if args.command == 'compress' and args.write_index:
        staged_index = Path(str(staged) + '.idx')
        write_index(scan_archive(staged), staged_index)
    output_bytes = staged.stat().st_size
    _publish_file(staged, args.output)
    report['output_published'] = True
    entry['output'] = str(args.output)
    if staged_index:
        try:
            _publish_file(staged_index, Path(str(args.output) + '.idx'))
        except OSError as exc:
            raise RuntimeError(f'Archive published at {args.output}, but index publication failed; '
                               f'retained index: {staged_index}') from exc
    staged.parent.rmdir()
    return {'skipped': True, 'reason': 'single_process', 'output_bytes': output_bytes,
            'note': 'Normal main in the launcher PID; no worker, partition, payload copy or merge.'}


def _cleanup_job(args, job_dir):
    if args.keep_temp:
        _message(f'Job files retained: {job_dir}')
        return
    if (job_dir.parent != args.temp_dir or job_dir.is_symlink()
            or not job_dir.name.startswith(('sigcomp-mgpu-', 'sigcomp-mcpu-'))):
        raise RuntimeError(f'Refusing cleanup of unexpected job path: {job_dir}')
    shutil.rmtree(job_dir)


def _run_single_job(args, job_dir, report, device_id, *, device_key='gpu',
                    base_worker=_base_worker, environment=None):
    started = time.perf_counter()
    report.update(execution_mode='single_process', workers=[], worker_wall_seconds=0.0,
                  merge_seconds=0.0)
    previous_term = signal.getsignal(signal.SIGTERM)

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt('termination requested')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        _start_shard_outputs(args, report)
        if getattr(args, 'no_merge', False):
            output = _decoded_output(args, job_dir, 0)
        else:
            stage = Path(tempfile.mkdtemp(prefix=f'.{args.output.name}.single-',
                                          dir=args.output.parent)).resolve()
            report['single_staging_directory'] = str(stage)
            output = stage / args.output.name
        command = _single_command(args, device_id, output, base_worker)
        entry = {'shard': 0, device_key: device_id, 'start_read': 0, 'read_count': None,
                 'command': command, 'output': str(output), 'pid': os.getpid(),
                 'log': None, 'log_note': 'Normal codec output goes directly to the terminal.'}
        report['single_run'] = entry
        report['preparation_seconds'] = time.perf_counter() - started
        report['model_identity'] = _signature(args.model_path) if args.model_path else None
        report['status'] = 'running'
        _save_manifest(job_dir, report)
        _message(f'One selected {device_key} ({device_id}): running main in this process; no workers or merge.')
        codec_started = time.perf_counter()
        _invoke_codec(command, _worker_environment(args) if environment is None else environment)
        entry['elapsed_seconds'] = report['codec_seconds'] = time.perf_counter() - codec_started
        entry['returncode'] = 0
        _assert_unchanged(report['source_identity'])
        if report['model_identity']:
            _assert_unchanged(report['model_identity'])
        report['status'] = 'publishing'
        _save_manifest(job_dir, report)
        publish_started = time.perf_counter()
        report['merge'] = _publish_single_result(args, report)
        report['publication_seconds'] = time.perf_counter() - publish_started
        report['total_seconds'] = time.perf_counter() - started
        report['status'] = 'complete'
        _save_manifest(job_dir, report)
        if args.report:
            with args.report.open('x', encoding='utf-8') as stream:
                json.dump(report, stream, indent=2)
                stream.write('\n')
        destination = report.get('output_directory', str(args.output))
        _message(f'Saved {destination}: {report["num_reads"]:,} reads; '
                 f'codec {report["codec_seconds"]:.2f}s; no worker/merge phase.')
    except BaseException as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        report['total_seconds'] = time.perf_counter() - started
        _save_manifest(job_dir, report)
        _message(f'Job files retained for recovery: {job_dir}')
        for key in ('single_staging_directory', 'output_directory'):
            if report.get(key):
                _message(f'Retained output path: {report[key]}')
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    _cleanup_job(args, job_dir)
    return 0


def _compression_model(args):
    from . import main as codec
    model_path = args.model_path
    if model_path is None:
        if args.lsb_eliminate == 0:
            model_path = Path(codec.DEFAULT_QUANT_LOSSLESS_MODEL_PATH)
        elif args.lsb_eliminate == codec.DEFAULT_LSB_SETTING_FOR_LOSSY_MODEL:
            model_path = Path(codec.DEFAULT_QUANT_LOSSY_MODEL_PATH)
        else:
            raise ValueError('A custom --model-path is required for this --lsb-eliminate value')
    model_path = model_path.resolve(strict=True)
    if not model_path.is_file():
        raise ValueError('--model-path must identify a checkpoint file')
    args.model_path = model_path
    return _signature(model_path)


def _prepare_compression(args, job_dir, *, worker_ids=None, worker_key='gpu', base_worker=None, report=None):
    from . import main as codec
    from .archive_shards import write_header
    from .read_assignment import sample_balanced_ranges, assignment_summary

    worker_ids = args.gpus if worker_ids is None else worker_ids
    base_worker = _base_worker if base_worker is None else base_worker
    model_identity = _compression_model(args)
    model_path = args.model_path
    _message('Reading source metadata/read IDs once; signal arrays are not partitioned into temporary files...')
    inventory_started = time.perf_counter()
    metadata, read_ids, sample_counts = codec.collect_source_metadata(
        str(args.input), max_reads=args.max_reads, include_sample_counts=True)
    inventory_seconds = time.perf_counter() - inventory_started
    if not read_ids:
        raise ValueError('The selected input contains no reads; no workers were started')
    assignment_started = time.perf_counter()
    sample_plans = None
    if metadata['source_format'] == 'pod5':
        if sample_counts is None or len(sample_counts) != len(read_ids):
            raise ValueError('POD5 UUID and sample-count inventory lengths differ')
        sample_plans = sample_balanced_ranges(sample_counts, len(worker_ids))
        ranges = [(plan.start, plan.stop) for plan in sample_plans]
        balance = assignment_summary(sample_plans, largest_read_samples=max(sample_counts, default=0))
    else:
        ranges = list(_ranges(len(read_ids), len(worker_ids)))
        balance = {'algorithm': 'contiguous_equal_read_count', 'basis': 'pyslow5_get_read_ids',
                   'active_workers': len(ranges), 'worker_read_counts': [stop - start for start, stop in ranges]}
    balance['assignment_seconds'] = time.perf_counter() - assignment_started
    balance['inventory_seconds'] = inventory_seconds
    if report is not None:
        report['read_assignment'] = balance
        if sample_plans is not None:
            report['num_samples'] = balance['total_samples']
    _message(f"Read assignment: {balance['algorithm']}; {len(ranges)} workers; "
             f"planning {balance['assignment_seconds']:.3f}s (inventory {inventory_seconds:.3f}s).")
    if sample_plans is not None and balance['max_to_mean_ratio'] is not None:
        _message(f"  Samples per worker: {balance['min_worker_samples']:,}–{balance['max_worker_samples']:,}; "
                 f"largest/mean {balance['max_to_mean_ratio']:.4f}.")
    header = dict(metadata)
    header.update({
        'format_version': 2, 'chunk_len': args.chunk_len,
        'model_embed': args.model_embed,
        'model_data': (model_path.read_bytes() if args.model_embed == 'embed'
                       else str(model_path) if args.model_embed == 'path' else None),
        'model_dim': args.dim, 'model_num_dec_layers': args.num_dec_layers,
        'model_window_size': args.window_size, 'lsb_eliminate': args.lsb_eliminate,
        'model_identity': model_identity,
        'command': shlex.join(sys.argv), 'datetime': datetime.now(timezone.utc).isoformat(),
    })
    metadata_path = job_dir / 'shared_header.sigcomp'
    with metadata_path.open('xb') as stream:
        write_header(stream, header)
    workers = []
    for index, (start, stop) in enumerate(ranges):
        device_id = worker_ids[index]
        ids_path = job_dir / f'part-{index:04d}.read_ids.txt'
        _write_ids(ids_path, (read_ids[i] for i in range(start, stop)))
        output = job_dir / f'part-{index:04d}.sigcomp'
        command = base_worker(args, device_id, output) + [
            '--read-ids-file', str(ids_path), '--metadata-header', str(metadata_path),
            '-m', str(model_path), '--model-embed', args.model_embed,
            '-L', str(args.chunk_len), '--dim', str(args.dim),
            '--num-dec-layers', str(args.num_dec_layers), '--window-size', str(args.window_size),
            '--lsb-eliminate', str(args.lsb_eliminate),
            '--pod5-reader-mode', getattr(args, 'pod5_reader_mode', 'reopen'),
        ]
        workers.append({'shard': index, worker_key: device_id, 'start_read': start,
                        'read_count': stop - start, 'read_ids_file': str(ids_path),
                        'output': str(output), 'command': command,
                        'log': str(job_dir / f'worker-{index:04d}.log')})
        if sample_plans is not None:
            workers[-1]['num_samples'] = sample_plans[index].num_samples
            _message(f'  {worker_key} {device_id}: {stop - start:,} reads, '
                     f'{sample_plans[index].num_samples:,} samples')
    return workers, header, model_identity


def _prepare_decompression(args, job_dir, *, worker_ids=None, worker_key='gpu', base_worker=None):
    from .archive_shards import scan_archive, plan_archive_shards, write_index

    worker_ids = args.gpus if worker_ids is None else worker_ids
    base_worker = _base_worker if base_worker is None else base_worker
    _message('Scanning SIG2 record framing once (compressed signal blobs are skipped)...')
    scan = scan_archive(args.input)
    source_format = scan.header.get('source_format', 'pod5')
    suffix = args.output.suffix.lower()
    if suffix != '.bin' and (source_format == 'pod5') != (suffix == '.pod5'):
        raise ValueError('Decode output must match the archive source format, or use .bin')
    if args.model_path is None:
        model_embed = scan.header.get('model_embed', 'path')
        if model_embed == 'path':
            stored_path = scan.header.get('model_data')
            if not stored_path:
                raise ValueError('Archive has no model path; provide --model-path')
            args.model_path = Path(stored_path).expanduser().resolve(strict=True)
        elif model_embed != 'embed':
            raise ValueError('Archive has no embedded model; provide --model-path')
    if args.model_path is not None and not args.model_path.is_file():
        raise ValueError('--model-path must identify a checkpoint file')
    index_path = job_dir / 'input.sigcomp.idx'
    write_index(scan, index_path)
    plans = plan_archive_shards(scan, len(worker_ids), max_reads=args.max_reads)
    if not plans:
        raise ValueError('The selected archive contains no reads; no workers were started')
    workers = []
    for index, plan in enumerate(plans):
        device_id = worker_ids[index]
        output = _decoded_output(args, job_dir, index)
        ids_path = job_dir / f'part-{index:04d}.read_ids.txt'
        _write_ids(ids_path, plan.read_ids)
        command = base_worker(args, device_id, output) + [
            '--record-start', str(plan.start_read), '--record-count', str(plan.stop_read - plan.start_read),
            '--index-file', str(index_path),
        ]
        if args.model_path:
            command.extend(['-m', str(args.model_path)])
        if source_format == 'blow5':
            command.extend(['--rec-press', args.rec_press, '--sig-press', args.sig_press])
        workers.append({'shard': index, worker_key: device_id, 'start_read': plan.start_read,
                        'read_count': plan.stop_read - plan.start_read,
                        'num_samples': plan.num_samples, 'read_ids_file': str(ids_path),
                        'output': str(output), 'command': command,
                        'log': str(job_dir / f'worker-{index:04d}.log')})
    return workers, scan.header, _signature(args.model_path) if args.model_path else None


def _worker_environment(args):
    env = os.environ.copy()
    root = str(Path(__file__).resolve().parent.parent)
    env['PYTHONPATH'] = root + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['PYTHONUNBUFFERED'] = '1'
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = str(args.worker_cpu_threads)
    return env


def _stop_processes(processes):
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    for process in processes:
        try:
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def _tail(path, limit=8192):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(max(0, os.fstat(stream.fileno()).st_size - limit))
            return stream.read().decode('utf-8', errors='replace')
    except OSError:
        return '(worker log unavailable)'


def _worker_label(worker):
    return (f"CPU socket {worker['socket_id']}" if 'socket_id' in worker
            else f"GPU {worker['gpu']}")


def _run_workers(args, job_dir, report, *, environment=None):
    processes, log_streams = [], []
    started = time.perf_counter()
    try:
        for worker in report['workers']:
            log_stream = open(worker['log'], 'xb')
            log_streams.append(log_stream)
            worker['started_monotonic'] = time.perf_counter()
            process = subprocess.Popen(worker['command'], stdout=log_stream, stderr=subprocess.STDOUT,
                                       env=(_worker_environment(args) if environment is None else environment),
                                       start_new_session=True)
            processes.append(process)
            worker['pid'] = process.pid
            count = worker.get('read_count')
            read_text = f'{count:,} reads' if count is not None else 'all selected reads'
            _message(f"{_worker_label(worker)}: {read_text}; log {worker['log']}")
        _save_manifest(job_dir, report)
        pending = set(range(len(processes)))
        heartbeat = time.monotonic()
        while pending:
            for index in tuple(pending):
                code = processes[index].poll()
                if code is None:
                    continue
                worker = report['workers'][index]
                worker['returncode'] = code
                worker['elapsed_seconds'] = time.perf_counter() - worker['started_monotonic']
                pending.remove(index)
                _save_manifest(job_dir, report)
                if code:
                    raise RuntimeError(f"{_worker_label(worker)} worker failed (exit {code}). "
                                       f"Log: {worker['log']}\n{_tail(worker['log'])}")
                if not Path(worker['output']).is_file():
                    raise RuntimeError(f"Worker exited successfully but produced no output: {worker['output']}")
                _message(f"{_worker_label(worker)} finished in {worker['elapsed_seconds']:.2f} s")
            if pending:
                if time.monotonic() - heartbeat >= 30:
                    _message(f'{len(pending)} worker(s) still running; elapsed {time.perf_counter() - started:.1f} s')
                    heartbeat = time.monotonic()
                time.sleep(0.25)
        return time.perf_counter() - started
    finally:
        _stop_processes(processes)
        for stream in log_streams:
            stream.close()
        for index, process in enumerate(processes):
            report['workers'][index]['returncode'] = process.returncode
        _save_manifest(job_dir, report)


def _publish_file(staged, destination):
    with staged.open('rb') as stream:
        os.fsync(stream.fileno())
    os.link(staged, destination)
    staged.unlink()


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('compress', 'decompress'):
        sub = commands.add_parser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  help=('Compress reads into one SIG2 archive' if name == 'compress' else 'Decompress into independent files in OUTPUT.parts/'))
        sub.add_argument('-i', '--input', type=Path, required=True, help='One source file (directories are not accepted)')
        sub.add_argument('-o', '--output', type=Path, required=True,
                         help='Compression: new SIG2 archive path. Decompression: output name with .pod5/.blow5/.slow5/.bin extension; files are retained in OUTPUT.parts/')
        sub.add_argument('--gpus', type=_gpu_ids, help='Logical CUDA GPU ordinals, e.g. 0,1,2,3; omit for all visible GPUs')
        sub.add_argument('--temp-dir', type=Path, help='Parent for a unique job directory; defaults to output directory')
        sub.add_argument('--keep-temp', action='store_true', help='Retain completed shards, read lists, manifest and logs after success')
        sub.add_argument('--report', type=Path, help='Optional permanent JSON job/timing report; must not exist')
        sub.add_argument('-m', '--model-path', type=Path,
                         help='Quantized .pth checkpoint; compression selects the bundled model if omitted, decompression uses the archive model reference')
        sub.add_argument('-b', '--batch-size', type=_positive, default=20480, help='Concurrent neural chunks PER GPU')
        sub.add_argument('-B', '--batch-number', type=_positive, default=8, help='IO buffer multiplier PER GPU')
        sub.add_argument('-c', '--chunk-size', type=int, default=0, help='Reads per worker flush; 0 uses b*B chunk estimate')
        sub.add_argument('-n', '--max-reads', type=int, default=-1, help='Total reads across all workers; -1 means all')
        sub.add_argument('--worker-cpu-threads', type=_positive, default=1, help='OMP/MKL/OpenBLAS/NumExpr thread cap PER GPU process')
        if name == 'compress':
            sub.add_argument('--pod5-reader-mode', choices=('persistent', 'reopen'), default='reopen',
                             help='Per worker: reopen POD5 by read IDs after each IO flush; ignored for BLOW5/SLOW5')
            sub.add_argument('-L', '--chunk-len', type=_positive, default=2048,
                             help='Maximum small-delta tokens per neural chunk; stored in the archive for decompression')
            sub.add_argument('--dim', type=_positive, default=256,
                             help='MLP decoder dimension; must match the model checkpoint')
            sub.add_argument('--num-dec-layers', type=_positive, default=8,
                             help='Number of hidden MLP decoder layers; must match the model checkpoint')
            sub.add_argument('--window-size', type=_positive, default=64,
                             help='Context length in preceding tokens; must match the model checkpoint')
            sub.add_argument('--model-embed', choices=('path', 'embed', 'none'), default='path',
                             help='Model storage in the archive: path records the checkpoint path, embed includes its bytes, none requires --model-path when decoding')
            sub.add_argument('--lsb-eliminate', type=int, choices=range(16), default=0,
                             help='Signal LSB reduction in bits: 0 selects the bundled lossless model, 3 the bundled lossy model; other values require a matching custom model')
            sub.add_argument('--write-index', action='store_true', help='Also create OUTPUT.idx while merging SIG2 shards')
        else:
            sub.add_argument('--no-merge', action='store_true', default=True,
                             help='Default behavior: retain decoded files in OUTPUT.parts/ with manifest.json; merge manually if needed')
            sub.add_argument('--rec-press', choices=('none', 'zlib', 'zstd'), default='zlib',
                             help='Record compression for .blow5 output only; zstd requires a Zstandard-enabled pyslow5 build')
            sub.add_argument('--sig-press', choices=('none', 'svb-zd', 'ex-zd'), default='svb-zd',
                             help='Signal compression for .blow5 output only; none stores uncompressed signals')
    return _validate_file_args(parser.parse_args(argv), parser)


def _validate_file_args(args, parser):
    if os.name != 'posix':
        parser.error('This launcher targets Linux/POSIX servers; use main for single-device operation')
    if args.max_reads != -1 and args.max_reads < 1:
        parser.error('--max-reads must be -1 or a positive total read count')
    if args.chunk_size < 0:
        parser.error('--chunk-size must be nonnegative')
    args.input = args.input.expanduser().resolve(strict=True)
    if not args.input.is_file():
        parser.error('--input must be one regular file')
    args.output = args.output.expanduser().absolute()
    if args.output.exists() or args.output.is_symlink():
        parser.error('Output already exists; this launcher never overwrites it')
    if not args.output.parent.is_dir():
        parser.error('Create the output parent directory before starting')
    if args.model_path:
        args.model_path = args.model_path.expanduser().resolve(strict=True)
    args.temp_dir = (args.temp_dir or args.output.parent).expanduser().resolve()
    if not args.temp_dir.is_dir():
        parser.error('--temp-dir must be an existing directory')
    if args.report:
        args.report = args.report.expanduser().absolute()
        if args.report == args.output or args.report.exists() or args.report.is_symlink():
            parser.error('--report must be a new path distinct from the output')
        if not args.report.parent.is_dir():
            parser.error('Create the --report parent directory before starting')
    if args.command == 'compress':
        if args.dim < 2:
            parser.error('--dim must be at least 2')
        if args.input.suffix.lower() not in ('.pod5', '.blow5', '.slow5'):
            parser.error('Compression input must be .pod5, .blow5 or .slow5')
        if args.write_index and os.path.lexists(str(args.output) + '.idx'):
            parser.error('Output index already exists; it will not be overwritten')
        if args.report and args.write_index and str(args.report) == str(args.output) + '.idx':
            parser.error('--report cannot be the output index path')
    else:
        suffix = args.output.suffix.lower()
        if suffix not in ('.pod5', '.blow5', '.slow5', '.bin'):
            parser.error('Decode output must end in .pod5, .blow5, .slow5 or .bin')
        if getattr(args, 'no_merge', False):
            directory = Path(str(args.output) + '.parts')
            if directory.exists() or directory.is_symlink():
                parser.error(f'Shard output directory already exists: {directory}')
            if args.report == directory:
                parser.error('--report cannot be the shard output directory')
    return args


def main(argv=None):
    args = _parse_args(argv)
    requested_gpus = args.gpus
    import torch
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    args.gpus = list(range(available_gpus)) if args.gpus is None else args.gpus
    if not args.gpus or max(args.gpus) >= available_gpus:
        raise ValueError('--gpus must name available logical CUDA devices within CUDA_VISIBLE_DEVICES')
    if importlib.util.find_spec('triton') is None:
        raise ImportError('GPU workers require the existing Triton dependency')
    job_dir = Path(tempfile.mkdtemp(prefix='sigcomp-mgpu-', dir=args.temp_dir)).resolve()
    _message(f'Job directory: {job_dir}')
    started = time.perf_counter()
    report = {'schema_version': 2, 'operation': args.command,
              'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'preparing',
              'input': str(args.input), 'output': str(args.output), 'job_directory': str(job_dir),
              'source_identity': _signature(args.input), 'requested_gpus': requested_gpus,
              'selected_gpus': args.gpus, 'single_gpu_fallback': len(args.gpus) == 1,
              'execution_mode': 'workers', 'output_mode': 'single_file',
              'no_merge': getattr(args, 'no_merge', False),
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'batch_size_per_gpu': args.batch_size, 'batch_number_per_gpu': args.batch_number,
              'worker_cpu_threads': args.worker_cpu_threads, 'workers': []}
    if args.command == 'compress':
        report['pod5_reader_mode'] = args.pod5_reader_mode
    _save_manifest(job_dir, report)
    if len(args.gpus) == 1:
        torch.set_num_threads(args.worker_cpu_threads)
        return _run_single_job(args, job_dir, report, args.gpus[0])
    previous_term = signal.getsignal(signal.SIGTERM)

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt('termination requested')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        _start_shard_outputs(args, report)
        prepare = _prepare_compression if args.command == 'compress' else _prepare_decompression
        workers, header, model_identity = prepare(
            args, job_dir, **({'report': report} if args.command == 'compress' else {}))
        report['workers'] = workers
        report['num_reads'] = sum(worker['read_count'] for worker in workers)
        report['preparation_seconds'] = time.perf_counter() - started
        report['model_identity'] = model_identity
        _assert_unchanged(report['source_identity'])
        if model_identity:
            _assert_unchanged(model_identity)
        report['status'] = 'running'
        report['worker_wall_seconds'] = _run_workers(args, job_dir, report)
        _assert_unchanged(report['source_identity'])
        if model_identity:
            _assert_unchanged(model_identity)
        report['status'] = 'publishing' if getattr(args, 'no_merge', False) else 'merging'
        _save_manifest(job_dir, report)
        merge_started = time.perf_counter()
        if getattr(args, 'no_merge', False):
            report['merge'] = _retain_decoded_shards(args, report)
        else:
            _message('All workers completed; combining outputs...')
            from .archive_shards import merge_archives
            result = merge_archives(
                [worker['output'] for worker in workers], args.output,
                global_dict=header, write_index=args.write_index,
                expected_read_ids=_read_ids(worker['read_ids_file'] for worker in workers))
            report['merge'] = {'output_bytes': args.output.stat().st_size, 'details': str(result)}
        report['merge_seconds'] = (0.0 if getattr(args, 'no_merge', False)
                                   else time.perf_counter() - merge_started)
        report['total_seconds'] = time.perf_counter() - started
        report['status'] = 'complete'
        _save_manifest(job_dir, report)
        if args.report:
            with args.report.open('x', encoding='utf-8') as stream:
                json.dump(report, stream, indent=2)
                stream.write('\n')
        _message(f"Saved {report.get('output_directory', args.output)}: {report['num_reads']:,} reads, "
                 f"{report['merge']['output_bytes']:,} bytes. "
                 f"Preparation {report['preparation_seconds']:.2f} s; "
                 f"workers {report['worker_wall_seconds']:.2f} s; merge {report['merge_seconds']:.2f} s.")
    except BaseException as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        report['total_seconds'] = time.perf_counter() - started
        _save_manifest(job_dir, report)
        _message(f'Job files retained for recovery: {job_dir}')
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    _cleanup_job(args, job_dir)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        _message('Interrupted; active workers were stopped.')
        raise SystemExit(130)
    except Exception as exc:
        _message(f'Error: {exc}')
        raise SystemExit(1)
