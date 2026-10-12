from array import array
from bisect import bisect_left
from dataclasses import dataclass
import operator


@dataclass(frozen=True)
class SampleRange:
    start: int
    stop: int
    num_samples: int


def sample_balanced_ranges(sample_counts, workers):
    workers = operator.index(workers)
    if workers < 1:
        raise ValueError('workers must be positive')
    prefix = array('Q', [0])
    total = 0
    for value in sample_counts:
        samples = operator.index(value)
        if samples < 0:
            raise ValueError('Sample counts must be nonnegative integers')
        total += samples
        if total > 0xFFFFFFFFFFFFFFFF:
            raise ValueError('Total sample count exceeds uint64 capacity')
        prefix.append(total)
    count = len(prefix) - 1
    if count == 0:
        return ()
    parts = min(count, workers)
    boundaries = [0]
    for part in range(1, parts):
        earliest, latest = boundaries[-1] + 1, count - (parts - part)
        if total == 0:
            quotient, remainder = divmod(count, parts)
            boundary = part * quotient + min(part, remainder)
        else:
            numerator = total * part
            target = (numerator + parts - 1) // parts
            right = min(bisect_left(prefix, target, earliest, latest + 1), latest)
            left = max(earliest, right - 1)
            # Compare integer distances without losing precision above 2**53.
            # Ties favor the later boundary; every worker gets at least one read.
            boundary = min((right, left), key=lambda i: abs(prefix[i] * parts - numerator))
        boundaries.append(boundary)
    boundaries.append(count)
    return tuple(SampleRange(start, stop, int(prefix[stop] - prefix[start]))
                 for start, stop in zip(boundaries, boundaries[1:]))


def assignment_summary(plans, *, largest_read_samples=0):
    loads = [plan.num_samples for plan in plans]
    total = sum(loads)
    target = total / len(loads) if loads else 0
    return {
        'algorithm': 'contiguous_sample_quantiles', 'basis': 'pod5_num_samples',
        'complexity': 'O(N + W log N); no read sorting',
        'active_workers': len(loads), 'total_samples': total,
        'target_samples_per_worker': target, 'worker_samples': loads,
        'min_worker_samples': min(loads, default=0),
        'max_worker_samples': max(loads, default=0),
        'max_to_mean_ratio': max(loads) / target if target else None,
        'largest_read_samples': int(largest_read_samples),
        'note': 'Whole reads and contiguous source order constrain balance; sample count is a runtime proxy.',
    }
