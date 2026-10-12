import random
import numpy as np
import pod5
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


RUN_INFO_FIELDS = [
    "acquisition_id",
    "acquisition_start_time",
    "adc_max",
    "adc_min",
    "context_tags",
    "experiment_name",
    "flow_cell_id",
    "flow_cell_product_code",
    "protocol_name",
    "protocol_run_id",
    "protocol_start_time",
    "sample_id",
    "sample_rate",
    "sequencing_kit",
    "sequencer_position",
    "sequencer_position_type",
    "software",
    "system_name",
    "system_type",
    "tracking_id",
]


def extract_read_ids(reader):
    read_ids = []
    for read in tqdm(reader.reads(), desc="Extracting read_ids"):
        read_ids.append(str(read.read_id))
    return read_ids


def is_nan(x):
    try:
        return np.isnan(x)
    except Exception:
        return False


def deep_equal(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    if is_nan(a) and is_nan(b):
        return True

    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        return np.array_equal(a, b, equal_nan=True)  # type: ignore

    if type(a) != type(b):
        return False

    from collections.abc import Mapping
    if isinstance(a, Mapping):
        if a.keys() != b.keys():
            return False
        return all(deep_equal(a[k], b[k]) for k in a.keys())

    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(deep_equal(x, y) for x, y in zip(a, b))

    if hasattr(a, "__dict__") and hasattr(b, "__dict__"):
        a_dict = {k: v for k, v in vars(a).items() if not k.startswith("_")}
        b_dict = {k: v for k, v in vars(b).items() if not k.startswith("_")}
        return deep_equal(a_dict, b_dict)

    return a == b


def compare_run_info(ori_run_info, decomp_run_info):
    for field in RUN_INFO_FIELDS:
        a = getattr(ori_run_info, field, None)
        b = getattr(decomp_run_info, field, None)
        if not deep_equal(a, b):
            return False, field
    return True, None


def compare_read_records(ori_read, decomp_read):
    """
    Compare one read record:
      - read_id
      - pore
      - calibration
      - read_number
      - start_sample
      - median_before
      - end_reason
      - run_info (all required fields)
      - signal
    """
    if str(ori_read.read_id) != str(decomp_read.read_id):
        return False, "read_id"

    if not deep_equal(ori_read.pore, decomp_read.pore):
        return False, "pore"

    if not deep_equal(ori_read.calibration, decomp_read.calibration):
        return False, "calibration"

    if ori_read.read_number != decomp_read.read_number:
        return False, "read_number"

    if ori_read.start_sample != decomp_read.start_sample:
        return False, "start_sample"

    if not deep_equal(ori_read.median_before, decomp_read.median_before):
        return False, f"median_before"

    if not deep_equal(ori_read.end_reason, decomp_read.end_reason):
        return False, "end_reason"

    ok, bad_field = compare_run_info(ori_read.run_info, decomp_read.run_info)
    if not ok:
        return False, f"run_info.{bad_field}"

    if not np.array_equal(ori_read.signal, decomp_read.signal):
        return False, "signal"

    return True, None


if __name__ == "__main__":
    parser = ArgumentParser(
        description="validate decompressed POD5 output against the original POD5",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pod5", type=str, required=True,
                        help='Original POD5 file or directory used as the reference')
    parser.add_argument("--decomp", type=str, required=True,
                        help='Restored POD5 file or directory, including a complete decompression .parts directory')
    parser.add_argument("--sample-n", type=int, default=None,
                        help="Validate randomly sampled N reads")
    parser.add_argument("--sample-ratio", type=float, default=None,
                        help="Validate randomly sampled ratio of reads (0.0 to 1.0)")
    parser.add_argument('--head', type=int, default=None,
                        help='Validate the first N reference reads after checking all read-ID sets; used only when neither --sample-n nor --sample-ratio is set')
    args = parser.parse_args()

    ori = pod5.DatasetReader(args.pod5)
    decomp = pod5.DatasetReader(args.decomp)

    ori_read_ids = extract_read_ids(ori)
    decomp_read_ids = extract_read_ids(decomp)

    ori_id_set = set(ori_read_ids)
    decomp_id_set = set(decomp_read_ids)

    missing_in_decomp = ori_id_set - decomp_id_set
    extra_in_decomp = decomp_id_set - ori_id_set

    print(f"Original read count: {len(ori_read_ids)}")
    print(f"Decompressed read count: {len(decomp_read_ids)}")
    print(f"Unique original read_id count: {len(ori_id_set)}")
    print(f"Unique decompressed read_id count: {len(decomp_id_set)}")

    if missing_in_decomp:
        print(f"ERROR: {len(missing_in_decomp)} read_id(s) missing in decompressed POD5")
    if extra_in_decomp:
        print(f"ERROR: {len(extra_in_decomp)} extra read_id(s) in decompressed POD5")

    if missing_in_decomp or extra_in_decomp:
        print("read_id contents are not identical.")
    else:
        print("read_id contents are identical.")

    exception_reads = 0
    mismatch_reads = 0

    reads_to_validate = ori_read_ids
    if args.sample_n is not None:
        num_to_sample = min(len(reads_to_validate), args.sample_n)
        reads_to_validate = random.sample(reads_to_validate, num_to_sample)
        print(f"Randomly sampling {num_to_sample} reads for validation (--sample-n).")
    elif args.sample_ratio is not None:
        num_to_sample = max(1, int(len(reads_to_validate) * args.sample_ratio))
        num_to_sample = min(len(reads_to_validate), num_to_sample)
        reads_to_validate = random.sample(reads_to_validate, num_to_sample)
        print(f"Randomly sampling {num_to_sample} reads for validation (--sample-ratio).")
    elif args.head is not None:
        num_to_sample = max(1, int(args.head))
        num_to_sample = min(len(reads_to_validate), num_to_sample)
        reads_to_validate = reads_to_validate[:num_to_sample]

    for read_id in tqdm(reads_to_validate, desc="Validating reads"):
        try:
            ori_read = ori.get_read(read_id)
        except Exception as e:
            print(f"ERROR: Failed to get original read {read_id}: {e}")
            exception_reads += 1
            continue

        try:
            decomp_read = decomp.get_read(read_id)
        except Exception as e:
            print(f"ERROR: Failed to get decompressed read {read_id}: {e}")
            exception_reads += 1
            continue

        ok, bad_field = compare_read_records(ori_read, decomp_read)
        if not ok:
            mismatch_reads += 1
            print(f"NOT EQUAL: read_id={read_id}, first mismatch field={bad_field}")

    print(f"Exception reads: {exception_reads}")
    print(f"Not equal reads: {mismatch_reads}")
