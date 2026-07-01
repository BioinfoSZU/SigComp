import os
import argparse
import numpy as np
import pod5

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table
from rich import box


def prepare_dataset(args):
    console = Console()

    console.print(
        f"[bold cyan]SigComp — Prepare Dataset[/]\n"
        f"  Input        [bold]{args.input}[/]\n"
        f"  Output dir   [bold]{args.output}[/]\n"
        f"  Slice length [bold]{args.slice_len}[/]\n"
        f"  Slices/read  [bold]{args.num_slices_per_read}[/]\n"
        f"  Max slices   [bold]{args.max_slices if args.max_slices > 0 else 'unlimited'}[/]\n"
        f"  Seed         [bold]{args.seed}[/]"
    )

    os.makedirs(args.output, exist_ok=True)
    rng = np.random.default_rng(seed=args.seed)

    pod5_dr = pod5.DatasetReader(args.input)

    console.print("[dim]Phase 1 — Counting reads…[/]")
    read_count = 0
    for _ in pod5_dr.reads():
        read_count += 1
    console.print(f"  Found [bold]{read_count:,}[/] reads")

    console.print("[dim]Phase 2 — Extracting signal slices…[/]")
    slices = []
    skipped_short = 0
    total_collected = 0
    max_slices = args.max_slices if args.max_slices > 0 else float('inf')

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
    ) as progress:
        task = progress.add_task("Extracting", total=read_count)

        for read in pod5_dr.reads():
            progress.update(task, advance=1)

            if total_collected >= max_slices:
                break

            signal = read.signal  # int16 ndarray
            sig_len = len(signal)

            if sig_len < args.slice_len:
                skipped_short += 1
                continue

            max_possible = sig_len - args.slice_len + 1
            n_to_take = min(args.num_slices_per_read, max_possible)
            n_to_take = min(n_to_take, int(max_slices - total_collected))

            if n_to_take <= 0:
                continue

            if n_to_take <= max_possible:
                starts = rng.choice(max_possible, size=n_to_take, replace=False)
            else:
                starts = rng.choice(max_possible, size=n_to_take, replace=True)

            for s in starts:
                slices.append(signal[s: s + args.slice_len].astype(np.int16))
                total_collected += 1

    console.print("[dim]Phase 3 — Shuffling and saving…[/]")
    perm = rng.permutation(len(slices))
    signal_raw = np.stack([slices[i] for i in perm], axis=0)  # (N, slice_len) int16

    out_path = os.path.join(args.output, 'signal_raw.npy')
    np.save(out_path, signal_raw)

    tbl = Table(
        title="✅ Dataset Summary",
        box=box.ROUNDED, border_style="green", title_style="bold green",
    )
    tbl.add_column("Metric", style="bold")
    tbl.add_column("Value", justify="right")
    tbl.add_row("Total reads", f"{read_count:,}")
    tbl.add_row("Skipped (too short)", f"{skipped_short:,}")
    tbl.add_row("Slices extracted", f"{len(signal_raw):,}")
    tbl.add_row("Slice length", f"{args.slice_len:,}")
    tbl.add_row("Shape", f"{signal_raw.shape}")
    tbl.add_row("File size", f"{os.path.getsize(out_path) / 1e6:.2f} MB")
    console.print(tbl)
    console.print(f"[bold green]✓ Saved to {out_path}[/]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare fixed-length signal slices from POD5 for SigComp training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-i', '--input', required=True,
                        help='Input .pod5 file or directory')
    parser.add_argument('-o', '--output', required=True,
                        help='Output directory (will contain signal_raw.npy)')
    parser.add_argument('--slice-len', type=int, default=2000,
                        help='Length of each signal slice (samples)')
    parser.add_argument('--num-slices-per-read', type=int, default=1,
                        help='Number of random slices to extract per read')
    parser.add_argument('--max-slices', type=int, default=-1,
                        help='Maximum total slices to collect (-1 = unlimited)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()
    prepare_dataset(args)
