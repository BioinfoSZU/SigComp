import argparse
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

try:
    from .version import __version__
except ImportError:
    __version__ = "0.0.0"


@dataclass(frozen=True)
class Command:
    module: str
    help: str


COMMANDS = {
    "multi-gpu": Command(
        module="multi_gpu",
        help="compress or decompress with all or selected GPUs",
    ),
    "multi-cpu-socket": Command(
        module="multi_cpu_socket",
        help="compress or decompress with all or selected CPU sockets",
    ),
    "compressor": Command(
        module="main",
        help="compress, decompress, inspect, index, and view SIGCOMP files",
    ),
    "dataset": Command(
        module="prepare_dataset",
        help="prepare fixed-length POD5 signal slices for training",
    ),
    "train": Command(
        module="train",
        help="train the SigComp MLP entropy model",
    ),
    "ptq": Command(
        module="ptq",
        help="run post-training quantization for a trained model",
    ),
    "check": Command(
        module="result_validation",
        help="validate decompressed POD5 output against the original POD5",
    ),
}


class DefaultAndRawFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


def _package_name() -> str:
    if __package__:
        return __package__

    package_dir = Path(__file__).resolve().parent
    package_parent = str(package_dir.parent)
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)
    return package_dir.name


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sigcomp",
        description="SigComp command line interface",
        formatter_class=DefaultAndRawFormatter,
        epilog=(
            "Examples:\n"
            "  sigcomp compressor compress -i reads.pod5 -o reads.sigcomp --device cpu\n"
            "  sigcomp compressor decompress -i reads.sigcomp -o reads.decompressed.pod5 --device cuda\n"
            "  sigcomp compressor inspect -i reads.sigcomp\n"
            "  sigcomp dataset -i HG002.pod5 -o dataset_dir\n"
            "  sigcomp train --dataset dataset_dir --output-dir runs/model\n"
            "  sigcomp ptq --calibration-dataset dataset_dir --FP32-model fp32.pth --output-model int8.pth\n"
            "  sigcomp check --pod5 reads.pod5 --decomp reads.decompressed.pod5\n\n"
            "Use 'sigcomp <command> --help' to show command-specific options."
        ),
    )
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"sigcomp {__version__}",
        help="show the SigComp version and exit",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="<command>",
        title="commands",
    )

    for name, command in COMMANDS.items():
        subparser = subparsers.add_parser(
            name,
            add_help=False,
            help=command.help,
            description=command.help,
        )
        subparser.add_argument(
            "args",
            nargs=argparse.REMAINDER,
            help=f"arguments passed to the {name} command",
        )
        subparser.set_defaults(command_name=name, module=command.module)

    return parser


def _run_module(command_name: str, module: str, args: Sequence[str]) -> None:
    package = _package_name()
    old_argv = sys.argv[:]
    try:
        sys.argv = [f"sigcomp {command_name}", *args]
        runpy.run_module(f"{package}.{module}", run_name="__main__", alter_sys=False)
    finally:
        sys.argv = old_argv


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv:
        parser.print_help()
        return 0

    first = argv[0]

    if first in {"-h", "--help", "-V", "--version"}:
        parser.parse_args(argv[:1])
        return 0

    command = COMMANDS.get(first)
    if command is None:
        choices = ", ".join(COMMANDS)
        parser.error(
            f"argument <command>: invalid choice: {first!r} "
            f"(choose from {choices})"
        )

    _run_module(first, command.module, argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
