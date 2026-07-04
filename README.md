SigComp: A learned compressor for nanopore sequencing signal data
-----

SigComp is a learned compressor for nanopore sequencing signal data, currently support ONT R10.4.1 and RNA004 datasets.
It compresses signal data from POD5 / BLOW5 (SLOW5) files into a `.sigcomp` file using a quantized MLP entropy model losslessly.

## 🚀 Download and Installation

### System Requirements

  * **GPU:** NVIDIA GPU with CUDA compute capability \>= 8.x (e.g., Ampere, Ada, or Hopper GPUs like A100, RTX 3090, RTX 4090, H100)
  * **Driver:** NVIDIA driver version \>= 525

SigComp is compatible with Linux and has been fully tested on Ubuntu 22.04.

### Install from conda and pip

We recommend installing SigComp in a [conda](https://www.anaconda.com/download/success) environment.

```shell
conda create -n sigcomp python=3.10
conda activate sigcomp

sudo apt-get install libzstd-dev
export PYSLOW5_ZSTD=1  # for pyslow5 with zstd enabled 
pip install sigcomp -i https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu126
```

**Note:** The pre-trained model checkpoints are bundled inside the package (`sigcomp/pretrained_models/`), so no separate download is required. You can verify the installation with:

```shell
sigcomp --version
sigcomp compressor -h
```

-----

## ⚙️ CLI Overview

SigComp provides a unified command-line interface with five subcommands:

```text
sigcomp <command> [options]
# equivalently:
python -m sigcomp <command> [options]

Commands:
  compressor    Compress, decompress, inspect, index, and view SIGCOMP files
  dataset       Prepare fixed-length POD5 signal slices for training
  train         Train the SigComp MLP entropy model
  ptq           Run post-training quantization for a trained model
  check         Validate decompressed POD5 output against the original POD5
```

Use `-h` with any subcommand for detailed help:
```shell
sigcomp compressor -h
sigcomp dataset -h
sigcomp train -h
sigcomp ptq -h
sigcomp check -h
```

-----

## 🗜️ Compression (`compressor`)

The `compressor` subcommand groups the five file operations: `compress`, `decompress`, `inspect`, `index`, and `view`.

```shell
sigcomp compressor <compress|decompress|inspect|index|view> [options]
```

### Compress (`compress`)

Compress nanopore signal data from a POD5 or BLOW5/SLOW5 file into a `.sigcomp` archive (All read metadata in original POD5 or BLOW5 file are also stored in the archive).

```text
usage: sigcomp compressor compress [-h] -i INPUT -o OUTPUT [-m MODEL_PATH] [-g GPU] [-b BATCH_SIZE] [-B BATCH_NUMBER] [-L CHUNK_LEN] [-c CHUNK_SIZE]
                                   [-n MAX_READS] [--model-embed {path,embed,none}] [--dim DIM] [--num-dec-layers NUM_DEC_LAYERS]
                                   [--window-size WINDOW_SIZE] [--lsb-eliminate LSB_ELIMINATE]

options:
  -h, --help            show this help message and exit
  -i, --input INPUT     Input signal file (.pod5 or .blow5/.slow5)
  -o, --output OUTPUT   Output .sigcomp file
  -m, --model-path MODEL_PATH
                        Path to quantized MLP model weights (.pth). If omitted, the model is auto-selected from
                        --lsb-eliminate: 0 → lossless model, 3 → lossy model.
  -g, --gpu GPU         GPU device ID (default: 0)
  -b, --batch-size BATCH_SIZE
                        GPU batch size for neural coding (default: 20480)
  -B, --batch-number BATCH_NUMBER
                        The number of GPU compact batches for neural coding (default: 64)
  -L, --chunk-len CHUNK_LEN
                        Token chunk length for neural coding (default: 2048)
  -c, --chunk-size CHUNK_SIZE
                        Number of reads per IO flush batch. 0 = auto-tune from signal lengths and batch_size. (default: 0)
  -n, --max-reads MAX_READS
                        Max reads to compress (-1 = all) (default: -1)
  --model-embed {path,embed,none}
                        How to store model reference in compressed file (default: path)
  --dim DIM             MLP hidden dimension (default: 256)
  --num-dec-layers NUM_DEC_LAYERS
                        Number of MLP decoder layers (default: 8)
  --window-size WINDOW_SIZE
                        Context window size (number of previous tokens) (default: 64)
  --lsb-eliminate LSB_ELIMINATE
                        Lossy LSB reduction bits (0=lossless; 3=lossy) (default: 0)
```

#### Lossless vs. lossy

During `compress`, if no custom `--model-path` is provided, the model is **auto-selected** from `--lsb-eliminate`:

| `--lsb-eliminate` | Model          | Quantization for raw signal                                                               |
|-------------------|----------------|-------------------------------------------------------------------------------------------|
| `0` (default)     | `lossless.pth` | N/A                                                                                       | 
| `3`               | `lossy.pth`    | adopt from [lossy ex-zd method](https://hasindu2008.github.io/slow5tools/bits-lossy.html) | 

#### Examples

```shell
# Lossless compression given a POD5 file (uses the default lossless model)
sigcomp compressor compress -i reads.pod5 -o reads.sigcomp

# Lossless compression given a BLOW5/SLOW5 file
sigcomp compressor compress -i reads.blow5 -o reads.sigcomp

# Lossy compression (auto-selects the default lossy model)
sigcomp compressor compress -i reads.pod5 -o reads.sigcomp --lsb-eliminate 3

# Use a specific GPU and a custom model checkpoint
sigcomp compressor compress -i reads.pod5 -o reads.sigcomp -m my_model.pth -g 1
```

### Decompress (`decompress`)

Restore a `.sigcomp` file back to POD5, BLOW5, or a simple raw binary (containing only signals). The output format is detected from the file extension.

```text
usage: sigcomp compressor decompress [-h] -i INPUT -o OUTPUT [-m MODEL_PATH] [-g GPU] [-b BATCH_SIZE] [-B BATCH_NUMBER] [-c CHUNK_SIZE]
                                     [-n MAX_READS] [--rec-press {none,zlib,zstd}] [--sig-press {none,svb-zd,ex-zd}]

options:
  -h, --help            show this help message and exit
  -i, --input INPUT     Input .sigcomp file
  -o, --output OUTPUT   Output file. Format detected by extension: .pod5 → POD5, .blow5/.slow5 → BLOW5,
                        .bin → raw binary [uint32 len][int16[] signal]
  -m, --model-path MODEL_PATH
                        Override model path (uses header info by default) (default: None)
  -g, --gpu GPU         GPU device ID (default: 0)
  -b, --batch-size BATCH_SIZE
                        GPU batch size for neural decoding (default: 20480)
  -B, --batch-number BATCH_NUMBER
                        The number of GPU compact batches for neural decoding (default: 8)
  -c, --chunk-size CHUNK_SIZE
                        Number of reads per IO flush batch. 0 = auto-tune from signal lengths and batch_size. (default: 0)
  -n, --max-reads MAX_READS
                        Max reads to decompress (-1 = all) (default: -1)
  --rec-press {none,zlib,zstd}
                        BLOW5 record compression (only for .blow5 output) (default: zlib)
  --sig-press {none,svb-zd,ex-zd}
                        BLOW5 signal compression (only for .blow5 output) (default: svb-zd)
```

#### Examples

```shell
# Decompress back to POD5
sigcomp compressor decompress -i reads.sigcomp -o reads.decompressed.pod5

# Decompress to BLOW5 with zstd record compression
sigcomp compressor decompress -i reads.sigcomp -o reads.decompressed.blow5 --rec-press zstd

# Decompress to raw binary
sigcomp compressor decompress -i reads.sigcomp -o reads.decompressed.bin
```

> **⚠️ Note — SigComp only support decompress to the original source format.**
> SigComp preserves the source file's metadata structure so it can only decompress back to the **same format it was compressed from**, or to a raw `.bin` signal dump.

### Inspect / Index / View

```text
usage: sigcomp compressor inspect [-h] -i INPUT [--no-scan]
usage: sigcomp compressor index   [-h] -i INPUT
usage: sigcomp compressor view    [-h] -i INPUT [-o OUTPUT] [-m MODEL_PATH] [-g GPU] [-b BATCH_SIZE]
                                  [--read-ids READ_IDS] [--head HEAD] [--tail TAIL] [--range RANGE]
```

```shell
# Show file metadata & compression metrics statistics
sigcomp compressor inspect -i reads.sigcomp

# Build a random-access index (.idx) to enable `view`
sigcomp compressor index -i reads.sigcomp

# View the first 5 reads summary on the console (requires an index)
sigcomp compressor view -i reads.sigcomp --head 5

# Export the first 10 reads to a POD5 subset
sigcomp compressor view -i reads.sigcomp -o subset.pod5 --head 10

# View specific reads by UUID to a POD5 subset 
sigcomp compressor view -i reads.sigcomp --read-ids <uuid1>,<uuid2> -o subset.pod5 
```

-----

## 📦 Training Dataset Preparation (`dataset`)

Extract fixed-length signal slices from POD5 files to build a training dataset for the SigComp MLP entropy model.

```text
usage: sigcomp dataset [-h] -i INPUT -o OUTPUT [--slice-len SLICE_LEN] [--num-slices-per-read NUM_SLICES_PER_READ]
                       [--max-slices MAX_SLICES] [--seed SEED]

options:
  -h, --help            show this help message and exit
  -i, --input INPUT     Input .pod5 file or directory
  -o, --output OUTPUT   Output directory (will contain signal_raw.npy)
  --slice-len SLICE_LEN
                        Length of each signal slice (samples) (default: 2000)
  --num-slices-per-read NUM_SLICES_PER_READ
                        Number of random slices to extract per read (default: 1)
  --max-slices MAX_SLICES
                        Maximum total slices to collect (-1 = unlimited) (default: -1)
  --seed SEED           Random seed for reproducibility (default: 42)
```

#### Example

```shell
sigcomp dataset -i HG002.pod5 -o dataset_dir --slice-len 2000 --num-slices-per-read 4 --max-slices 3000000 
```

-----

## 🏋️ Training (`train`)

Train the SigComp MLP entropy model on a prepared dataset.

```text
usage: sigcomp train [-h] [--in-dim IN_DIM] [--dim DIM] [--layers LAYERS] [--window-size WINDOW_SIZE] [--quant]
                     --dataset DATASET --output-dir OUTPUT_DIR [--batch-size BATCH_SIZE] [--num-epochs NUM_EPOCHS]
                     [--num-workers NUM_WORKERS] [--lr LR] [--weight-decay WEIGHT_DECAY] [--max-norm MAX_NORM]
                     [--gpu GPU] [--lsb-eliminate LSB_ELIMINATE]

options:
  -h, --help            show this help message and exit
  --in-dim IN_DIM       Embedding input dimension (default: 128)
  --dim DIM             MLP hidden dimension (default: 256)
  --layers LAYERS       Number of MLP decoder layers (default: 8)
  --window-size WINDOW_SIZE
                        Context window size (number of previous tokens) (default: 64)
  --quant               Enable quantization-aware training (default: False)
  --dataset DATASET     Training dataset directory (from `sigcomp dataset`)
  --output-dir OUTPUT_DIR
                        Output directory for logs and weights
  --batch-size BATCH_SIZE
                        Batch size (default: 256)
  --num-epochs NUM_EPOCHS
                        Number of epochs (default: 10)
  --num-workers NUM_WORKERS
                        Number of data-loader workers (default: 1)
  --lr LR               Learning rate (default: 0.002)
  --weight-decay WEIGHT_DECAY
                        Weight decay (default: 0.01)
  --max-norm MAX_NORM   Gradient clipping max-norm (default: 1.0)
  --gpu GPU             GPU device ID (default: 0)
  --lsb-eliminate LSB_ELIMINATE
                        Lossy LSB reduction bits (0=lossless). Applies BLOW5 lossy bit-elimination to signals
                        before delta-zigzag so the model learns the reduced distribution. (default: 0)
```

#### Example

```shell
# Train a lossless model
sigcomp train --dataset dataset_dir --output-dir runs/model --num-epochs 10

# Train a lossy model (matching --lsb-eliminate 3)
sigcomp train --dataset dataset_dir --output-dir runs/model_lossy --num-epochs 10 --lsb-eliminate 3
```

-----

## 🎯 Post-Training Quantization (`ptq`)

Quantize a trained FP32 model to INT8 using a calibration dataset.

```text
usage: sigcomp ptq [-h] --calibration-dataset CALIBRATION_DATASET --FP32-model FP32_MODEL --output-model OUTPUT_MODEL
                   [--batch-size BATCH_SIZE] [--window-size WINDOW_SIZE] [--in-dim IN_DIM] [--dim DIM]
                   [--num-layers NUM_LAYERS] [--lsb-eliminate LSB_ELIMINATE]

options:
  -h, --help            show this help message and exit
  --calibration-dataset CALIBRATION_DATASET
                        Calibration dataset directory
  --FP32-model FP32_MODEL
                        FP32 model checkpoint path
  --output-model OUTPUT_MODEL
                        Output INT8 model checkpoint path
  --batch-size BATCH_SIZE
                        Calibration batch size (default: 64)
  --window-size WINDOW_SIZE
                        Context window size (default: 64)
  --in-dim IN_DIM       Embedding input dimension (default: 128)
  --dim DIM             MLP hidden dimension (default: 256)
  --num-layers NUM_LAYERS
                        Number of MLP decoder layers (default: 8)
  --lsb-eliminate LSB_ELIMINATE
                        Lossy LSB reduction bits (0=lossless) (default: 0)
```

#### Example

```shell
sigcomp ptq \
  --calibration-dataset dataset_dir \
  --FP32-model runs/model/fp32.pth \
  --output-model runs/model/int8.pth
```

-----

## ✅ Validation (`check`)

Validate that a decompressed POD5 file matches the original POD5 (including signal array, read IDs and other metadata).

```text
usage: sigcomp check [-h] --pod5 POD5 --decomp DECOMP [--sample-n SAMPLE_N] [--sample-ratio SAMPLE_RATIO] [--head HEAD]

options:
  -h, --help            show this help message and exit
  --pod5 POD5           Path to the original POD5 file
  --decomp DECOMP       Path to the decompressed POD5 file
  --sample-n SAMPLE_N   Validate randomly sampled N reads (default: None)
  --sample-ratio SAMPLE_RATIO
                        Validate randomly sampled ratio of reads (0.0 to 1.0) (default: None)
  --head HEAD           Validate only the first HEAD reads (default: None)
```

#### Example

```shell
# Full round-trip check
sigcomp compressor compress   -i reads.pod5    -o reads.sigcomp
sigcomp compressor decompress -i reads.sigcomp -o reads.decompressed.pod5
sigcomp check --pod5 reads.pod5 --decomp reads.decompressed.pod5

# Validate a random 10% sample
sigcomp check --pod5 reads.pod5 --decomp reads.decompressed.pod5 --sample-ratio 0.1
```

-----

## 🙏 Acknowledgement

Our GPU-accelerated range coder is a [Triton](https://github.com/triton-lang/triton) reimplementation of the range coder from the [constriction](https://github.com/bamler-lab/constriction) library.
We additionally use [pod5](https://github.com/nanoporetech/pod5-file-format) / [pyslow5](https://github.com/hasindu2008/slow5lib) for reading and writing nanopore signal files.

## ©️ Copyright

Copyright 2026 Zexuan Zhu <zhuzx@szu.edu.cn>.<br>
This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.
