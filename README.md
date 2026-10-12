SigComp: A learned compressor for nanopore sequencing signal data
-----

SigComp compresses nanopore signal data using an integer-quantized MLP entropy model. It supports lossless and lossy compression of signal data from POD5 and BLOW5/SLOW5 files. The bundled pretrained models target ONT R10.4.1 and RNA004 data.

Compression and decompression can run on a CPU or GPU. Multi-GPU and multi-CPU-socket launchers distribute reads across devices. CPU and GPU backends use the same archive format, allowing compression on one backend and decompression on another with the matching model.

## 🚀 Download and Installation

### System Requirements

1. **All platforms:** Linux x86-64; Python 3.10–3.12 and PyTorch 2.8 or newer.
2. **NVIDIA GPU:** Ampere or newer GPU with CUDA compute capability \>= 8.x. The Linux driver version should be compatible with CUDA 12.6 or later (e.g, **560.28.03**). 
3. **AMD GPU:** a ROCm-supported GPU, such as AMD Instinct MI210, MI250 or MI300X, with AMD GPU Driver compatible with ROCm 6.4 or newer. AMD does not specify one universal minimum driver across all GPUs, operating systems and ROCm versions; select a supported combination from the [AMD driver matrix](https://instinct.docs.amd.com/projects/amdgpu-docs/en/docs-30.10/compatibility/compatibility-matrix.html). For Radeon RX 7900 XTX / PRO W7900, an example baseline is [Radeon Software for Linux 25.10.1 with ROCm 6.4.1](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-UNIFIED-LINUX-25-10-1-ROCM-6-4-1.html).

### Install from conda and pip

Create a [conda](https://www.anaconda.com/download/success) environment for installing SigComp.

```shell
conda create -n sigcomp python=3.11 -y
conda activate sigcomp
python -m pip install --upgrade pip

# Ubuntu / Debian: system libraries and compiler for Python extension builds
sudo apt-get update
sudo apt-get install -y build-essential zlib1g-dev libzstd-dev libnuma1

# Red Hat / Rocky Linux: use this instead of the apt-get commands above
# sudo dnf install -y gcc gcc-c++ make zlib-devel libzstd-devel numactl-libs

export PYSLOW5_ZSTD=1
```

Then, install **one** of the following backend configurations. 

```shell
# CPU-only platform
python -m pip install 'sigcomp>=1.1.0' 'torch==2.8.0+cpu' \
  --extra-index-url https://download.pytorch.org/whl/cpu --no-binary=pyslow5

# OR NVIDIA GPU platform
python -m pip install 'sigcomp>=1.1.0' 'torch==2.8.0+cu126' \
  --extra-index-url https://download.pytorch.org/whl/cu126 --no-binary=pyslow5

# OR AMD GPU platform
python -m pip install 'sigcomp>=1.1.0' 'torch==2.8.0+rocm6.4' \
  --extra-index-url https://download.pytorch.org/whl/rocm6.4 --no-binary=pyslow5
```

Both pretrained models are included. Check the installation:

```shell
sigcomp --version
sigcomp --help
```

-----

## ⚙️ CLI Overview

Use `sigcomp <command>` or `python -m sigcomp <command>`:

| Command            | Purpose                                                                     |
|--------------------|-----------------------------------------------------------------------------|
| `compressor`       | Single CPU/GPU compression, decompression, inspection, indexing and viewing |
| `multi-gpu`        | Compression/decompression across selected or all visible GPUs               |
| `multi-cpu-socket` | Compression/decompression across selected or all available CPU sockets      |
| `dataset`          | Prepare training signal slices from POD5                                    |
| `train`            | Train the MLP entropy model                                                 |
| `ptq`              | Quantize a trained model to INT8                                            |
| `check`            | Compare original and restored POD5 reads                                    |

The direct aliases `sigcomp-codec`, `sigcomp-multi-gpu` and `sigcomp-multi-cpu-socket` are also installed. For example, `sigcomp-codec compress ...` is equivalent to `sigcomp compressor compress ...`.

-----

## 🗜️ Compression and Decompression (`compressor`)

**For routine compression/decompression, start with [multi-GPU](#multiple-gpus) or [multi-CPU-socket](#multiple-cpu-sockets).** These commands select available devices and automatically fall back to the ordinary codec when only one GPU/socket is available or selected. Use `compressor` below when you want direct control over one device.

### 1. Single GPU (NVIDIA or AMD)

NVIDIA CUDA and AMD ROCm use the same commands, including `--device cuda` and the logical GPU number `-g`. 

```shell
sigcomp compressor compress -i reads.pod5 -o reads.gpu.sigcomp --device cuda -g 0 

sigcomp compressor decompress -i reads.gpu.sigcomp -o reads.gpu.restored.pod5 --device cuda -g 0
```

### 2. CPU and socket-local execution

**Without a socket ID:** the default thread count is the number of available physical CPU cores across the inherited allocation. CPU affinity and memory policy remain inherited.

```shell
sigcomp compressor compress -i reads.pod5 -o reads.cpu.sigcomp --device cpu 

sigcomp compressor decompress -i reads.cpu.sigcomp -o reads.cpu.restored.pod5 --device cpu
```

**Select a socket, preserving the existing memory policy:** execution is restricted to available physical cores on the selected socket. No new memory-binding policy is applied.

```shell
sigcomp compressor compress \
  -i reads.pod5 -o reads.socket0.sigcomp --device cpu \
  --cpu-socket-id 0 --cpu-memory-policy inherit 

sigcomp compressor decompress \
  -i reads.socket0.sigcomp -o reads.socket0.restored.pod5 --device cpu \
  --cpu-socket-id 0 --cpu-memory-policy inherit
```

**Select a socket and bind memory:** execution uses the selected socket's available physical cores, and new memory allocations are bound to its allowed NUMA memory nodes.

```shell
sigcomp compressor compress \
  -i reads.pod5 -o reads.socket0.bound.sigcomp --device cpu \
  --cpu-socket-id 0 --cpu-memory-policy bind 

sigcomp compressor decompress \
  -i reads.socket0.bound.sigcomp -o reads.socket0.bound.restored.pod5 --device cpu \
  --cpu-socket-id 0 --cpu-memory-policy bind
```

List available socket IDs with `python -m sigcomp.cpu_socket --list`.

### 3. CPU ↔ GPU interoperability

```shell
# Compress on CPU, decompress on GPU
sigcomp compressor compress \
  -i reads.pod5 -o cpu.encoded.sigcomp --device cpu 
sigcomp compressor decompress \
  -i cpu.encoded.sigcomp -o gpu.decoded.pod5 --device cuda -g 0

# Compress on GPU, decompress on CPU
sigcomp compressor compress \
  -i reads.pod5 -o gpu.encoded.sigcomp --device cuda -g 0 
sigcomp compressor decompress \
  -i gpu.encoded.sigcomp -o cpu.decoded.pod5 --device cpu
```

### 4. Model selection

During `compress`, if no custom `--model-path` is provided, the model is **auto-selected** from `--lsb-eliminate`:

| `--lsb-eliminate` | Model          | Quantization for raw signal                                                               |
|-------------------|----------------|-------------------------------------------------------------------------------------------|
| `0` (default)     | `lossless.pth` | N/A                                                                                       | 
| `3`               | `lossy.pth`    | adopt from [lossy ex-zd method](https://hasindu2008.github.io/slow5tools/bits-lossy.html) | 

```shell
# Bundled lossless model (default)
sigcomp compressor compress \
  -i reads.pod5 -o reads.lossless.sigcomp --device cpu \
  --lsb-eliminate 0 --model-embed path 

# Bundled lossy model
sigcomp compressor compress \
  -i reads.pod5 -o reads.lossy.sigcomp --device cpu \
  --lsb-eliminate 3 --model-embed path

# Custom quantized model
sigcomp compressor compress \
  -i reads.pod5 -o reads.custom.sigcomp --device cpu \
  --model-path my_model.pth --model-embed embed

# Decoding reads the model and lossy setting from the archive
sigcomp compressor decompress \
  -i reads.lossy.sigcomp -o reads.lossy.pod5 --device cpu
```

For a custom lossy model, supply its matching `--lsb-eliminate` value during compression. Keep custom architecture settings consistent with training. The default `--model-embed path` stores a checkpoint path in the archive, while `--model-embed embed` stores the checkpoint binary into archive. `--model-path` can supply the matching checkpoint at decompression if that path is unavailable. 

### 5. Read selection

```shell
# Compress the first 100 reads
sigcomp compressor compress \
  -i reads.pod5 -o first100.sigcomp --device cpu -n 100

# Compress selected UUIDs: ids.txt contains one read UUID per line
sigcomp compressor compress \
  -i reads.pod5 -o selected.sigcomp --device cpu \
  --read-ids-file ids.txt 

# Extract selected UUIDs from an existing full archive
sigcomp compressor index -i reads.gpu.sigcomp
sigcomp compressor decompress \
  -i reads.gpu.sigcomp -o selected.pod5 --device cpu --read-ids-file ids.txt
```

`-n / --max-reads` also limits decompression. Indexed UUID selection avoids decoding the entire archive.

### 6. Decompression output format

SigComp only supports decompression to the original source format. SigComp preserves the source file's metadata structure, so an archive can be restored to its **original source format**, or to a raw `.bin` signal dump. POD5 and SLOW5/BLOW5 metadata are not converted between formats.

| Compression source  | Decompression output         |
|---------------------|------------------------------|
| `.pod5`             | `.pod5` or `.bin`            |
| `.blow5` / `.slow5` | `.blow5`, `.slow5` or `.bin` |

```shell
# Decompress back to POD5
sigcomp compressor decompress -i reads.sigcomp -o reads.decompressed.pod5 --device cpu 

# Decompress to BLOW5 with zstd record compression
sigcomp compressor decompress -i reads.sigcomp -o reads.decompressed.blow5 --rec-press zstd --device cpu 

# Decompress to raw binary without read metadata 
# Each `.bin` record is `[uint32 sample_count][int16 signal samples]`. 
sigcomp compressor decompress -i reads.sigcomp -o reads.decompressed.bin --device cpu 
```

### 7. Inspect, index and view an archive

```shell
# Inspect archive metadata and read statistics
sigcomp compressor inspect -i reads.gpu.sigcomp

# Build the read index
sigcomp compressor index -i reads.gpu.sigcomp

# Decode and display the first three reads
sigcomp compressor view -i reads.gpu.sigcomp --device cpu --head 3

# Decode and export the first three reads using a GPU
sigcomp compressor view \
  -i reads.gpu.sigcomp -o first3.pod5 --device cuda -g 0 --head 3
```

Use `sigcomp compressor <command> --help` for the complete option list.

-----

## 🖥️ Multi-GPU and Multi-Socket CPU

Both launchers accept one source file. POD5 compression uses read IDs and sample-count metadata to balance read assignments approximately by total signal samples. BLOW5/SLOW5 uses `pyslow5.get_read_ids()` and sequential assignments by read count, without load-balance during assignment. 

### Multiple GPUs

```shell
mkdir -p scratch
sigcomp multi-gpu compress \
  -i reads.pod5 -o reads.multi.sigcomp --gpus 0,1,2,3 \
  --temp-dir scratch --write-index --report gpu.compress.json

sigcomp multi-gpu decompress \
  -i reads.multi.sigcomp -o reads.gpu.pod5 --gpus 0,1,2,3 \
  --temp-dir scratch --report gpu.decompress.json
```

Omit `--gpus` to use all visible GPUs. AMD uses the same launcher after installing the matching ROCm stack.

### Multiple CPU sockets

```shell
sigcomp multi-cpu-socket compress \
  -i reads.pod5 -o reads.sockets.sigcomp --sockets 0,1 \
  --temp-dir scratch --write-index --report cpu.compress.json

sigcomp multi-cpu-socket decompress \
  -i reads.sockets.sigcomp -o reads.cpu.pod5 --sockets 0,1 \
  --temp-dir scratch --report cpu.decompress.json
```

Omit `--sockets` to use all available physical sockets. Each worker binds to physical cores and memory nodes on its own socket (`--cpu-memory-policy bind` by default).

When exactly one GPU/socket is available or selected, the launcher invokes the ordinary codec in the same process, with no worker process, read partitioning or merge phase.

### Outputs and optional manual merging

Compression produces one `.sigcomp` archive. Decompression leaves independent output files by default. For `-o reads.gpu.pod5`, the decompressed output is:

```text
reads.gpu.pod5.parts/
├── part-0000.pod5
├── part-0001.pod5
├── ...
└── manifest.json
```

The nominal `reads.gpu.pod5` file is not created. BLOW5/SLOW5/BIN use the same `.parts` layout with their respective extensions. The manifest records status, read ranges and output paths. To merge later, run the native tools by yourself (NOTE: merging can require substantial extra RAM, time and storage; it is optional when downstream tools accept multiple files):

```shell
# merge POD5 outputs into one file 
pod5 merge reads.gpu.pod5.parts/part-*.pod5 -o reads.gpu.pod5 -t 8 

# BLOW5 merging example 
conda install -c conda-forge -c bioconda slow5tools
slow5tools merge reads.cpu.blow5.parts/part-*.blow5 \
  -o reads.cpu.blow5 --to blow5 -t 8 -c zlib -s svb-zd
```

-----

## 📦 Dataset Preparation (`dataset`)

Extract fixed-length random signal slices from a POD5 file or directory. The output directory contains `signal_raw.npy` for training/calibration.

```shell
sigcomp dataset -i HG002.pod5 -o dataset_dir \
  --slice-len 2000 --num-slices-per-read 4 --max-slices 3000000
```

Use `--seed` for reproducible sampling; `--max-slices -1` removes the total-slice limit. Run `sigcomp dataset --help` for all options.

-----

## 🏋️ Training (`train`)

Train the SigComp MLP entropy model on a prepared dataset.

```shell
# Lossless model
sigcomp train --dataset dataset_dir --output-dir runs/model --num-epochs 10 --gpu 0

# Lossy model
sigcomp train --dataset dataset_dir --output-dir runs/model_lossy \
  --num-epochs 10 --gpu 0 --lsb-eliminate 3
```

Checkpoints are saved under `weights/epoch_N.checkpoint.pth` in the output directory. Run `sigcomp train --help` for batch, learning-rate and architecture options.

-----

## 🎯 Post-Training Quantization (`ptq`)

Quantize a trained FP32 checkpoint using a calibration dataset:

```shell
sigcomp ptq \
  --calibration-dataset dataset_dir \
  --FP32-model runs/model/weights/epoch_10.checkpoint.pth \
  --output-model runs/model/int8.pth

sigcomp compressor compress \
  -i reads.pod5 -o reads.custom.sigcomp --device cpu \
  -m runs/model/int8.pth 
```

Keep architecture settings and `--lsb-eliminate` consistent between training, calibration and compression. For a lossy model, add `--lsb-eliminate 3` to PTQ and compression. Run `sigcomp ptq --help` for calibration options.

-----

## ✅ Validation (`check`)

Compare restored POD5 read IDs, supported metadata and signal arrays against the original POD5. This exact-equality check is intended for lossless output.

```shell
sigcomp check --pod5 reads.pod5 --decomp reads.restored.pod5

# Compare a random 10% of read records
sigcomp check --pod5 reads.pod5 --decomp reads.restored.pod5 --sample-ratio 0.1
```

The checker compares read-ID sets before selecting records to validate. BLOW5/SLOW5 validation is not provided by this command.

-----

## 🙏 Acknowledgement

Our GPU-accelerated range coder is a [Triton](https://github.com/triton-lang/triton) reimplementation of the range coder from [constriction](https://github.com/bamler-lab/constriction). We use [pod5](https://github.com/nanoporetech/pod5-file-format) and [pyslow5](https://github.com/hasindu2008/slow5lib) for nanopore file I/O.

## ©️ Copyright

Copyright 2026 Zexuan Zhu <zhuzx@szu.edu.cn>.<br>
This project is licensed under the Apache License 2.0. See the [LICENSE](https://github.com/BioinfoSZU/SigComp/blob/main/LICENSE) file for details.
