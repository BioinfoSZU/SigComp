import os
import sys
import struct
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as quant
import torch.ao.nn.intrinsic.quantized
import triton
import triton.language as tl
from tqdm import tqdm
from pathlib import Path
from .model import QAT_MLP_Compression
from .gpu_entropy_coding import (
    GpuRangeDecoder, GpuRangeStepEncoder, build_integer_exp_lut,
)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def bit_exact_linear_kernel(
        x_ptr, w_ptr, b_ptr, out_ptr,
        M0, shift, has_relu: tl.constexpr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_outm, stride_outn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_bn[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0)
        accumulator += tl.dot(x, w, out_dtype=tl.int32)
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk

    b = tl.load(b_ptr + offs_bn, mask=offs_bn < N, other=0)
    accumulator += b[None, :]

    acc_64 = accumulator.to(tl.int64) * M0
    acc_scaled = acc_64 >> shift

    min_val = 0 if has_relu else -128
    acc_clamped = tl.maximum(acc_scaled, min_val)
    acc_clamped = tl.minimum(acc_clamped, 127)

    out = acc_clamped.to(tl.int8)
    offs_outm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_outn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + stride_outm * offs_outm[:, None] + stride_outn * offs_outn[None, :]
    mask = (offs_outm[:, None] < M) & (offs_outn[None, :] < N)
    tl.store(out_ptrs, out, mask=mask)


class FastBitExactInt8Linear(nn.Module):
    def __init__(self, weight_q, bias_q, M0, shift, has_relu=False):
        super().__init__()
        self.register_buffer('weight_q', weight_q.contiguous().clone())
        self.register_buffer('bias_q', bias_q.contiguous().clone())
        self.M0 = int(M0)
        self.shift = int(shift)
        self.has_relu = has_relu

    def forward(self, x_q):
        orig_shape = x_q.shape
        x_flat = x_q.view(-1, orig_shape[-1]).contiguous()
        M, K = x_flat.shape
        N, K_w = self.weight_q.shape
        out = torch.empty((M, N), device=x_q.device, dtype=torch.int8)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
        bit_exact_linear_kernel[grid](
            x_flat, self.weight_q, self.bias_q, out, self.M0, self.shift, self.has_relu,
            M, N, K, x_flat.stride(0), x_flat.stride(1),
            self.weight_q.stride(0), self.weight_q.stride(1), out.stride(0), out.stride(1)
        )
        return out.view(*orig_shape[:-1], N)


class BitExactInt8Embedding(nn.Module):
    def __init__(self, weight_q_int8):
        super().__init__()
        self.register_buffer('weight_q', weight_q_int8.clone())

    def forward(self, input_ids):
        return F.embedding(input_ids, self.weight_q)  # type: ignore


class GPUBitExactMLPDecoder(nn.Module):
    def __init__(self, vocab_size: int, window_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.embedding = None
        self.mlp = nn.ModuleList()
        self.register_buffer('output_scale', torch.tensor(1.0, dtype=torch.float32))

    def forward(self, targets):
        B, L = targets.shape
        pad = targets[:, :1].expand(-1, self.window_size - 1)
        padded = torch.cat([pad, targets], dim=1)
        windows = padded.unfold(dimension=1, size=self.window_size, step=1)
        embs_q = self.embedding(windows)  # noqa
        x = embs_q.view(B, L, -1)
        for layer in self.mlp:
            x = layer(x)
        return x

    def _ar_step(self, context_window):
        embs_q = self.embedding(context_window)  # noqa
        x = embs_q.view(context_window.size(0), -1)
        for layer in self.mlp:
            x = layer(x)
        return x.unsqueeze(1)

    def dequantize_logits(self, int8_logits):
        return int8_logits.float() * self.output_scale


def float_to_multiplier_shift(m):
    if m == 0:
        return 0, 0
    significand, exponent = math.frexp(m)
    m0 = int(round(significand * (2 ** 31)))
    shift = 31 - exponent
    return m0, shift


def extract_embedding_to_int8(ptq_embedding_layer, quant_stub):
    fp32_weights = ptq_embedding_layer.weight.data
    input_scale = quant_stub.scale.item()
    q_weights = torch.round(fp32_weights / input_scale)
    q_weights = torch.clamp(q_weights, -128, 127).to(torch.int8)
    return q_weights, input_scale


def extract_to_gpu(q_layer, input_scale, device='cuda'):
    weight_q_cpu = q_layer.weight().int_repr()
    bias_fp32_cpu = q_layer.bias()

    weight_scale = q_layer.weight().q_scale()
    output_scale = q_layer.scale

    bias_scale = input_scale * weight_scale
    if bias_fp32_cpu is not None:
        bias_q_cpu = torch.round(bias_fp32_cpu / bias_scale).to(torch.int32)
    else:
        bias_q_cpu = torch.zeros(weight_q_cpu.shape[0], dtype=torch.int32)

    M = (input_scale * weight_scale) / output_scale
    M0, shift = float_to_multiplier_shift(M)

    return weight_q_cpu.to(device), bias_q_cpu.to(device), M0, shift, output_scale


def convert_ptq_to_gpu_pipeline(cpu_ptq_model: QAT_MLP_Compression, vocab_size, window_size, device='cuda'):
    gpu_model = GPUBitExactMLPDecoder(vocab_size, window_size)

    embed_q_weight, input_scale = extract_embedding_to_int8(
        cpu_ptq_model.decoder.embedding_linear,
        cpu_ptq_model.decoder.quant
    )
    gpu_model.embedding = BitExactInt8Embedding(embed_q_weight.to(device))

    current_scale = input_scale
    for layer in cpu_ptq_model.decoder.mlp:
        if isinstance(layer, nn.Identity):
            continue  # PyTorch puts nn.Identity in place of ReLU after fusion

        has_relu = isinstance(layer, torch.ao.nn.intrinsic.quantized.LinearReLU)
        w_gpu, b_gpu, m0, shift, current_scale = extract_to_gpu(layer, current_scale, device)
        gpu_model.mlp.append(FastBitExactInt8Linear(w_gpu, b_gpu, m0, shift, has_relu=has_relu))

    gpu_model.output_scale.fill_(current_scale)

    return gpu_model


MODEL_DIR = Path(__file__).parent / "pretrained_models"
DEFAULT_QUANT_LOSSLESS_MODEL_PATH = os.path.join(MODEL_DIR, "lossless.pth")
DEFAULT_QUANT_LOSSY_MODEL_PATH = os.path.join(MODEL_DIR, "lossy.pth")
DEFAULT_LSB_SETTING_FOR_LOSSY_MODEL = 3

DEFAULT_QUANT_MODEL_PATH = DEFAULT_QUANT_LOSSLESS_MODEL_PATH

class QuantMLPCompression(torch.nn.Module):
    DEFAULT_DIM = 256
    DEFAULT_NUM_DEC_LAYERS = 8
    DEFAULT_WINDOW_SIZE = 64

    def __init__(
            self,
            chunk_len=1024,
            gpu=0,
            model_path=None,
            dim=None,
            num_dec_layers=None,
            window_size=None,
    ):
        super().__init__()
        dim = dim or self.DEFAULT_DIM
        num_dec_layers = num_dec_layers or self.DEFAULT_NUM_DEC_LAYERS
        window_size = window_size or self.DEFAULT_WINDOW_SIZE
        in_dim = dim // 2

        quant_model = QAT_MLP_Compression(
            in_dim=in_dim,
            dim=dim,
            num_dec_layers=num_dec_layers,
            vocab_size=257,
            max_context_len=1024,
            window_size=window_size,
            enable_quant=True,
        )
        quant_model.eval()
        quant_model.decoder.fuse_model()

        qconfig = quant.QConfig(
            activation=quant.MovingAverageMinMaxObserver.with_args(dtype=torch.qint8,
                                                                   qscheme=torch.per_tensor_symmetric,
                                                                   reduce_range=False),
            weight=quant.MovingAverageMinMaxObserver.with_args(dtype=torch.qint8,
                                                               qscheme=torch.per_tensor_symmetric
                                                               ),
        )
        quant_model.qconfig = qconfig
        quant_model.decoder.embedding_linear.qconfig = None

        quant.prepare(quant_model, inplace=True)
        quant.convert(quant_model, inplace=True)
        PTQ_weight_path = model_path or DEFAULT_QUANT_MODEL_PATH
        quant_model.load_state_dict(torch.load(PTQ_weight_path, weights_only=True))

        self.gpu = gpu
        self.device = f'cuda:{gpu}'
        torch.cuda.set_device(self.device)  # ensure Triton autotuner runs on the correct GPU
        self.SOS_token = 256
        self.window_size = window_size
        self.chunk_len = chunk_len

        self.gpu_engine = convert_ptq_to_gpu_pipeline(quant_model, vocab_size=257, window_size=window_size, device=self.device)
        self.gpu_engine.eval()
        output_scale = float(self.gpu_engine.output_scale)  # type: ignore
        self._exp_lut = build_integer_exp_lut(output_scale).to(self.device)
        print(f'Load models successfully. (GPU entropy coding enabled)')

    def compress_chunks(self, chunks, max_batch_size=64, use_perfect=False, use_tqdm=False):
        return self._compress_chunks_gpu_ar(chunks, max_batch_size, use_tqdm)

    def _compress_chunks_gpu_ar(self, chunks, max_batch_size=64, use_tqdm=False):
        self.gpu_engine.eval()
        all_compressed_bytes = []

        progress_iter = range(0, len(chunks), max_batch_size)
        if use_tqdm:
            progress_iter = tqdm(progress_iter, desc='Compressing chunks (AR GPU)')

        for i in progress_iter:
            batch_chunks = chunks[i: i + max_batch_size]
            B = len(batch_chunks)
            valid_lens = [len(c) for c in batch_chunks]
            max_len_in_batch = max(valid_lens)

            gt_symbols = torch.full([B, self.chunk_len], fill_value=self.SOS_token,
                                    device=self.device, dtype=torch.long)
            for b in range(B):
                gt_symbols[b, :valid_lens[b]] = torch.from_numpy(
                    batch_chunks[b].astype(np.int64)
                )

            valid_lens_gpu = torch.tensor(valid_lens, dtype=torch.int64, device=self.device)

            context_buf = torch.full(
                [B, self.window_size + max_len_in_batch],
                fill_value=self.SOS_token, device=self.device, dtype=torch.long,
            )

            max_words = self.chunk_len + 32
            gpu_encoder = GpuRangeStepEncoder(B, max_words, device=self.device)

            with torch.no_grad():
                for t in range(max_len_in_batch):
                    context_window = context_buf[:, t:t + self.window_size]

                    int8_logits = self.gpu_engine._ar_step(context_window)
                    symbols_t = gt_symbols[:, t].contiguous()

                    gpu_encoder.encode_step_int8(
                        int8_logits.squeeze(1), self._exp_lut,
                        symbols_t, valid_lens_gpu, t
                    )

                    context_buf[:, t + self.window_size] = symbols_t

            gpu_encoder.seal()
            batch_bytes = gpu_encoder.get_compressed()
            all_compressed_bytes.extend(batch_bytes)

        return all_compressed_bytes

    def decompress_chunks(self, compressed_bytes_list, valid_lengths, max_batch_size=64, use_perfect=False,
                          use_tqdm=False):
        return self._decompress_chunks_gpu(compressed_bytes_list, valid_lengths, max_batch_size, use_tqdm)

    def _decompress_chunks_gpu(self, compressed_bytes_list, valid_lengths,
                               max_batch_size=64, use_tqdm=False):
        self.gpu_engine.eval()
        all_decompressed = []

        if use_tqdm:
            progress_bar = tqdm(range(0, len(compressed_bytes_list), max_batch_size),
                                desc='Decompressing chunks (GPU entropy)')
        else:
            progress_bar = range(0, len(compressed_bytes_list), max_batch_size)

        for i in progress_bar:
            batch_bytes = compressed_bytes_list[i: i + max_batch_size]
            batch_valid_lens = valid_lengths[i: i + max_batch_size]
            B = len(batch_bytes)

            decoder_cls = GpuRangeDecoder
            gpu_decoder = decoder_cls(batch_bytes, device=self.device)
            max_len_in_batch = int(max(batch_valid_lens))
            valid_lens_gpu = torch.tensor(batch_valid_lens, dtype=torch.int64,
                                          device=self.device)

            context_buf = torch.full(
                [B, self.window_size + max_len_in_batch],
                fill_value=self.SOS_token, device=self.device, dtype=torch.long,
            )

            with torch.no_grad():
                for t in range(max_len_in_batch):
                    context_window = context_buf[:, t:t + self.window_size]

                    int8_logits = self.gpu_engine._ar_step(context_window)

                    symbols = gpu_decoder.decode_step_int8(
                        int8_logits.squeeze(1), self._exp_lut,
                        valid_lens_gpu, t
                    )

                    context_buf[:, t + self.window_size] = symbols

            torch.cuda.synchronize()

            decoded_np = context_buf[:, self.window_size: self.window_size + max_len_in_batch].cpu().numpy()
            for b in range(B):
                all_decompressed.append(
                    decoded_np[b, :batch_valid_lens[b]].astype(np.uint8)
                )

        return all_decompressed

    def batch_inference_reads(
            self, list_of_reads=None, list_of_compressed_reads=None,
            list_of_lengths=None, batch_size=64, use_tqdm=False, prob_quant=0):
        if list_of_reads is not None:
            read_mapping = []
            chunks = []
            _map_range_st = 0
            for _, small_vals in enumerate(list_of_reads):
                num_chunks = math.ceil(len(small_vals) / self.chunk_len)
                for i in range(num_chunks):
                    st = i * self.chunk_len
                    en = min((i + 1) * self.chunk_len, len(small_vals))
                    chunks.append(small_vals[st:en])
                read_mapping.append((_map_range_st, len(chunks)))
                _map_range_st = len(chunks)

            chunk_compressed_result = self.compress_chunks(
                chunks, max_batch_size=batch_size, use_perfect=False, use_tqdm=use_tqdm,
            )

            final_compressed = []
            for _map_range in read_mapping:
                combined_bytes = bytearray()
                for b_chunk_bytes in chunk_compressed_result[_map_range[0]:_map_range[1]]:
                    combined_bytes.extend(struct.pack('<H', len(b_chunk_bytes)))
                    combined_bytes.extend(b_chunk_bytes)
                final_compressed.append(bytes(combined_bytes))
            return final_compressed

        else:
            assert list_of_compressed_reads is not None and list_of_lengths is not None

            compressed_bytes_list = []
            valid_lengths = []
            read_mapping = []
            _map_range_st = 0
            for r_idx, combined_bytes in enumerate(list_of_compressed_reads):
                rem_len = list_of_lengths[r_idx]
                pos = 0
                while pos < len(combined_bytes):
                    (block_comp_len,) = struct.unpack('<H', combined_bytes[pos:pos + 2])
                    pos += 2
                    chunk_bytes = combined_bytes[pos:pos + block_comp_len]
                    pos += block_comp_len
                    L = min(self.chunk_len, rem_len)
                    compressed_bytes_list.append(chunk_bytes)
                    valid_lengths.append(L)
                    rem_len -= L
                read_mapping.append((_map_range_st, len(compressed_bytes_list)))
                _map_range_st = len(compressed_bytes_list)

            flat_decoded_results = self.decompress_chunks(
                compressed_bytes_list=compressed_bytes_list,
                valid_lengths=valid_lengths,
                max_batch_size=batch_size,
                use_perfect=False,
                use_tqdm=use_tqdm,
            )

            final_decoded = []
            for _map_range in read_mapping:
                final_decoded.append(np.concatenate(flat_decoded_results[_map_range[0]: _map_range[1]]))
            return final_decoded

