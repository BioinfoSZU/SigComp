import torch
import triton
import triton.language as tl
import numpy as np

PRECISION = 24

def build_integer_exp_lut(output_scale, lut_precision_bits=38):
    from decimal import Decimal, getcontext
    saved_prec = getcontext().prec
    getcontext().prec = 50

    S = Decimal(1 << lut_precision_bits)
    scale_dec = Decimal(str(float(output_scale)))

    lut = torch.zeros(256, dtype=torch.int64)
    for i in range(256):
        exp_val = (Decimal(-i) * scale_dec).exp()
        lut[i] = max(int(exp_val * S), 0)

    getcontext().prec = saved_prec
    return lut


def int8_logits_to_cdf(int8_logits, exp_lut, precision=PRECISION):
    Q = 1 << precision
    B, V = int8_logits.shape
    device = int8_logits.device

    max_vals = int8_logits.max(dim=-1, keepdim=True).values
    shifted = max_vals.to(torch.int16) - int8_logits.to(torch.int16)

    unnorm = exp_lut[shifted.to(torch.long)]

    total = unnorm.sum(dim=-1, keepdim=True)

    freqs = (unnorm * Q) // total
    freqs = torch.clamp(freqs, min=1)

    total_freqs = freqs.sum(dim=-1, keepdim=True)
    diff = total_freqs - Q
    max_freq = freqs.max(dim=-1, keepdim=True).values
    idx_range = torch.arange(V, device=device).expand_as(freqs)
    max_idx = torch.where(
        freqs == max_freq, idx_range, torch.full_like(idx_range, V)
    ).min(dim=-1, keepdim=True).values
    freqs.scatter_add_(-1, max_idx, -diff)

    cdf = torch.zeros(B, V + 1, dtype=torch.int64, device=device)
    cdf[:, 1:] = torch.cumsum(freqs, dim=-1)
    return cdf


def quantize_probs_to_cdf(probs, precision=PRECISION, deterministic=False):
    M = 1 << precision
    V = probs.shape[-1]
    device = probs.device

    freqs = torch.clamp(torch.floor(probs * M).to(torch.int64), min=1)

    total = freqs.sum(dim=-1, keepdim=True)
    diff = total - M
    if deterministic:
        max_freq = freqs.max(dim=-1, keepdim=True).values
        idx_range = torch.arange(V, device=device).expand_as(freqs)
        max_idx = torch.where(
            freqs == max_freq, idx_range, torch.full_like(idx_range, V)
        ).min(dim=-1, keepdim=True).values
    else:
        max_idx = freqs.argmax(dim=-1, keepdim=True)
    freqs.scatter_add_(-1, max_idx, -diff)

    cdf = torch.zeros(*probs.shape[:-1], V + 1, dtype=torch.int64, device=device)
    cdf[..., 1:] = torch.cumsum(freqs, dim=-1)
    return cdf


@triton.jit
def _range_seal_kernel(
        state_lower_ptr,  # (B,) int64   — bitcast to uint64
        state_range_ptr,  # (B,) int64   — bitcast to uint64
        situation_ptr,  # (B,) int32
        num_inverted_ptr,  # (B,) int32
        first_inv_word_ptr,  # (B,) int32   — bitcast to uint32
        output_words_ptr,  # (B, MAX_WORDS) int32
        write_pos_ptr,  # (B,) int32   [in/out]
        B,
        MAX_WORDS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= B:
        return

    lower = tl.load(state_lower_ptr + pid).to(tl.uint64, bitcast=True)
    rnge = tl.load(state_range_ptr + pid).to(tl.uint64, bitcast=True)
    situation = tl.load(situation_ptr + pid)
    num_inverted = tl.load(num_inverted_ptr + pid)
    first_inv_word = tl.load(first_inv_word_ptr + pid).to(tl.uint32, bitcast=True)
    write_pos = tl.load(write_pos_ptr + pid)

    if rnge != tl.cast(0xFFFFFFFFFFFFFFFF, tl.uint64):
        point = lower + tl.cast(0xFFFFFFFF, tl.uint64)

        if situation == 1:
            if point < lower:
                emit_first = first_inv_word + tl.cast(1, tl.uint32)
                emit_rest = tl.cast(0x00000000, tl.uint32)
            else:
                emit_first = first_inv_word
                emit_rest = tl.cast(0xFFFFFFFF, tl.uint32)

            tl.store(output_words_ptr + pid * MAX_WORDS + write_pos,
                     emit_first.to(tl.int32, bitcast=True))
            write_pos += tl.cast(1, tl.int32)

            ni = num_inverted
            while ni > 1:
                tl.store(output_words_ptr + pid * MAX_WORDS + write_pos,
                         emit_rest.to(tl.int32, bitcast=True))
                write_pos += tl.cast(1, tl.int32)
                ni -= tl.cast(1, tl.int32)

        point_word = tl.cast(point >> 32, tl.uint32)
        tl.store(output_words_ptr + pid * MAX_WORDS + write_pos,
                 point_word.to(tl.int32, bitcast=True))
        write_pos += tl.cast(1, tl.int32)

        upper_word = tl.cast((lower + rnge) >> 32, tl.uint32)
        if upper_word == point_word:
            tl.store(output_words_ptr + pid * MAX_WORDS + write_pos,
                     tl.cast(0, tl.int32))
            write_pos += tl.cast(1, tl.int32)

    tl.store(write_pos_ptr + pid, write_pos)


@triton.jit
def _range_encode_step_int8_kernel(
        int8_logits_ptr,  # (B, V) int8
        exp_lut_ptr,  # (256,) int64
        symbols_ptr,  # (B,) int64
        state_lower_ptr,  # (B,) int64   [in/out]
        state_range_ptr,  # (B,) int64   [in/out]
        situation_ptr,  # (B,) int32   [in/out]
        num_inverted_ptr,  # (B,) int32   [in/out]
        first_inv_word_ptr,  # (B,) int32   [in/out]
        output_words_ptr,  # (B, MAX_WORDS) int32   [out]
        write_pos_ptr,  # (B,) int32   [in/out]
        valid_lens_ptr,  # (B,) int64
        t, B,
        V: tl.constexpr,  # 256
        MAX_WORDS: tl.constexpr,
        PRECISION: tl.constexpr,  # 24
):
    pid = tl.program_id(0)
    if pid >= B:
        return

    valid_len = tl.load(valid_lens_ptr + pid)
    if t >= valid_len:
        return

    v_offs = tl.arange(0, V)
    v_offs_i64 = v_offs.to(tl.int64)

    lut_vec = tl.load(exp_lut_ptr + v_offs_i64)  # (V,) int64

    logits = tl.load(int8_logits_ptr + pid * V + v_offs)
    logits_i16 = logits.to(tl.int16)
    max_val = tl.max(logits_i16)
    shifted = max_val - logits_i16  # [0, 255]

    unnorm = tl.load(exp_lut_ptr + shifted.to(tl.int64))

    M = tl.cast(1 << PRECISION, tl.int64)
    total = tl.sum(unnorm)
    freqs = tl.maximum((unnorm * M) // total, 1)

    total_freqs = tl.sum(freqs)
    diff = total_freqs - M
    max_freq = tl.max(freqs)
    first_max_idx = tl.min(tl.where(freqs == max_freq, v_offs_i64,
                                    tl.cast(V, tl.int64)))
    freqs = freqs + tl.where(v_offs_i64 == first_max_idx, -diff,
                             tl.cast(0, tl.int64))

    cdf_inclusive = tl.cumsum(freqs, axis=0)
    excdf = cdf_inclusive - freqs

    symbol = tl.load(symbols_ptr + pid)

    freq_sym = tl.sum(tl.where(v_offs_i64 == symbol, freqs,
                               tl.cast(0, tl.int64))).to(tl.uint64, bitcast=True)
    cum_sym = tl.sum(tl.where(v_offs_i64 == symbol, excdf,
                              tl.cast(0, tl.int64))).to(tl.uint64, bitcast=True)

    lower = tl.load(state_lower_ptr + pid).to(tl.uint64, bitcast=True)
    rnge = tl.load(state_range_ptr + pid).to(tl.uint64, bitcast=True)
    situation = tl.load(situation_ptr + pid)
    num_inverted = tl.load(num_inverted_ptr + pid)
    first_inv_word = tl.load(first_inv_word_ptr + pid).to(tl.uint32, bitcast=True)
    write_pos = tl.load(write_pos_ptr + pid)

    scale = rnge >> PRECISION
    new_range = scale * freq_sym
    new_lower = lower + scale * cum_sym

    if situation == 1:
        if new_lower + new_range > new_lower:
            if new_lower < lower:
                emit_first = first_inv_word + tl.cast(1, tl.uint32)
                emit_rest = tl.cast(0x00000000, tl.uint32)
            else:
                emit_first = first_inv_word
                emit_rest = tl.cast(0xFFFFFFFF, tl.uint32)
            tl.store(output_words_ptr + pid * MAX_WORDS + write_pos,
                     emit_first.to(tl.int32, bitcast=True))
            write_pos += tl.cast(1, tl.int32)
            ni = num_inverted
            while ni > 1:
                tl.store(output_words_ptr + pid * MAX_WORDS + write_pos,
                         emit_rest.to(tl.int32, bitcast=True))
                write_pos += tl.cast(1, tl.int32)
                ni -= tl.cast(1, tl.int32)
            situation = tl.cast(0, tl.int32)
            num_inverted = tl.cast(0, tl.int32)

    lower = new_lower
    rnge = new_range

    limit = tl.cast(1, tl.uint64) << 32
    if rnge < limit:
        rnge = rnge << 32
        lower_word = tl.cast(lower >> 32, tl.uint32)
        lower = lower << 32
        if situation == 1:
            num_inverted += tl.cast(1, tl.int32)
        elif lower + rnge > lower:
            tl.store(output_words_ptr + pid * MAX_WORDS + write_pos,
                     lower_word.to(tl.int32, bitcast=True))
            write_pos += tl.cast(1, tl.int32)
        else:
            situation = tl.cast(1, tl.int32)
            num_inverted = tl.cast(1, tl.int32)
            first_inv_word = lower_word

    tl.store(state_lower_ptr + pid, lower.to(tl.int64, bitcast=True))
    tl.store(state_range_ptr + pid, rnge.to(tl.int64, bitcast=True))
    tl.store(situation_ptr + pid, situation)
    tl.store(num_inverted_ptr + pid, num_inverted)
    tl.store(first_inv_word_ptr + pid, first_inv_word.to(tl.int32, bitcast=True))
    tl.store(write_pos_ptr + pid, write_pos)


class GpuRangeStepEncoder:
    def __init__(self, B, max_words, device='cuda'):
        self.B = B
        self.device = device
        self.max_words = max_words

        self.state_lower = torch.zeros(B, dtype=torch.int64, device=device)
        self.state_range = torch.full((B,), -1, dtype=torch.int64, device=device)
        self.situation = torch.zeros(B, dtype=torch.int32, device=device)
        self.num_inverted = torch.zeros(B, dtype=torch.int32, device=device)
        self.first_inv_word = torch.zeros(B, dtype=torch.int32, device=device)

        self.output_words = torch.zeros(B, max_words, dtype=torch.int32, device=device)
        self.write_pos = torch.zeros(B, dtype=torch.int32, device=device)

    def encode_step_int8(self, int8_logits, exp_lut, symbols, valid_lens_tensor, t):
        _range_encode_step_int8_kernel[(self.B,)](
            int8_logits, exp_lut, symbols,
            self.state_lower, self.state_range,
            self.situation, self.num_inverted, self.first_inv_word,
            self.output_words, self.write_pos,
            valid_lens_tensor,
            t, self.B,
            V=256, MAX_WORDS=self.max_words, PRECISION=PRECISION,
            num_warps=4,
        )

    def seal(self):
        _range_seal_kernel[(self.B,)](
            self.state_lower, self.state_range,
            self.situation, self.num_inverted, self.first_inv_word,
            self.output_words, self.write_pos,
            self.B,
            MAX_WORDS=self.max_words,
            num_warps=1,
        )

    def get_compressed(self):
        torch.cuda.synchronize()
        write_pos_cpu = self.write_pos.cpu().numpy()
        output_words_cpu = self.output_words.cpu().numpy()

        compressed_list = []
        for b in range(self.B):
            n = int(write_pos_cpu[b])
            words = output_words_cpu[b, :n].tobytes()
            compressed_list.append(words)

        return compressed_list


@triton.jit
def _range_decode_step_int8_kernel(
        int8_logits_ptr,  # (B, V) int8
        exp_lut_ptr,  # (256,) int64
        state_lower_ptr,  # (B,) int64   [in/out]
        state_range_ptr,  # (B,) int64   [in/out]
        point_ptr,  # (B,) int64   [in/out]
        comp_words_ptr,  # (total_words,) int32
        read_pos_ptr,  # (B,) int64   [in/out]
        end_pos_ptr,  # (B,) int64   [in]
        symbols_out_ptr,  # (B,) int64   [out]
        valid_lens_ptr,  # (B,) int64
        t, B,
        V: tl.constexpr,  # 256
        PRECISION: tl.constexpr,  # 24
):
    pid = tl.program_id(0)
    if pid >= B:
        return

    valid_len = tl.load(valid_lens_ptr + pid)
    if t >= valid_len:
        tl.store(symbols_out_ptr + pid, tl.cast(256, tl.int64))
        return

    v_offs = tl.arange(0, V)
    v_offs_i64 = v_offs.to(tl.int64)

    lut_vec = tl.load(exp_lut_ptr + v_offs_i64)  # (V,) int64

    logits = tl.load(int8_logits_ptr + pid * V + v_offs)
    logits_i16 = logits.to(tl.int16)
    max_val = tl.max(logits_i16)
    shifted = max_val - logits_i16

    unnorm = tl.load(exp_lut_ptr + shifted.to(tl.int64))  # gather (L1-hot)

    M = tl.cast(1 << PRECISION, tl.int64)
    total = tl.sum(unnorm)
    freqs = tl.maximum((unnorm * M) // total, 1)

    total_freqs = tl.sum(freqs)
    diff = total_freqs - M
    max_freq = tl.max(freqs)
    first_max_idx = tl.min(tl.where(freqs == max_freq, v_offs_i64,
                                    tl.cast(V, tl.int64)))
    freqs = freqs + tl.where(v_offs_i64 == first_max_idx, -diff,
                             tl.cast(0, tl.int64))

    cdf_inclusive = tl.cumsum(freqs, axis=0)
    excdf = cdf_inclusive - freqs

    lower = tl.load(state_lower_ptr + pid).to(tl.uint64, bitcast=True)
    rnge = tl.load(state_range_ptr + pid).to(tl.uint64, bitcast=True)
    point = tl.load(point_ptr + pid).to(tl.uint64, bitcast=True)

    scale = rnge >> PRECISION
    quantile = (point - lower) // scale
    quantile_i64 = quantile.to(tl.int64, bitcast=True)

    symbol = tl.sum(tl.where(excdf <= quantile_i64,
                             tl.cast(1, tl.int64),
                             tl.cast(0, tl.int64))) - 1
    symbol = tl.minimum(tl.maximum(symbol, 0), tl.cast(V - 1, tl.int64))

    freq_sym = tl.sum(tl.where(v_offs_i64 == symbol, freqs,
                               tl.cast(0, tl.int64))).to(tl.uint64, bitcast=True)
    cum_sym = tl.sum(tl.where(v_offs_i64 == symbol, excdf,
                              tl.cast(0, tl.int64))).to(tl.uint64, bitcast=True)

    lower = lower + scale * cum_sym
    rnge = scale * freq_sym

    limit = tl.cast(1, tl.uint64) << 32
    read_pos = tl.load(read_pos_ptr + pid)
    end_pos = tl.load(end_pos_ptr + pid)

    if rnge < limit:
        lower = lower << 32
        rnge = rnge << 32
        point = point << 32
        mask = read_pos < end_pos
        word_val = tl.load(comp_words_ptr + read_pos, mask=mask, other=0)
        word = word_val.to(tl.uint32, bitcast=True).to(tl.uint64)
        point = point | word
        read_pos += 1

    tl.store(state_lower_ptr + pid, lower.to(tl.int64, bitcast=True))
    tl.store(state_range_ptr + pid, rnge.to(tl.int64, bitcast=True))
    tl.store(point_ptr + pid, point.to(tl.int64, bitcast=True))
    tl.store(read_pos_ptr + pid, read_pos)
    tl.store(symbols_out_ptr + pid, symbol)


class GpuRangeDecoder:
    def __init__(self, compressed_bytes_list, device='cuda'):
        self.B = len(compressed_bytes_list)
        self.device = device

        points = []
        read_poses = []
        end_poses = []
        all_words = []
        current_offset = 0

        for data in compressed_bytes_list:
            words = np.frombuffer(data, dtype=np.uint32).copy()
            point = 0
            num_read = 0
            for i in range(min(len(words), 2)):
                point = (point << 32) | int(words[i])
                num_read += 1

            if num_read < 2:
                if num_read != 0:
                    point = point << (64 - num_read * 32)

            points.append(point)

            all_words.append(words)
            words_to_skip = min(len(words), 2)
            read_poses.append(current_offset + words_to_skip)
            current_offset += len(words)
            end_poses.append(current_offset)

        if all_words:
            flat_words = np.concatenate(all_words)
        else:
            flat_words = np.array([], dtype=np.uint32)

        self.comp_words = torch.from_numpy(flat_words.view(np.int32)).to(device)
        self.read_pos = torch.tensor(read_poses, dtype=torch.int64, device=device)
        self.end_pos = torch.tensor(end_poses, dtype=torch.int64, device=device)
        self.point = torch.from_numpy(np.array(points, dtype=np.uint64).view(np.int64)).to(device)
        self.state_lower = torch.zeros(self.B, dtype=torch.int64, device=device)
        self.state_range = torch.full((self.B,), -1, dtype=torch.int64, device=device)
        self.symbols_out = torch.zeros(self.B, dtype=torch.int64, device=device)

    def decode_step_int8(self, int8_logits, exp_lut, valid_lens_tensor, t):
        _range_decode_step_int8_kernel[(self.B,)](
            int8_logits, exp_lut,
            self.state_lower, self.state_range, self.point,
            self.comp_words, self.read_pos, self.end_pos,
            self.symbols_out, valid_lens_tensor,
            t, self.B,
            V=256, PRECISION=PRECISION,
            num_warps=4,
        )
        return self.symbols_out
