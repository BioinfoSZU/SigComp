from functools import lru_cache
import math
from pathlib import Path
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


_BACKENDS = {"auto", "int_mm", "numba", "numpy"}
_EPILOGUES = {"auto", "numba", "torch"}
_PARALLEL_POLICIES = {"auto", "serial", "parallel"}


def float_to_multiplier_shift(multiplier):
    if multiplier == 0:
        return 0, 0
    significand, exponent = math.frexp(multiplier)
    return int(round(significand * (2 ** 31))), 31 - exponent


@lru_cache(maxsize=1)
def _int_mm_conforms():
    if not hasattr(torch, "_int_mm"):
        return False
    values = np.array([-128, 127, -127, 126, 0, 1, -1, 64], dtype=np.int8)
    try:
        for width in (64, 8192):
            x = np.resize(values, (33, width)).copy()
            x[0, :] = -128
            x[1, :] = 127
            w = np.resize(values[::-1], (64, width)).copy()
            w[0, :] = -128
            w[1, :] = 127
            w[1, 0] = 126
            expected = x.astype(np.int64) @ w.astype(np.int64).T
            weights = torch.from_numpy(w).t()
            for rows in (1, 17, 33):
                actual = torch._int_mm(torch.from_numpy(x[:rows]), weights)
                if actual.dtype != torch.int32 or not np.array_equal(
                    actual.numpy(), expected[:rows]
                ):
                    return False
    except (RuntimeError, NotImplementedError):
        return False
    return True


@lru_cache(maxsize=1)
def _int_mm_out_conforms():
    if not _int_mm_conforms():
        raise RuntimeError("CPU INT8 GEMM failed the integer conformance probe")
    values = np.array([-128, 127, -127, 126, 0, 1, -1, 64], dtype=np.int8)
    for width in (64, 8192):
        x = np.resize(values, (33, width)).copy()
        x[0, :] = -128
        x[1, :] = 127
        w = np.resize(values[::-1], (64, width)).copy()
        w[0, :] = -128
        w[1, :] = 127
        w[1, 0] = 126
        expected = x.astype(np.int64) @ w.astype(np.int64).T
        weights = torch.from_numpy(w).t()
        storage = torch.empty((33, 64), dtype=torch.int32)
        for rows in (1, 17, 33, 17):
            output = storage[:rows]
            output_pointer = output.data_ptr()
            try:
                actual = torch._int_mm(torch.from_numpy(x[:rows]), weights, out=output)
            except (TypeError, RuntimeError, NotImplementedError):
                return False
            if (not isinstance(actual, torch.Tensor)
                    or actual.dtype != torch.int32 or actual.shape != (rows, 64)
                    or actual.data_ptr() != output_pointer or output.data_ptr() != output_pointer
                    or not np.array_equal(output.numpy(), expected[:rows])):
                raise RuntimeError(
                    "CPU INT8 GEMM out= failed the integer conformance probe; "
                    "disable the CPU inference workspace and inspect this PyTorch build"
                )
    return True


def _has_vnni_gemm():
    if not (torch.backends.mkldnn.is_available() and torch.backends.mkldnn.enabled):
        return False
    capabilities = getattr(getattr(torch, "cpu", None), "get_capabilities", None)
    if capabilities is not None:
        try:
            if capabilities().get("avx512_vnni", False):
                return True
        except (RuntimeError, AttributeError, NotImplementedError):
            pass
    check = getattr(getattr(torch, "cpu", None), "_is_vnni_supported", None)
    try:
        return check is not None and bool(check())
    except (RuntimeError, AttributeError, NotImplementedError):
        return False


@lru_cache(maxsize=1)
def _auto_backend():
    if _has_vnni_gemm() and _int_mm_conforms():
        return "int_mm"
    try:
        from . import cpu_int_kernels  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Efficient CPU integer inference requires numba on this CPU. "
            "Install numba, or explicitly select the slower numpy backend."
        ) from exc
    return "numba"


@lru_cache(maxsize=1)
def _auto_epilogue():
    try:
        from . import cpu_int_kernels  # noqa: F401
    except ImportError:
        return "torch"
    return "numba"


class CPUBitExactInt8Linear(nn.Module):

    def __init__(self, weight_q, bias_q, M0, shift, has_relu=False, backend="auto",
                 *, epilogue="auto", epilogue_parallel="auto", epilogue_min_work=1 << 18,
                 profiler=None, profile_name="inference.layer"):
        super().__init__()
        if backend not in _BACKENDS:
            raise ValueError(f"Unknown CPU integer backend: {backend!r}")
        if epilogue not in _EPILOGUES:
            raise ValueError(f"Unknown CPU integer epilogue: {epilogue!r}")
        if epilogue_parallel not in _PARALLEL_POLICIES:
            raise ValueError(f"Unknown CPU epilogue parallel policy: {epilogue_parallel!r}")
        if (isinstance(epilogue_min_work, bool) or int(epilogue_min_work) != epilogue_min_work
                or epilogue_min_work < 1):
            raise ValueError("epilogue_min_work must be a positive integer")
        if weight_q.dtype != torch.int8 or weight_q.ndim != 2:
            raise ValueError("weight_q must be a two-dimensional int8 tensor")
        if bias_q.dtype != torch.int32 or bias_q.shape != (weight_q.shape[0],):
            raise ValueError("bias_q must be an int32 vector of output width")
        if min(weight_q.shape) < 1:
            raise ValueError("Linear layer dimensions must be positive")
        self.M0 = int(M0)
        self.shift = int(shift)
        if not 0 <= self.M0 <= (1 << 31):
            raise ValueError("The quantized multiplier must be in [0, 2**31]")
        if not 0 <= self.shift <= 63:
            raise ValueError("The quantized right shift must be in [0, 63]")
        self.has_relu = bool(has_relu)
        self.register_buffer("weight_q", weight_q.detach().cpu().contiguous().clone())
        self.register_buffer("bias_q", bias_q.detach().cpu().contiguous().clone())
        self.backend = _auto_backend() if backend == "auto" else backend
        self.epilogue = _auto_epilogue() if epilogue == "auto" else epilogue
        self.epilogue_parallel = epilogue_parallel
        self.epilogue_min_work = int(epilogue_min_work)
        if self.epilogue == "numba":
            from . import cpu_int_kernels  # noqa: F401
        self.profiler = profiler
        self.profile_name = str(profile_name)
        self._allow_fallback = backend == "auto"
        if self.backend == "int_mm" and not _int_mm_conforms():
            raise RuntimeError("This PyTorch CPU INT8 GEMM failed the integer conformance probe; "
                               "use --cpu-backend numba and inspect python -m sigcomp.cpu_runtime "
                               "--probe-int-mm. Do not bypass the check for compressed-file compatibility.")
        self._numpy_weight = None

    def _apply_epilogue(self, accumulator):
        if self.epilogue == "numba":
            from .cpu_int_kernels import choose_epilogue_kernel

            kernel = choose_epilogue_kernel(
                *accumulator.shape, self.epilogue_parallel, self.epilogue_min_work
            )
            output = kernel(
                accumulator.numpy(), self.bias_q.numpy(), np.int64(self.M0),
                np.int64(self.shift), self.has_relu,
            )
            return torch.from_numpy(output)
        accumulator.add_(self.bias_q)  # INT32 overflow is intentional.
        scaled = accumulator.to(torch.int64).mul_(self.M0)
        scaled.bitwise_right_shift_(self.shift)
        return scaled.clamp_(0 if self.has_relu else -128, 127).to(torch.int8)

    def _numba_forward(self, x_flat):
        from .cpu_int_kernels import linear_parallel, linear_serial

        operations = x_flat.shape[0] * self.weight_q.numel()
        kernel = linear_parallel if operations >= (1 << 20) else linear_serial
        result = kernel(
            x_flat.numpy(), self.weight_q.numpy(), self.bias_q.numpy(),
            np.int64(self.M0), np.int64(self.shift), self.has_relu,
        )
        return torch.from_numpy(result)

    def _numpy_forward(self, x_flat):
        if self._numpy_weight is None:
            self._numpy_weight = self.weight_q.numpy().astype(np.int32).T
        accumulator = x_flat.numpy().astype(np.int32) @ self._numpy_weight
        np.add(accumulator, self.bias_q.numpy(), out=accumulator)
        scaled = (accumulator.astype(np.int64) * self.M0) >> self.shift
        return torch.from_numpy(np.clip(scaled, 0 if self.has_relu else -128, 127).astype(np.int8))

    def forward(self, x_q):
        if x_q.device.type != "cpu" or x_q.dtype != torch.int8:
            raise ValueError("CPU integer inference requires CPU int8 activations")
        if x_q.ndim < 1 or x_q.shape[-1] != self.weight_q.shape[1]:
            raise ValueError("Activation width does not match the linear layer")
        original_shape = x_q.shape
        x_flat = x_q.reshape(-1, original_shape[-1]).contiguous()
        if x_flat.stride(0) != x_flat.shape[1]:
            x_flat = x_flat.clone(memory_format=torch.contiguous_format)
        if x_flat.shape[0] == 0:
            return torch.empty((*original_shape[:-1], self.weight_q.shape[0]), dtype=torch.int8)
        if self.backend == "numba":
            if self.profiler is None:
                output = self._numba_forward(x_flat)
            else:
                with self.profiler.measure(self.profile_name + ".fused_linear"):
                    output = self._numba_forward(x_flat)
        elif self.backend == "numpy":
            if self.profiler is None:
                output = self._numpy_forward(x_flat)
            else:
                with self.profiler.measure(self.profile_name + ".reference_linear"):
                    output = self._numpy_forward(x_flat)
        else:
            try:
                if self.profiler is None:
                    accumulator = torch._int_mm(x_flat, self.weight_q.t())
                else:
                    with self.profiler.measure(self.profile_name + ".gemm"):
                        accumulator = torch._int_mm(x_flat, self.weight_q.t())
            except (RuntimeError, NotImplementedError) as exc:
                if not self._allow_fallback:
                    raise
                warnings.warn(
                    f"CPU INT8 GEMM unavailable for this layer; using Numba: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.backend = "numba"
                if self.profiler is None:
                    output = self._numba_forward(x_flat)
                else:
                    with self.profiler.measure(self.profile_name + ".fused_linear"):
                        output = self._numba_forward(x_flat)
            else:
                if self.profiler is None:
                    output = self._apply_epilogue(accumulator)
                else:
                    with self.profiler.measure(self.profile_name + ".epilogue"):
                        output = self._apply_epilogue(accumulator)
        return output.reshape(*original_shape[:-1], self.weight_q.shape[0])


class CPUBitExactInt8Embedding(nn.Module):
    def __init__(self, weight_q_int8):
        super().__init__()
        if weight_q_int8.dtype != torch.int8 or weight_q_int8.ndim != 2:
            raise ValueError("Embedding weights must be a two-dimensional int8 tensor")
        self.register_buffer("weight_q", weight_q_int8.detach().cpu().contiguous().clone())

    def forward(self, input_ids):
        return F.embedding(input_ids, self.weight_q)


class CPUBitExactMLPDecoder(nn.Module):
    def __init__(self, vocab_size, window_size, *, profiler=None):
        super().__init__()
        if vocab_size < 2 or window_size < 1:
            raise ValueError("Vocabulary and context window must be positive")
        self.vocab_size = int(vocab_size)
        self.window_size = int(window_size)
        self.profiler = profiler
        self.embedding = None
        self.mlp = nn.ModuleList()
        self.register_buffer("output_scale", torch.tensor(1.0, dtype=torch.float32))

    def set_profiler(self, profiler):
        self.profiler = profiler
        for index, layer in enumerate(self.mlp):
            layer.profiler = profiler
            layer.profile_name = f"inference.layer_{index}"
        return self

    def create_session(self, max_rows, enabled=True):
        if not enabled:
            return None
        if (isinstance(max_rows, bool) or int(max_rows) != max_rows or max_rows < 1):
            raise ValueError("max_rows must be a positive integer")
        if (not isinstance(self.embedding, CPUBitExactInt8Embedding) or not self.mlp
                or any(not isinstance(layer, CPUBitExactInt8Linear)
                       or layer.backend != "int_mm" or layer.epilogue != "numba"
                       for layer in self.mlp)):
            return None
        from .cpu_inference_workspace import CPUInferenceSession

        return CPUInferenceSession(self, int(max_rows), gemm_out_available=_int_mm_out_conforms())

    def forward(self, targets):
        if targets.ndim != 2 or targets.device.type != "cpu":
            raise ValueError("targets must be a CPU tensor shaped [batch, sequence]")
        batch, length = targets.shape
        if batch == 0 or length == 0:
            return torch.empty((batch, length, self.vocab_size - 1), dtype=torch.int8)
        padding = targets[:, :1].expand(-1, self.window_size - 1)
        windows = torch.cat((padding, targets), dim=1).unfold(1, self.window_size, 1)
        if self.profiler is None:
            x = self.embedding(windows).reshape(batch, length, -1)
        else:
            with self.profiler.measure("inference.embedding"):
                x = self.embedding(windows).reshape(batch, length, -1)
        if self.profiler is None:
            for layer in self.mlp:
                x = layer(x)
        else:
            for layer in self.mlp:
                with self.profiler.measure(layer.profile_name + ".total"):
                    x = layer(x)
        return x

    def _ar_step(self, context_window):
        if context_window.ndim != 2 or context_window.shape[1] != self.window_size:
            raise ValueError("context_window must be shaped [batch, window_size]")
        if context_window.device.type != "cpu":
            raise ValueError("CPU decoder requires a CPU context window")
        if context_window.shape[0] == 0:
            return torch.empty((0, 1, self.vocab_size - 1), dtype=torch.int8)
        if self.profiler is None:
            x = self.embedding(context_window).reshape(context_window.shape[0], -1)
        else:
            with self.profiler.measure("inference.embedding"):
                x = self.embedding(context_window).reshape(context_window.shape[0], -1)
        if self.profiler is None:
            for layer in self.mlp:
                x = layer(x)
        else:
            for layer in self.mlp:
                with self.profiler.measure(layer.profile_name + ".total"):
                    x = layer(x)
        return x.unsqueeze(1)

    def dequantize_logits(self, int8_logits):
        return int8_logits.float() * self.output_scale


def _extract_linear(weight, bias, input_scale, output_scale, has_relu, backend,
                    *, epilogue="auto", epilogue_parallel="auto", epilogue_min_work=1 << 18,
                    profiler=None, profile_name="inference.layer"):
    if weight.dtype != torch.qint8 or weight.ndim != 2:
        raise ValueError("Expected per-tensor quantized int8 linear weights")
    if weight.qscheme() not in (torch.per_tensor_affine, torch.per_tensor_symmetric):
        raise ValueError("Per-channel weights are unsupported by the existing GPU format")
    weight_q = weight.int_repr()
    bias_scale = input_scale * weight.q_scale()
    if bias is None:
        bias_q = torch.zeros(weight_q.shape[0], dtype=torch.int32)
    else:
        bias_q = torch.round(bias.detach().cpu() / bias_scale).to(torch.int32)
    multiplier, shift = float_to_multiplier_shift(bias_scale / output_scale)
    return CPUBitExactInt8Linear(
        weight_q, bias_q, multiplier, shift, has_relu, backend,
        epilogue=epilogue, epilogue_parallel=epilogue_parallel,
        epilogue_min_work=epilogue_min_work, profiler=profiler, profile_name=profile_name,
    )


def convert_ptq_to_cpu_pipeline(cpu_ptq_model, vocab_size, window_size, backend="auto",
                                *, epilogue="auto", epilogue_parallel="auto",
                                epilogue_min_work=1 << 18, profiler=None):
    decoder = CPUBitExactMLPDecoder(vocab_size, window_size, profiler=profiler)
    input_scale = cpu_ptq_model.decoder.quant.scale.item()
    embedding = cpu_ptq_model.decoder.embedding_linear.weight.detach().cpu()
    decoder.embedding = CPUBitExactInt8Embedding(
        torch.round(embedding / input_scale).clamp(-128, 127).to(torch.int8)
    )
    for layer in cpu_ptq_model.decoder.mlp:
        if isinstance(layer, nn.Identity):
            continue
        has_relu = isinstance(layer, torch.ao.nn.intrinsic.quantized.LinearReLU)
        decoder.mlp.append(_extract_linear(
            layer.weight(), layer.bias(), input_scale, layer.scale, has_relu, backend,
            epilogue=epilogue, epilogue_parallel=epilogue_parallel,
            epilogue_min_work=epilogue_min_work, profiler=profiler,
            profile_name=f"inference.layer_{len(decoder.mlp)}",
        ))
        input_scale = layer.scale
    decoder.output_scale.fill_(input_scale)
    return decoder.eval()


def load_cpu_decoder(model_path, dim=256, num_dec_layers=8, window_size=64, backend="auto",
                     *, epilogue="auto", epilogue_parallel="auto", epilogue_min_work=1 << 18,
                     profiler=None):
    dim = 256 if dim is None else int(dim)
    num_dec_layers = 8 if num_dec_layers is None else int(num_dec_layers)
    window_size = 64 if window_size is None else int(window_size)
    if dim < 2 or num_dec_layers < 0 or window_size < 1:
        raise ValueError("Invalid MLP architecture")
    state = torch.load(Path(model_path), map_location="cpu", weights_only=True)
    decoder = CPUBitExactMLPDecoder(257, window_size, profiler=profiler)
    try:
        embedding = state["decoder.embedding_linear.weight"]
        input_scale = float(state["decoder.quant.scale"].item())
        if embedding.shape != (257, dim // 2):
            raise ValueError("Checkpoint embedding shape does not match --dim")
        if not math.isfinite(input_scale) or input_scale <= 0:
            raise ValueError("Invalid embedding quantization scale")
        decoder.embedding = CPUBitExactInt8Embedding(
            torch.round(embedding / input_scale).clamp(-128, 127).to(torch.int8)
        )
        expected_input = (dim // 2) * window_size
        for index in range(num_dec_layers + 1):
            prefix = f"decoder.mlp.{index * 2}"
            weight, bias = state[f"{prefix}._packed_params._packed_params"]
            output_scale = float(state[f"{prefix}.scale"].item())
            output_width = dim * 2 if index < num_dec_layers else 256
            if weight.shape != (output_width, expected_input):
                raise ValueError(
                    f"Checkpoint {prefix} shape does not match --dim, "
                    "--num-dec-layers or --window-size"
                )
            if not math.isfinite(output_scale) or output_scale <= 0:
                raise ValueError(f"Invalid quantization scale for {prefix}")
            decoder.mlp.append(_extract_linear(
                weight, bias, input_scale, output_scale, index < num_dec_layers, backend,
                epilogue=epilogue, epilogue_parallel=epilogue_parallel,
                epilogue_min_work=epilogue_min_work, profiler=profiler,
                profile_name=f"inference.layer_{index}",
            ))
            input_scale = output_scale
            expected_input = output_width
        actual_layers = sum(
            key.startswith("decoder.mlp.") and key.endswith("._packed_params._packed_params")
            for key in state
        )
        if actual_layers != num_dec_layers + 1:
            raise ValueError("Checkpoint layer count does not match --num-dec-layers")
    except KeyError as exc:
        raise ValueError(f"Incomplete or incompatible quantized MLP checkpoint: {exc}") from exc
    decoder.output_scale.fill_(input_scale)
    return decoder.eval()
