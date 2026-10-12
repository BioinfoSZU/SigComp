import numpy as np
import torch

from .cpu_int_kernels import choose_embedding_into_kernel, choose_epilogue_into_kernel


class _RetryWithoutOut(Exception):
    """Leave any profiling scopes before retrying the complete inference call."""


class _RetryWithReference(Exception):
    """Preserve the model's established automatic backend fallback."""


class CPUInferenceSession:

    def __init__(self, decoder, max_rows, *, gemm_out_available):
        self.decoder = decoder
        self.max_rows = int(max_rows)
        self.window_size = int(decoder.window_size)
        self.gemm_out_available = bool(gemm_out_available)
        self._gemm_out_probe_passed = bool(gemm_out_available)
        self._gemm_out_fallback_reason = None
        self._reference_fallback_reason = None
        self._reference_mode = False
        self.embedding_weights = decoder.embedding.weight_q.numpy()
        input_width = self.window_size * self.embedding_weights.shape[1]
        self._layers = []
        for layer in decoder.mlp:
            if (layer.weight_q.device.type != "cpu" or layer.bias_q.device.type != "cpu"
                    or layer.weight_q.dtype != torch.int8 or layer.bias_q.dtype != torch.int32
                    or layer.weight_q.ndim != 2 or layer.weight_q.shape[1] != input_width
                    or layer.bias_q.shape != (layer.weight_q.shape[0],)):
                raise ValueError("CPU inference workspace requires a consistent CPU integer MLP")
            self._layers.append((
                layer.weight_q.t(), layer.bias_q.numpy(), np.int64(layer.M0),
                np.int64(layer.shift), layer.has_relu, layer.epilogue_parallel,
                layer.epilogue_min_work, layer.profile_name, layer._allow_fallback,
            ))
            input_width = int(layer.weight_q.shape[0])
        self.output_width = input_width
        if self.output_width != decoder.vocab_size - 1:
            raise ValueError("CPU inference workspace output width does not match vocabulary")
        embedding_width = self.window_size * self.embedding_weights.shape[1]
        widths = [int(layer.weight_q.shape[0]) for layer in decoder.mlp]
        self._embedding = torch.empty((self.max_rows, embedding_width), dtype=torch.int8)
        self._output = torch.empty((self.max_rows, self.output_width), dtype=torch.int8)
        self._errors = np.empty(self.max_rows, dtype=np.uint8)
        hidden_width = max(widths[:-1], default=0)
        self._activations = [
            torch.empty(self.max_rows * hidden_width, dtype=torch.int8)
            for _ in range(min(2, len(widths) - 1))
        ]
        self._accumulator = (
            torch.empty(self.max_rows * max(widths), dtype=torch.int32)
            if self.gemm_out_available else None
        )
        self._full_layers = []
        for index, (constants, width) in enumerate(zip(self._layers, widths)):
            if index == len(widths) - 1:
                output = self._output
            else:
                output = self._activations[index % 2][:self.max_rows * width].view(self.max_rows, width)
            accumulator = (
                self._accumulator[:self.max_rows * width].view(self.max_rows, width)
                if self._accumulator is not None else None
            )
            self._full_layers.append((constants, accumulator, output))
        self._cached_rows = None
        self._cached_views = None
        self._cached_embedding = None
        self._cached_embedding_numpy = None
        self._cached_errors = None
        self._cached_logits = None
        self._embedding_kernels = None

    def describe(self):
        buffers = {
            "embedding": self._embedding.numel() * self._embedding.element_size(),
            "activation_ping_pong": sum(x.numel() * x.element_size() for x in self._activations),
            "output": self._output.numel() * self._output.element_size(),
            "accumulator": (self._accumulator.numel() * self._accumulator.element_size()
                            if self._accumulator is not None else 0),
            "bounds_status": self._errors.nbytes,
        }
        return {
            "enabled": True,
            "mode": "reference_fallback" if self._reference_mode else "invocation_owned_reusable_buffers",
            "backend": "reference" if self._reference_mode else "int_mm",
            "epilogue": "numba",
            "max_rows": self.max_rows,
            "gemm_out_available": self.gemm_out_available,
            "gemm_out_probe_passed": self._gemm_out_probe_passed,
            "gemm_out_fallback_reason": self._gemm_out_fallback_reason,
            "reference_fallback_reason": self._reference_fallback_reason,
            "buffer_bytes": sum(buffers.values()),
            "buffers": buffers,
            "output_lifetime": "Borrowed until the next call on this session; copy before retaining.",
            "buffer_note": "Persistent session storage only; excludes model, contexts, entropy coding, JIT and library scratch.",
        }

    def _prepare_views(self, rows):
        if rows == self._cached_rows:
            return
        views = []
        for constants, full_accumulator, full_output in self._full_layers:
            accumulator = full_accumulator[:rows] if full_accumulator is not None else None
            output = full_output[:rows]
            kernel = choose_epilogue_into_kernel(rows, output.shape[1], constants[5], constants[6])
            views.append((
                constants, accumulator, accumulator.numpy() if accumulator is not None else None,
                output, output.numpy(), kernel,
            ))
        self._cached_embedding = self._embedding[:rows]
        self._cached_embedding_numpy = self._cached_embedding.numpy()
        self._cached_errors = self._errors[:rows]
        self._cached_logits = self._output[:rows].unsqueeze(1)
        self._embedding_kernels = (
            choose_embedding_into_kernel(rows, self._embedding.shape[1]),
            choose_embedding_into_kernel(rows, self._embedding.shape[1], windows=True),
        )
        self._cached_views = views
        self._cached_rows = rows

    def _context_array(self, context, dimensions):
        if isinstance(context, torch.Tensor):
            if context.device.type != "cpu" or context.dtype not in (torch.int32, torch.int64):
                raise ValueError("CPU inference context must contain CPU int32 or int64 indices")
            context = context.numpy()
        if (not isinstance(context, np.ndarray) or context.ndim != dimensions
                or context.dtype not in (np.dtype(np.int32), np.dtype(np.int64))
                or context.shape[-1] != self.window_size):
            shape = "[rows, window_size]" if dimensions == 2 else "[streams, time, window_size]"
            raise ValueError(f"CPU inference context must be an int32/int64 array shaped {shape}")
        rows = int(context.shape[0]) if dimensions == 2 else int(context.shape[0]) * int(context.shape[1])
        if rows > self.max_rows:
            raise ValueError(f"CPU inference workspace capacity {self.max_rows} is smaller than {rows} rows")
        return context, rows

    def _gather(self, context, windows):
        self._embedding_kernels[int(windows)](
            context, self.embedding_weights, self._cached_embedding_numpy, self._cached_errors
        )
        if np.any(self._cached_errors):
            raise ValueError("CPU inference context token is outside the embedding vocabulary")

    def _run_accelerated(self, context, dimensions):
        profiler = self.decoder.profiler
        if profiler is None:
            self._gather(context, dimensions == 3)
            x = self._cached_embedding
            if self.gemm_out_available:
                for constants, accumulator, accumulator_numpy, output, output_numpy, kernel in self._cached_views:
                    try:
                        torch._int_mm(x, constants[0], out=accumulator)
                    except (TypeError, RuntimeError, NotImplementedError) as exc:
                        raise _RetryWithoutOut(str(exc)) from exc
                    kernel(accumulator_numpy, constants[1], constants[2], constants[3], constants[4], output_numpy)
                    x = output
            else:
                for constants, _, _, output, output_numpy, kernel in self._cached_views:
                    try:
                        accumulator = torch._int_mm(x, constants[0])
                    except (RuntimeError, NotImplementedError) as exc:
                        if not constants[8]:
                            raise
                        raise _RetryWithReference(str(exc)) from exc
                    kernel(accumulator.numpy(), constants[1], constants[2], constants[3], constants[4], output_numpy)
                    x = output
                    del accumulator
        else:
            with profiler.measure("inference.embedding"):
                self._gather(context, dimensions == 3)
            x = self._cached_embedding
            for constants, accumulator, accumulator_numpy, output, output_numpy, kernel in self._cached_views:
                name = constants[7]
                with profiler.measure(name + ".total"):
                    with profiler.measure(name + ".gemm"):
                        if self.gemm_out_available:
                            try:
                                torch._int_mm(x, constants[0], out=accumulator)
                            except (TypeError, RuntimeError, NotImplementedError) as exc:
                                raise _RetryWithoutOut(str(exc)) from exc
                        else:
                            try:
                                accumulator = torch._int_mm(x, constants[0])
                            except (RuntimeError, NotImplementedError) as exc:
                                if not constants[8]:
                                    raise
                                raise _RetryWithReference(str(exc)) from exc
                            accumulator_numpy = accumulator.numpy()
                    with profiler.measure(name + ".epilogue"):
                        kernel(accumulator_numpy, constants[1], constants[2], constants[3], constants[4], output_numpy)
                    x = output
                    if not self.gemm_out_available:
                        del accumulator_numpy, accumulator
        return self._cached_logits

    def _run_reference(self, context):
        if np.any(context < 0) or np.any(context >= self.embedding_weights.shape[0]):
            raise ValueError("CPU inference context token is outside the embedding vocabulary")
        profiler = self.decoder.profiler
        if profiler is None:
            indices = torch.from_numpy(np.array(context, dtype=np.int64, order="C", copy=True)).reshape(
                self._cached_rows, self.window_size
            )
            logits = self.decoder._ar_step(indices)
            self._cached_logits.copy_(logits)
        else:
            with profiler.measure("inference.reference_context_copy"):
                indices = torch.from_numpy(np.array(context, dtype=np.int64, order="C", copy=True)).reshape(
                    self._cached_rows, self.window_size
                )
            logits = self.decoder._ar_step(indices)
            with profiler.measure("inference.reference_output_copy"):
                self._cached_logits.copy_(logits)
        return self._cached_logits

    def _step(self, context, dimensions):
        context, rows = self._context_array(context, dimensions)
        self._prepare_views(rows)
        if rows == 0:
            return self._cached_logits
        if self._reference_mode:
            return self._run_reference(context)
        while True:
            try:
                return self._run_accelerated(context, dimensions)
            except _RetryWithoutOut as exc:
                self.gemm_out_available = False
                self._gemm_out_fallback_reason = str(exc)
            except _RetryWithReference as exc:
                self._reference_mode = True
                self.gemm_out_available = False
                self._reference_fallback_reason = str(exc)
                return self._run_reference(context)

    def step(self, context_window):
        return self._step(context_window, 2)

    def step_windows(self, context_windows):
        return self._step(context_windows, 3)
