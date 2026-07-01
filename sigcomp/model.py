import random
import torch
import torch.nn as nn
import torch.ao.quantization as quant


def _reduce_lsb_torch(signal: torch.Tensor, bit: int) -> torch.Tensor:
    step = 1 << bit
    rounding = ((signal >> (bit - 1)) & 1) * step
    upper = signal & (~(step - 1))
    return upper + rounding


def _lossy_encode_torch(signal: torch.Tensor, bit: int) -> torch.Tensor:
    return signal >> bit


class QAT_MLPDecoder(nn.Module):
    def __init__(self, vocab_size=257, window_size=32, decoder_dim=128, num_layers=8, enable_quant=False, input_dim=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.window_size = window_size

        # 1. Toggle Quant Stubs vs Identity
        if enable_quant:
            self.quant = quant.QuantStub()
            self.dequant = quant.DeQuantStub()
        else:
            self.quant = nn.Identity()
            self.dequant = nn.Identity()

        self.input_dim = input_dim if input_dim is not None else decoder_dim

        self.embedding_linear = torch.nn.Embedding(self.vocab_size, self.input_dim)

        layers = []
        in_dim = self.input_dim * window_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, decoder_dim * 2))
            layers.append(nn.ReLU())
            in_dim = decoder_dim * 2

        layers.append(nn.Linear(in_dim, vocab_size - 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, targets):
        B, L = targets.shape
        pad = targets[:, :1].expand(-1, self.window_size - 1)
        padded = torch.cat([pad, targets], dim=1)

        windows = padded.unfold(dimension=1, size=self.window_size, step=1)  # [B, L, window_size]
        embs = self.embedding_linear(windows)  # [B, L, window_size, decoder_dim]
        embs_flat = embs.view(B, L, -1)

        embs_flat = self.quant(embs_flat)
        logits = self.mlp(embs_flat)
        logits = self.dequant(logits)
        return logits

    def _ar_step(self, context_window):
        embs = self.embedding_linear(context_window)  # [B, window_size, decoder_dim]
        embs_flat = embs.view(context_window.size(0), -1)
        embs_flat = self.quant(embs_flat)
        logits = self.mlp(embs_flat)
        logits = self.dequant(logits)  # [B, vocab_size - 1]
        return logits.unsqueeze(1)

    def fuse_model(self):
        for i in range(0, len(self.mlp) - 1, 2):
            if isinstance(self.mlp[i], nn.Linear) and isinstance(self.mlp[i + 1], nn.ReLU):
                quant.fuse_modules(self.mlp, [str(i), str(i + 1)], inplace=True)


class QAT_MLP_Compression(nn.Module):
    def __init__(
            self,
            in_dim=128,
            dim=128,
            num_dec_layers=8,
            vocab_size=257,
            max_context_len=1024,
            window_size=32,
            enable_quant=False,
            lsb_eliminate=0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.SOS_token = vocab_size - 1
        self.max_context_len = max_context_len
        self.window_size = window_size
        self.lsb_eliminate = lsb_eliminate

        self.decoder = QAT_MLPDecoder(
            vocab_size=vocab_size,
            window_size=window_size,
            input_dim=in_dim,
            decoder_dim=dim,
            num_layers=num_dec_layers,
            enable_quant=enable_quant,
        )

    def get_logits(self, signals):
        B = signals.size(0)
        device = signals.device
        max_len = self.max_context_len

        batched_seq = torch.zeros([B, max_len], device=device, dtype=torch.long)
        batched_seq_lengths = torch.zeros([B], device=device, dtype=torch.long)

        for b in range(B):
            sig = signals[b]
            if self.lsb_eliminate > 0:
                sig = _lossy_encode_torch(
                    _reduce_lsb_torch(sig, self.lsb_eliminate),
                    self.lsb_eliminate,
                )
            delta = sig[1:] - sig[:-1]
            zigzag = torch.where(delta >= 0, 2 * delta, -2 * delta - 1)
            small_vals = zigzag[zigzag < 256]
            L = len(small_vals)
            if L == 0:
                batched_seq_lengths[b] = 0
                continue
            if L > max_len:
                start = random.randint(0, L - max_len)
                sampled = small_vals[start: start + max_len]
                actual_len = max_len
            else:
                sampled = small_vals
                actual_len = L
            batched_seq[b, :actual_len] = sampled
            batched_seq_lengths[b] = actual_len

        valid_mask = torch.arange(max_len, device=device).unsqueeze(0) < batched_seq_lengths.unsqueeze(1)
        batched_seq = batched_seq * valid_mask

        batched_qual = torch.zeros([B, max_len + 1], device=device, dtype=torch.long)
        batched_qual[:, 0] = self.SOS_token
        batched_qual[:, 1:] = batched_seq

        decoder_input = batched_qual[:, :-1]
        decoder_output = batched_qual[:, 1:].clone().contiguous()

        # Execute Teacher-Forcing forward
        logits = self.decoder.forward(targets=decoder_input)

        return {
            'logits': logits,
            'decoder_output': decoder_output,
            'batched_seq_lengths': batched_seq_lengths,
        }

    def forward(self, signals):
        device = signals.device
        ret = self.get_logits(signals)

        logits = ret['logits']
        decoder_output = ret['decoder_output']
        batched_seq_lengths = ret['batched_seq_lengths']

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, self.vocab_size - 1),
            decoder_output.view(-1),
            reduction='none',
        )

        loss_mask = torch.arange(self.max_context_len, device=device).unsqueeze(0) < batched_seq_lengths.unsqueeze(1)
        loss_mask = loss_mask.view(-1)

        loss = loss * loss_mask

        mask_sum = loss_mask.sum()
        if mask_sum > 0:
            loss = loss.sum() / mask_sum
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        return loss, {'loss': loss}


