import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop1(self.self_attn(self.norm1(x), mask))
        x = x + self.drop2(self.feed_forward(self.norm2(x)))
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, d_model: int = 128,
                 n_layers: int = 6, n_heads: int = 8, d_ff: int = 2048,
                 max_seq_len: int = 128, dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model**-0.5)
        if self.embedding.padding_idx is not None:
            self.embedding.weight.data[self.embedding.padding_idx].zero_()

        depth_scale = (2 * self.n_layers) ** -0.5

        for layer in self.encoder_layers:
            for module in [layer.self_attn.W_q, layer.self_attn.W_k, layer.self_attn.W_v]:
                nn.init.xavier_uniform_(module.weight)
            nn.init.xavier_uniform_(layer.self_attn.W_o.weight, gain=depth_scale)
            nn.init.zeros_(layer.self_attn.W_o.bias)

            nn.init.xavier_uniform_(layer.feed_forward.linear1.weight)
            nn.init.zeros_(layer.feed_forward.linear1.bias)
            nn.init.xavier_uniform_(layer.feed_forward.linear2.weight, gain=depth_scale)
            nn.init.zeros_(layer.feed_forward.linear2.bias)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def create_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (input_ids != self.pad_idx).float()

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = self.create_padding_mask(input_ids)
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, mask)

        mask_exp = mask.unsqueeze(-1).expand(x.size())
        sum_mask = mask_exp.sum(dim=1).clamp(min=1e-9)
        pooled = torch.sum(x * mask_exp, dim=1) / sum_mask
        return pooled

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pooled = self.encode(input_ids)
        return self.classifier(self.dropout(pooled))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model_configs():
    return [
        {"name": "tiny-1",   "d_model": 32,  "n_layers": 1, "n_heads": 2, "d_ff": 128},
        {"name": "small-1",  "d_model": 64,  "n_layers": 1, "n_heads": 4, "d_ff": 256},
        {"name": "small-4",  "d_model": 64,  "n_layers": 4, "n_heads": 4, "d_ff": 256},
        {"name": "medium-4", "d_model": 128, "n_layers": 4, "n_heads": 8, "d_ff": 512},
        {"name": "large-6",  "d_model": 256, "n_layers": 6, "n_heads": 8, "d_ff": 1024},
    ]
