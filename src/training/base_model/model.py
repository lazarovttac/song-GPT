import torch
import torch.nn as nn
from torch.nn import functional as F


# Attention Head
class Head(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()

        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)

        # Creates a mask to prevent lookahead bias
        # The token t can only pay attention to tokens before it and not after it
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v

        return out


# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    """Múltiples cabezas de self-attention en paralelo"""

    def __init__(self, config, head_size):
        super().__init__()

        self.heads = nn.ModuleList(
            [Head(config, head_size) for _ in range(config.n_head)]
        )

        # Projection layer
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))

        return out


# Feed-Forward Network
class FeedForward(nn.Module):
    """Simple feed-forward network"""

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            # Expand dimensions (like zooming in)
            nn.Linear(config.n_embd, 4 * config.n_embd),
            # Filter most important features
            nn.GELU(),
            # Reduce back to initial dimensions (zooming out)
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


# Transformer Block
class Block(nn.Module):
    """Un bloque de Transformer: comunicación seguida de computación"""

    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head

        self.sa = MultiHeadAttention(config, head_size)
        self.ffwd = FeedForward(config)

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        # Self-Attention
        x = x + self.sa(self.ln1(x))

        # Feed-Forward
        x = x + self.ffwd(self.ln2(x))
        return x


# Generative Pretrained Transformer
class GPTLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        torch.manual_seed(config.seed)

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        self.ln_f = nn.LayerNorm(config.n_embd)

        # Stores concepts
        # like Paris being the capital city of France
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        # At this point, the last token recieved
        # contains all the meaning of the input, so
        # we compare it with every concept know to the network
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]

            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
