import torch
import torch.nn as nn
from einops import einsum, rearrange
import student.training as t

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.W)
    
    def forward(self, x):
        output = einsum(x, self.W, "... d_in, d_out d_in->... d_out")
        return output


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings,embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight)

    def forward(self, token_ids):
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32) 
        mean_sq = torch.mean(x**2, dim = -1, keepdim=True)
        rms = torch.rsqrt(mean_sq + self.eps)
        rms_norm = x * rms
        return rms_norm.to(in_dtype) * self.weight


def silu(x):
    return  x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
    
    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RoPE(nn.Module):
    def __init__(self, d_k, theta, max_seq_length, device=None):
        super().__init__()
        self.d_k = d_k
        positions = torch.arange(max_seq_length, device=device)
        k = torch.arange(0, d_k//2, device=device)
        freq = 1.0 / (theta ** (2 * k/d_k))
        angles = einsum(positions, freq, "i,j->i j")

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x, token_positions):
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x_even = x[...,0::2]
        x_odd = x[...,1::2]

        rotate_even = x_even * cos - x_odd * sin
        rotate_odd = x_even * sin + x_odd * cos
        # output = torch.stack((rotate_even, rotate_odd), dim=-1)
        # return output.flatten(-2)
        out = rearrange([rotate_even, rotate_odd], "two ... d -> ... (d two)")
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads,max_seq_length, theta=None, device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // self.num_heads

        self.q_proj = Linear(d_model, d_model, device=device)
        self.k_proj = Linear(d_model, d_model, device=device)
        self.v_proj = Linear(d_model, d_model, device=device)
        self.out_proj = Linear(d_model, d_model, device=device)
        self.rope = (
            RoPE(d_k=self.d_k, theta=theta, max_seq_length=max_seq_length, device=device)
            if theta is not None
            else None
        )

    def forward(self, x, token_positions=None):
        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
    
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q,"... seq (heads d_k) -> ... heads seq d_k", heads = self.num_heads)
        k = rearrange(k,"... seq (heads d_k) -> ... heads seq d_k", heads = self.num_heads)
        v = rearrange(v,"... seq (heads d_k) -> ... heads seq d_k", heads = self.num_heads)

        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        attention = t.scaled_dot_product_attention(q, k, v, mask=mask)
        merged_attention_heads = rearrange(attention, "... heads seq d_k -> ... seq (heads d_k)", heads = self.num_heads)
        return self.out_proj(merged_attention_heads)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, max_seq_len, device=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model,device=device)
        self.ln2 = RMSNorm(d_model,device=device)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device)
        self.ffnn = SwiGLU(d_model,d_ff,device=device)

    def forward(self, x):
        y = x + self.attn(self.ln1(x))
        output = y + self.ffnn(self.ln2(y))
        return output


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, theta, device=None):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device)
        self.transformer_layers = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, theta, context_length, device=device) for i in range(num_layers)
        )
        self.norm = RMSNorm(d_model, device=device)
        self.ln = Linear(d_model, vocab_size)

    def forward(self, token_ids):
        x = self.token_embeddings(token_ids)
        for layer in self.transformer_layers:
            x = layer(x)
        logits = self.ln(self.norm(x))
        return logits
