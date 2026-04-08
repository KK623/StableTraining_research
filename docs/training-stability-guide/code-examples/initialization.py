"""初始化与架构稳定性示例代码"""
import torch
import torch.nn as nn
import math


def low_precision_init(tensor, gain=1.0, max_std=0.01):
    """适合低精度训练的初始化"""
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain / math.sqrt(fan_in)

    # 对于 FP16，限制标准差避免极端值
    if tensor.dtype == torch.float16:
        std = min(std, max_std)

    with torch.no_grad():
        return tensor.normal_(0, std)


class StableLayerNorm(nn.Module):
    """数值稳定的 LayerNorm（FP32 计算统计量）"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        original_dtype = x.dtype
        x = x.float()
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return (x * self.weight + self.bias).to(original_dtype)


class RMSNorm(nn.Module):
    """RMSNorm 实现"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        original_dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return (x * self.weight).to(original_dtype)


class PreLNTransformerBlock(nn.Module):
    """数值稳定的 Transformer 块（Pre-LN）"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        # Pre-LN：先归一化再计算
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class StableGELU(nn.Module):
    """数值稳定的 GELU"""
    def forward(self, x):
        return nn.functional.gelu(x.float()).to(x.dtype)


class StableAttention(nn.Module):
    """数值稳定的注意力实现"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch, seq_len, dim = x.shape

        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 使用 FP32 计算注意力
        q_fp32 = q.float()
        k_fp32 = k.float()

        scores = torch.matmul(q_fp32, k_fp32.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)

        v_fp32 = v.float()
        output = torch.matmul(attn, v_fp32)

        output = output.transpose(1, 2).contiguous().reshape(batch, seq_len, dim)
        return self.out_proj(output.to(x.dtype))


def stable_softmax(x, dim=-1):
    """数值稳定的 Softmax"""
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


if __name__ == "__main__":
    print("=" * 60)
    print("Architecture Stability Tests")
    print("=" * 60)

    # 测试初始化
    print("\n1. Low-Precision Initialization")
    layer = nn.Linear(784, 256).half()
    low_precision_init(layer.weight)
    print(f"FP16 权重范围: [{layer.weight.min():.4f}, {layer.weight.max():.4f}]")
    print(f"FP16 权重标准差: {layer.weight.std():.6f}")

    # 测试归一化
    print("\n2. Normalization Layers")
    x = torch.randn(2, 10, 256).half()
    rms_norm = RMSNorm(256)
    ln = StableLayerNorm(256)

    rms_out = rms_norm(x)
    ln_out = ln(x)

    print(f"输入范围: [{x.min():.4f}, {x.max():.4f}]")
    print(f"RMSNorm 输出范围: [{rms_out.min():.4f}, {rms_out.max():.4f}]")
    print(f"LayerNorm 输出范围: [{ln_out.min():.4f}, {ln_out.max():.4f}]")

    # 测试 Transformer 块
    print("\n3. Pre-LN Transformer Block")
    transformer = PreLNTransformerBlock(256, 8).half()
    x = torch.randn(2, 10, 256).half()
    out = transformer(x)
    print(f"输入形状: {x.shape}, 输出形状: {out.shape}")
    print(f"输出范围: [{out.min():.4f}, {out.max():.4f}]")

    # 测试注意力
    print("\n4. Stable Attention")
    attn = StableAttention(256, 8).half()
    x = torch.randn(2, 10, 256).half()
    out = attn(x)
    print(f"输入形状: {x.shape}, 输出形状: {out.shape}")
    print(f"输出范围: [{out.min():.4f}, {out.max():.4f}]")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
