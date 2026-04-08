# 第3章 模型架构稳定性

模型架构设计决定了梯度的流动方式和数值的分布特性。本章覆盖初始化、归一化、残差连接、激活函数等关键组件的稳定性考量。

## 3.1 初始化策略

### 3.1.1 Xavier/Glorot 初始化

**适用场景**: Tanh、Sigmoid 等对称激活函数

**数学原理**：

$$
W_{ij} \sim U\left[-\frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}\right]
$$

目标：保持前向和后向传播的方差一致。

```python
import torch.nn as nn

# PyTorch 内置实现
layer = nn.Linear(784, 256)
nn.init.xavier_uniform_(layer.weight)
nn.init.zeros_(layer.bias)
```

**机构**: [Glorot & Bengio 2010] — University of Toronto  
**信源**: [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html)

### 3.1.2 Kaiming/He 初始化

**适用场景**: ReLU 及其变体

**数学原理**：

$$
W_{ij} \sim N\left(0, \frac{2}{n_{in}}\right)
$$

考虑 ReLU 的负值截断特性，放大方差。

```python
# PyTorch 内置实现
layer = nn.Linear(784, 256)
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

**机构**: [He et al. 2015] — Microsoft Research Asia  
**信源**: [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852)

### 3.1.3 Fixup 初始化

**适用场景**: 极深网络（100+ 层）无需归一化层

核心思想：缩放残差分支，初始时残差贡献为 0。

```python
class FixupLayer(nn.Module):
    """Fixup 初始化层示例"""
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.bias1 = nn.Parameter(torch.zeros(dim))
        self.bias2 = nn.Parameter(torch.zeros(dim))
        self.bias3 = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        return x + self.scale * (self.layer(x + self.bias1) + self.bias2) + self.bias3
```

**机构**: [Zhang et al. 2019] — University of Toronto, Vector Institute  
**信源**: [Fixup Initialization](https://arxiv.org/abs/1901.09321)

### 3.1.4 低精度训练中的初始化调整

在 FP16/BF16 训练中，初始化方差需要调整：

```python
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
```

## 3.2 归一化层的数值行为

### 3.2.1 BatchNorm 的数值稳定性

**关键参数**：`eps`（防止除零）

```python
# 默认 eps=1e-5，对于 FP16 可能需要增大
bn = nn.BatchNorm1d(256, eps=1e-3)  # FP16 训练建议

# 评估模式下的数值稳定
bn.eval()
with torch.no_grad():
    output = bn(input)  # 使用运行统计量
```

**数值陷阱**：
- 小 batch size（<32）时统计量不稳定
- 分布式训练中的同步 BN 可能累积数值误差

**机构**: [Ioffe & Szegedy 2015] — Google  
**信源**: [Batch Normalization](https://arxiv.org/abs/1502.03167)

### 3.2.2 LayerNorm 的 FP16 问题

LayerNorm 在 FP16 中容易出现溢出：

```python
class StableLayerNorm(nn.Module):
    """数值稳定的 LayerNorm"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        # 在 FP32 中计算均值和方差
        original_dtype = x.dtype
        x = x.float()
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return (x * self.weight + self.bias).to(original_dtype)
```

### 3.2.3 RMSNorm 的稳定性优势

RMSNorm 去除均值中心化，计算更简单，数值更稳定：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum x_i^2 + \epsilon}} \cdot \gamma
$$

```python
class RMSNorm(nn.Module):
    """RMSNorm 实现（LLaMA、T5 等使用）"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # FP32 计算范数
        original_dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return (x * self.weight).to(original_dtype)
```

**机构**: [Zhang & Sennrich 2019] — University of Edinburgh  
**信源**: [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)

### 3.2.4 GroupNorm 的数值特性

GroupNorm 在小 batch 时比 BatchNorm 更稳定：

```python
# GroupNorm 不受 batch size 影响
gn = nn.GroupNorm(num_groups=32, num_channels=256)

# 适用于小 batch 训练（如检测、分割任务）
```

## 3.3 残差连接与梯度流动

### 3.3.1 残差连接对梯度流的保护

残差连接缓解了梯度消失问题：

$$
y = F(x, \{W_i\}) + x
$$

梯度：$\frac{\partial y}{\partial x} = \frac{\partial F}{\partial x} + 1$

即使 $F$ 的梯度很小，+1 保证梯度不会消失。

### 3.3.2 Pre-LN vs Post-LN 的稳定性差异

```python
# Post-LN（原始 Transformer）
x = x + Sublayer(LN(x))  # 梯度路径短但数值可能爆炸

# Pre-LN（更稳定）
x = LN(x + Sublayer(x))  # 数值更稳定，但梯度路径长
```

**现代模型（LLaMA、GPT-3）普遍采用 Pre-LN**。

```python
class PreLNTransformerBlock(nn.Module):
    """Pre-LN Transformer 块"""
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
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x
```

### 3.3.3 梯度检查点的数值影响

```python
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(self, x):
    """使用梯度检查点节省内存"""
    # 前向时不保存中间激活，反向时重新计算
    return checkpoint(self.layer, x, use_reentrant=False)
```

**数值注意**：重新计算可能引入微小数值差异，通常可忽略。

## 3.4 激活函数的数值陷阱

### 3.4.1 GELU/Swish 在 FP16 中的溢出

GELU 包含指数运算，容易溢出：

$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}[1 + \text{erf}(x/\sqrt{2})]
$$

```python
class StableGELU(nn.Module):
    """数值稳定的 GELU"""
    def forward(self, x):
        # 在 FP32 中计算
        return nn.functional.gelu(x.float()).to(x.dtype)
```

### 3.4.2 ReLU 的 Dead Neuron 问题

ReLU 的负值截断可能导致神经元永久失活：

```python
# LeakyReLU 缓解此问题
activation = nn.LeakyReLU(negative_slope=0.01)

# 或 SiLU/Swish（平滑替代）
activation = nn.SiLU()  # x * sigmoid(x)
```

### 3.4.3 SwiGLU 与 Smooth-SWiGLU

SwiGLU 是 Swish + Gating 的组合：

```python
class SwiGLU(nn.Module):
    """SwiGLU 激活"""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.w3 = nn.Linear(dim, hidden_dim)
    
    def forward(self, x):
        # SwiGLU: (W1(x) * Swish(W3(x))) @ W2
        gate = nn.functional.silu(self.w3(x))
        return self.w2(self.w1(x) * gate)
```

DeepSeek-V3 提出平滑变体以改善 FP8 训练稳定性：

**机构**: DeepSeek (幻方量化)  
**信源**: [DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v1#S5)

## 3.5 注意力机制的数值稳定性

### 3.5.1 Softmax 的数值稳定性

```python
def stable_softmax(x, dim=-1):
    """数值稳定的 Softmax"""
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_shifted = x - x_max  # 防止指数溢出
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
```

### 3.5.2 Flash Attention 的数值精度

Flash Attention 在计算时保持 FP32 累加：

```python
# Flash Attention 自动处理数值精度
from flash_attn import flash_attn_func

output = flash_attn_func(q, k, v, causal=True)
```

### 3.5.3 长序列注意力的数值问题

```python
class StableAttention(nn.Module):
    """数值稳定的注意力实现"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
    
    def forward(self, q, k, v):
        # Q, K, V: [batch, seq, dim]
        batch, seq_len, dim = q.shape
        
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 使用 FP32 计算注意力
        q_fp32 = q.float()
        k_fp32 = k.float()
        
        scores = torch.matmul(q_fp32, k_fp32.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        
        v_fp32 = v.float()
        output = torch.matmul(attn, v_fp32)
        
        return output.to(q.dtype).transpose(1, 2).contiguous().view(batch, seq_len, dim)
```

## 3.6 本章小结

| 组件 | 稳定性要点 | 推荐配置 |
|------|-----------|----------|
| 初始化 | 根据激活选择，低精度减小方差 | He init for ReLU, 低精度限制 std |
| BatchNorm | eps 防止除零，小 batch 慎用 | eps=1e-3 for FP16 |
| LayerNorm | 在 FP32 中计算统计量 | custom implementation |
| RMSNorm | 无均值计算，更稳定 | LLaMA/GPT 首选 |
| 残差连接 | Pre-LN 更稳定 | Pre-LN for deep models |
| 激活函数 | GELU/Swish 在 FP32 计算 | wrapper pattern |
| 注意力 | Flash Attention 或 FP32 计算 | use flash_attn if available |

---

**上一章**: [第2章 数据层稳定性](./02-data-stability.md) | **下一章**: [第4章 数值精度与低精度训练](./04-numerical-precision.md)
