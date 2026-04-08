# 第4章 数值精度与低精度训练

本章是全文核心，系统介绍 FP32/FP16/BF16/FP8/MXFP 等精度格式的数值特性，以及混合精度训练的原理与实践。

## 4.1 浮点数值基础

### 4.1.1 IEEE 754 标准概述

浮点数由三部分组成：符号位 + 指数位 + 尾数位

$$
value = (-1)^{sign} \times 2^{exponent-bias} \times 1.mantissa
$$

### 4.1.2 各精度格式对比

| 格式 | 指数位 | 尾数位 | 动态范围 | 机器精度 | 典型用途 |
|------|--------|--------|----------|----------|----------|
| FP32 | 8 | 23 | ~1.7e38 | ~1e-7 | 主权重、损失计算 |
| FP16 | 5 | 10 | 6.55e4 | ~1e-3 | 前向/反向计算 |
| BF16 | 8 | 7 | ~1e38 | ~1e-2 | 训练默认格式 |
| E4M3 | 4 | 3 | 448 | ~0.125 | FP8 前向传播 |
| E5M2 | 5 | 2 | 57,344 | ~0.25 | FP8 梯度计算 |
| FP8-MX | 共享 | 3 | 块级 | 可变 | 微缩放 |

**信源**: IEEE 754 标准 — IEEE (电气电子工程师学会)

### 4.1.3 动态范围与精度权衡

```python
import torch

# 展示不同精度的数值范围
print("FP32 范围:", torch.finfo(torch.float32).min, "to", torch.finfo(torch.float32).max)
print("FP16 范围:", torch.finfo(torch.float16).min, "to", torch.finfo(torch.float16).max)
print("BF16 范围:", torch.finfo(torch.bfloat16).min, "to", torch.finfo(torch.bfloat16).max)
```

### 4.1.4 次正规数（Subnormal）问题

次正规数（denormalized numbers）用于表示接近零的极小值，但计算极慢。

```python
# 检测次正规数
def count_subnormals(tensor):
    fp32_info = torch.finfo(torch.float32)
    subnormal_mask = (tensor.abs() > 0) & (tensor.abs() < fp32_info.tiny)
    return subnormal_mask.sum().item()

# PyTorch 可禁用次正规数以提升性能
torch.set_flush_denormal(True)
```

## 4.2 混合精度训练机制

### 4.2.1 GradScaler 原理

动态损失缩放防止梯度下溢：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    with autocast(dtype=torch.bfloat16):
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**机构**: [Micikevicius et al. 2018] — NVIDIA, Baidu Research, UC Berkeley  
**信源**: [Mixed Precision Training](https://arxiv.org/abs/1710.03740)

### 4.2.2 梯度下溢检测

```python
def check_gradient_underflow(model):
    underflow_count = 0
    total_params = 0
    
    for param in model.parameters():
        if param.grad is not None:
            total_params += param.grad.numel()
            underflow_count += (param.grad.abs() < 1e-7).sum().item()
    
    underflow_ratio = underflow_count / total_params
    return underflow_ratio > 0.01
```

## 4.3 BF16 vs FP16 选择指南

### 4.3.1 硬件支持矩阵

| 硬件 | FP16 TensorCore | BF16 | FP8 | MXFP8 |
|------|-----------------|------|-----|-------|
| V100 | ✅ | ❌ | ❌ | ❌ |
| A100 | ✅ | ✅ | ❌ | ❌ |
| H100 | ✅ | ✅ | ✅ | ❌ |
| B200 | ✅ | ✅ | ✅ | ✅ |

**信源**: [BFloat16: The secret to high performance on Cloud TPUs](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus) — Google Brain

### 4.3.2 场景化选择建议

**选择 BF16 的场景**:
- 使用 A100/H100/TPU 等新型硬件
- 训练极深网络（梯度可能极小或极大）
- 不想处理损失缩放复杂性

**选择 FP16 的场景**:
- 使用 V100 等旧硬件
- 需要更高精度（如某些科学计算）

## 4.4 FP8 训练前沿

### 4.4.1 E4M3 vs E5M2 格式

- **E4M3**: 4 位指数，3 位尾数，范围 ±448，用于前向传播
- **E5M2**: 5 位指数，2 位尾数，范围 ±57,344，用于梯度计算

**信源**: [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) — NVIDIA, Arm, Intel, Qualcomm

### 4.4.2 Transformer Engine 实践

```python
import transformer_engine.pytorch as te

layer = te.Linear(768, 3072)

with te.fp8_autocast():
    output = layer(input)
```

### 4.4.3 缩放因子管理

```python
class FP8Scaler:
    def __init__(self):
        self.forward_scale = 1.0
        self.backward_scale = 1.0
    
    def compute_scaling(self, tensor, format='e4m3'):
        max_val = tensor.abs().max()
        max_representable = 448.0 if format == 'e4m3' else 57344.0
        scale = max_representable / (max_val * 1.25)
        return scale
```

## 4.5 微缩放（Microscaling/MXFP）技术

### 4.5.1 MXFP8 格式

**机构**: [Rouhani et al. 2023] — Microsoft Research  
**信源**: [Microscaling Formats for Deep Learning](https://arxiv.org/abs/2310.10537)

### 4.5.2 DeepSeek-V3 细粒度量化实践

**机构**: DeepSeek (幻方量化)  
**信源**: [DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v1#S5)

## 4.6 量化感知训练（QAT）稳定性

```python
class FakeQuantize(nn.Module):
    def __init__(self, num_bits=8):
        super().__init__()
        self.num_bits = num_bits
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        if self.training:
            x_quant = torch.round(x / self.scale)
            x_quant = torch.clamp(x_quant, -128, 127)
            x_dequant = x_quant * self.scale
            return x_dequant + (x - x_dequant).detach()
        return x
```

## 4.7 本章小结

| 主题 | 关键要点 |
|------|----------|
| 格式选择 | BF16 为现代硬件默认；FP8 用于极致性能 |
| 混合精度 | 使用 GradScaler 防止梯度下溢 |
| FP8 训练 | E4M3 用于前向，E5M2 用于反向 |

---

**上一章**: [第3章 模型架构稳定性](./03-architecture-stability.md) | **下一章**: [第5章 算法设计层面的稳定性保障](./05-algorithm-design.md)
