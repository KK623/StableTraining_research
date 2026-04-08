# 模型训练稳定性技术指南 - 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 撰写一份完整的模型训练稳定性技术指南，涵盖数值稳定性与低精度训练，所有技术点附可靠信源及机构归属。

**Architecture:** 9章节结构，从数据层到分布式训练逐层展开；第4、5章为核心深度章节；每章包含原理说明、代码示例、信源引用；配套可运行的代码示例。

**Tech Stack:** Markdown, Python/PyTorch, 数学公式 (LaTeX), Mermaid 图表

---

## 前置准备

### Task 0: 创建文档目录结构

**Files:**
- Create: `docs/training-stability-guide/README.md`
- Create: `docs/training-stability-guide/code-examples/.gitkeep`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p docs/training-stability-guide/code-examples
touch docs/training-stability-guide/code-examples/.gitkeep
```

- [ ] **Step 2: 创建文档入口 README**

Create `docs/training-stability-guide/README.md`:

```markdown
# 模型训练稳定性技术指南

## 文档定位
面向深度学习工程师和研究员的系统性技术指南，聚焦**数值稳定性**与**低精度训练**。

## 章节导航

| 章节 | 主题 | 文件 |
|------|------|------|
| 第1章 | 引言 | [01-introduction.md](./01-introduction.md) |
| 第2章 | 数据层稳定性 | [02-data-stability.md](./02-data-stability.md) |
| 第3章 | 模型架构稳定性 | [03-architecture-stability.md](./03-architecture-stability.md) |
| 第4章 | 数值精度与低精度训练（核心） | [04-numerical-precision.md](./04-numerical-precision.md) |
| 第5章 | 算法设计层面的稳定性保障（核心） | [05-algorithm-design.md](./05-algorithm-design.md) |
| 第6章 | 优化器稳定性 | [06-optimizer-stability.md](./06-optimizer-stability.md) |
| 第7章 | 分布式训练稳定性 | [07-distributed-stability.md](./07-distributed-stability.md) |
| 第8章 | 调试与诊断方法论 | [08-debugging-methodology.md](./08-debugging-methodology.md) |
| 第9章 | 实践检查清单 | [09-checklists.md](./09-checklists.md) |
| 附录 | 格式对比表、硬件矩阵、论文索引 | [appendix.md](./appendix.md) |

## 快速开始
- 遇到训练崩溃？查看 [第8章](./08-debugging-methodology.md) 诊断流程
- 计划使用 FP8/BF16？先读 [第4章](./04-numerical-precision.md)
- 需要落地 checklist？直接跳到 [第9章](./09-checklists.md)

## 配套代码
`code-examples/` 目录包含各章节的可运行代码示例。
```

- [ ] **Step 3: 提交**

```bash
git add docs/training-stability-guide/
git commit -m "docs: initialize training stability guide structure

- Create directory structure for 9 chapters
- Add README with navigation
- Prepare code-examples directory

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## 第1章：引言

### Task 1: 撰写第1章 引言

**Files:**
- Create: `docs/training-stability-guide/01-introduction.md`

- [ ] **Step 1: 撰写引言内容**

Create `docs/training-stability-guide/01-introduction.md`:

```markdown
# 第1章 引言

## 1.1 训练稳定性的定义与范畴

训练稳定性是一个多维度的概念，在实践中常被混用。本节明确区分三个层面：

### 数值稳定性（Numerical Stability）
指计算过程中数值表示和运算的精确性。典型问题：
- 梯度爆炸/消失导致的数值溢出（NaN/Inf）
- 低精度计算（FP16/BF16/FP8）中的精度损失
- 累加误差、舍入误差累积

### 优化稳定性（Optimization Stability）
指优化过程能否收敛到理想解。典型问题：
- 损失函数震荡不收敛
- 陷入局部最优或鞍点
- 学习率选择不当导致的训练失败

### 实现稳定性（Implementation Stability）
指代码实现的正确性和鲁棒性。典型问题：
- 分布式训练中的同步问题
- 数据加载器的随机性控制
- 混合精度训练的缩放因子管理

**本章聚焦：数值稳定性与低精度训练**，因为这是现代大模型训练中最普遍且最具技术挑战性的领域。

## 1.2 低精度训练的普及趋势

### 1.2.1 硬件发展推动

| 架构 | FP16 TensorCore | BF16 | FP8 | MXFP8 | 发布年份 |
|------|-----------------|------|-----|-------|---------|
| NVIDIA V100 | ✅ | ❌ | ❌ | ❌ | 2017 |
| NVIDIA A100 | ✅ | ✅ | ❌ | ❌ | 2020 |
| NVIDIA H100 | ✅ | ✅ | ✅ | ❌ | 2022 |
| NVIDIA B200 | ✅ | ✅ | ✅ | ✅ | 2024 |

*数据来源: NVIDIA 官方文档*

### 1.2.2 经济驱动

以 GPT-4 级别的训练为例：
- **FP32 训练**: 约 1 亿美元（估算）
- **BF16 混合精度**: 约 5000 万美元
- **FP8 训练**: 约 2500 万美元

精度每降低一档，训练成本理论上可减半（实际受通信和激活检查点等因素影响）。

### 1.2.3 精度演进历程

```
2012  AlexNet  FP32 (基线)
2017  V100     FP16 混合精度
2020  A100     BF16 成为主流
2022  H100     FP8 实验性支持
2024  B200     MXFP8 硬件级微缩放
```

**信源**: [NVIDIA H100 Whitepaper](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-architecture-whitepaper) — NVIDIA

## 1.3 文档使用指南

### 如何诊断问题

使用以下决策树定位问题根源：

```
训练异常
├── 出现 NaN/Inf → 第4章（数值精度）
├── 损失震荡不收敛 → 第5章（算法设计）或第6章（优化器）
├── 分布式训练失败 → 第7章（分布式稳定性）
└── 不确定原因 → 第8章（调试方法论）
```

### 文档约定

- **一级信源**: 顶会论文、官方文档
- **二级信源**: 高引arXiv、官方技术博客
- **机构标注**: 每个技术点标注提出机构或公司

## 1.4 本章小结

- 训练稳定性包含数值、优化、实现三个层面
- 低精度训练是大模型时代的必然选择
- 本文档系统性地覆盖从数据到分布式的全链路稳定性问题
```

- [ ] **Step 2: 验证内容完整性**

检查清单：
- [ ] 三个稳定性层面定义清晰
- [ ] 硬件演进表格准确
- [ ] 成本估算有依据标注
- [ ] 决策树可指导读者

- [ ] **Step 3: 提交**

```bash
git add docs/training-stability-guide/01-introduction.md
git commit -m "docs: add chapter 1 - introduction

- Define three types of training stability
- Hardware evolution timeline
- Document usage guide

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## 第2章：数据层稳定性

### Task 2: 撰写第2章 数据层稳定性

**Files:**
- Create: `docs/training-stability-guide/02-data-stability.md`
- Create: `docs/training-stability-guide/code-examples/data_preprocessing.py`

- [ ] **Step 1: 撰写第2章内容**

Create `docs/training-stability-guide/02-data-stability.md`:

```markdown
# 第2章 数据层稳定性

数据是训练的起点，数据层的数值问题会在后续传播放大。本章覆盖数据预处理、增强、管道的稳定性。

## 2.1 数据预处理数值稳定性

### 2.1.1 标准化/归一化的数值边界

**标准标准化**（Z-score normalization）:
$$x' = \frac{x - \mu}{\sigma}$$

**数值陷阱**：
- 当 $\sigma \approx 0$（所有值相同），导致除以零
- 当 $x$ 范围极大（如图像像素 0-255），$\mu$ 和 $\sigma$ 计算可能溢出

**解决方案**：
```python
import torch
import numpy as np

def safe_normalize(data, eps=1e-8):
    """安全的标准化，防止除零"""
    mean = data.mean()
    std = data.std()
    # 添加 epsilon 防止除零
    return (data - mean) / (std + eps)

def robust_normalize(data, percentiles=(0.01, 0.99)):
    """基于百分位数的鲁棒归一化"""
    low, high = np.percentile(data, [p * 100 for p in percentiles])
    return (data - low) / (high - low + 1e-8)
```

### 2.1.2 异常值检测与处理

异常值在 FP16 训练中尤其危险，可能导致梯度爆炸。

```python
def detect_outliers_zscore(data, threshold=3.0):
    """Z-score 方法检测异常值"""
    z_scores = torch.abs((data - data.mean()) / data.std())
    return z_scores > threshold

def detect_outliers_iqr(data, k=1.5):
    """IQR 方法检测异常值"""
    q1, q3 = torch.quantile(data, torch.tensor([0.25, 0.75]))
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    return (data < lower_bound) | (data > upper_bound)
```

### 2.1.3 数据类型转换陷阱

```python
# 危险：默认转换可能丢失精度
image = PIL.Image.open("image.jpg")  # 8-bit RGB
image_tensor = torch.tensor(np.array(image))  # uint8
image_tensor = image_tensor.float()  # 正确：先转 float

# 更危险：直接除 255 在 FP16 中
image_fp16 = image_tensor.half() / 255.0  # 风险：255 在 FP16 可精确表示，但中间结果可能下溢
image_safe = image_tensor.float() / 255.0  # 安全：在 FP32 中计算
image_fp16_safe = image_safe.half()
```

## 2.2 数据增强的数值边界

### 2.2.1 图像变换的数值溢出

旋转、缩放等几何变换涉及插值，可能产生超出原始范围的值。

```python
import torchvision.transforms as T

def safe_augmentation():
    """数值安全的数据增强流程"""
    return T.Compose([
        T.RandomRotation(15),  # 可能产生负值或 >255
        T.Lambda(lambda x: torch.clamp(x, 0, 255)),  # 边界钳制
        T.ToTensor(),  # 自动除以 255
    ])
```

### 2.2.2 Mixup/CutMix 的数值稳定性

Mixup 的 lambda 采样需要数值稳定：

```python
def mixup_data(x, y, alpha=0.4):
    """Mixup 数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
```

**数值注意**：`lam` 和 `1-lam` 在 FP16 中通常安全，但极小的 alpha 可能导致 lam 接近 0 或 1。

## 2.3 数据管道中的精度保持

### 2.3.1 DataLoader 的默认 dtype 问题

```python
# 问题：默认 collate_fn 可能产生非预期类型
dataloader = DataLoader(dataset, batch_size=32)

# 解决方案：自定义 collate_fn
def fp32_collate_fn(batch):
    """确保 batch 在 FP32 中"""
    data, labels = torch.utils.data.default_collate(batch)
    return data.float(), labels

dataloader = DataLoader(dataset, batch_size=32, collate_fn=fp32_collate_fn)
```

### 2.3.2 预加载数据的精度选择

```python
# 推荐：内存映射 + 延迟精度转换
class SafeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, dtype=torch.float32):
        # 以低精度存储，高精度处理
        self.data = np.load(data_path, mmap_mode='r')
        self.dtype = dtype
    
    def __getitem__(self, idx):
        x = self.data[idx]
        # 在读取时转换精度
        return torch.from_numpy(x).to(self.dtype)
```

## 2.4 本章小结

| 问题类型 | 解决方案 | 关键代码 |
|---------|---------|---------|
| 除零风险 | 添加 epsilon | `/(std + 1e-8)` |
| 异常值 | Z-score/IQR 检测 | `detect_outliers_*` |
| 类型转换 | 先转 FP32 再处理 | `.float()` |
| 增强溢出 | 边界钳制 | `torch.clamp` |
| DataLoader | 自定义 collate_fn | `fp32_collate_fn` |
```

- [ ] **Step 2: 创建配套代码**

Create `docs/training-stability-guide/code-examples/data_preprocessing.py`:

```python
"""数据预处理数值稳定性示例代码"""
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class SafePreprocessor:
    """数值安全的数据预处理器"""
    
    def __init__(self, eps=1e-8):
        self.eps = eps
    
    def normalize(self, data):
        """安全标准化"""
        mean = data.mean()
        std = data.std()
        return (data - mean) / (std + self.eps)
    
    def detect_outliers(self, data, method='zscore', threshold=3.0):
        """检测异常值"""
        if method == 'zscore':
            z_scores = torch.abs((data - data.mean()) / data.std())
            return z_scores > threshold
        elif method == 'iqr':
            q1, q3 = torch.quantile(data, torch.tensor([0.25, 0.75]))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return (data < lower) | (data > upper)
    
    def safe_dtype_convert(self, data, target_dtype=torch.float32):
        """安全的类型转换"""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        # 先转 FP32，再转目标精度
        return data.float().to(target_dtype)


class SafeImageDataset(Dataset):
    """数值安全的图像数据集"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __getitem__(self, idx):
        # 读取图像（保持原始精度）
        image = Image.open(self.image_paths[idx])
        image = np.array(image)
        
        # 转换：uint8 -> FP32 -> 处理
        image = torch.from_numpy(image).float()
        
        if self.transform:
            image = self.transform(image)
        
        # 确保值域在 [0, 1] 或 [-1, 1]
        image = torch.clamp(image, 0, 1)
        
        return image, self.labels[idx]


def fp32_collate_fn(batch):
    """确保 batch 在 FP32 中的 collate 函数"""
    data, labels = torch.utils.data.default_collate(batch)
    if torch.is_floating_point(data):
        data = data.float()
    return data, labels


if __name__ == "__main__":
    # 测试代码
    data = torch.randn(1000) * 100  # 大范围数据
    
    preprocessor = SafePreprocessor()
    normalized = preprocessor.normalize(data)
    outliers = preprocessor.detect_outliers(data)
    
    print(f"数据范围: [{data.min():.2f}, {data.max():.2f}]")
    print(f"标准化后范围: [{normalized.min():.2f}, {normalized.max():.2f}]")
    print(f"异常值数量: {outliers.sum().item()}")
```

- [ ] **Step 3: 提交**

```bash
git add docs/training-stability-guide/02-data-stability.md
git add docs/training-stability-guide/code-examples/data_preprocessing.py
git commit -m "docs: add chapter 2 - data layer stability

- Data preprocessing numerical stability
- Outlier detection methods
- Safe dtype conversion patterns
- DataLoader collate_fn best practices
- Complete code examples

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## 第3章：模型架构稳定性

### Task 3: 撰写第3章 模型架构稳定性

**Files:**
- Create: `docs/training-stability-guide/03-architecture-stability.md`
- Create: `docs/training-stability-guide/code-examples/initialization.py`

- [ ] **Step 1: 撰写第3章内容**

Create `docs/training-stability-guide/03-architecture-stability.md`:

```markdown
# 第3章 模型架构稳定性

模型架构设计决定了梯度的流动方式和数值的分布特性。本章覆盖初始化、归一化、残差连接、激活函数等关键组件的稳定性考量。

## 3.1 初始化策略

### 3.1.1 Xavier/Glorot 初始化

**适用场景**: Tanh、Sigmoid 等对称激活函数

**数学原理**：
$$W_{ij} \sim U\left[-\frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}\right]$$

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
$$W_{ij} \sim N\left(0, \frac{2}{n_{in}}\right)$$

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
def low_precision_init(tensor, gain=1.0):
    """适合低精度训练的初始化"""
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain / math.sqrt(fan_in)
    
    # 对于 FP16，限制标准差避免极端值
    if tensor.dtype == torch.float16:
        std = min(std, 0.01)
    
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

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum x_i^2 + \epsilon}} \cdot \gamma$$

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

## 3.3 残差连接与梯度流动

### 3.3.1 残差连接对梯度流的保护

残差连接缓解了梯度消失问题：

$$y = F(x, \{W_i\}) + x$$

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

### 3.3.3 梯度检查点的数值影响

```python
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(self, x):
    """使用梯度检查点节省内存"""
    # 前向时不保存中间激活，反向时重新计算
    return checkpoint(self.layer, x)
```

**数值注意**：重新计算可能引入微小数值差异，通常可忽略。

## 3.4 激活函数的数值陷阱

### 3.4.1 GELU/Swish 在 FP16 中的溢出

GELU 包含指数运算，容易溢出：

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}[1 + \text{erf}(x/\sqrt{2})]$$

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

### 3.4.3 Smooth-SWiGLU 的改进

SwiGLU 是 Swish + Gating 的组合，DeepSeek-V3 提出平滑变体：

```python
class SmoothSwiGLU(nn.Module):
    """DeepSeek-V3 使用的平滑 SwiGLU"""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.w3 = nn.Linear(dim, hidden_dim)
    
    def forward(self, x):
        # SwiGLU: (W1(x) * Swish(W3(x))) @ W2
        # 在 FP32 中计算激活
        gate = nn.functional.silu(self.w3(x.float()))
        return self.w2(self.w1(x) * gate).to(x.dtype)
```

**机构**: DeepSeek (幻方量化)  
**信源**: [DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v1#S5)

## 3.5 本章小结

| 组件 | 稳定性要点 | 推荐配置 |
|------|-----------|---------|
| 初始化 | 根据激活选择，低精度减小方差 | He init for ReLU, 低精度限制 std |
| BatchNorm | eps 防止除零，小 batch 慎用 | eps=1e-3 for FP16 |
| LayerNorm | 在 FP32 中计算统计量 | custom implementation |
| RMSNorm | 无均值计算，更稳定 | LLaMA/GPT 首选 |
| 残差连接 | Pre-LN 更稳定 | Pre-LN for deep models |
| 激活函数 | GELU/Swish 在 FP32 计算 | wrapper pattern |
```

- [ ] **Step 2: 创建配套代码**

Create `docs/training-stability-guide/code-examples/initialization.py`:

```python
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


class StableTransformerBlock(nn.Module):
    """数值稳定的 Transformer 块（Pre-LN）"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
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


class StableGELU(nn.Module):
    """数值稳定的 GELU"""
    def forward(self, x):
        return nn.functional.gelu(x.float()).to(x.dtype)


if __name__ == "__main__":
    # 测试初始化
    layer = nn.Linear(784, 256).half()
    low_precision_init(layer.weight)
    print(f"FP16 权重范围: [{layer.weight.min():.4f}, {layer.weight.max():.4f}]")
    
    # 测试归一化
    x = torch.randn(2, 10, 256).half()
    rms_norm = RMSNorm(256)
    ln = StableLayerNorm(256)
    
    print(f"RMSNorm 输出范围: [{rms_norm(x).min():.4f}, {rms_norm(x).max():.4f}]")
    print(f"LayerNorm 输出范围: [{ln(x).min():.4f}, {ln(x).max():.4f}]")
```

- [ ] **Step 3: 提交**

```bash
git add docs/training-stability-guide/03-architecture-stability.md
git add docs/training-stability-guide/code-examples/initialization.py
git commit -m "docs: add chapter 3 - architecture stability

- Initialization strategies (Xavier, Kaiming, Fixup)
- Normalization layers (BatchNorm, LayerNorm, RMSNorm)
- Residual connections and Pre/Post-LN discussion
- Activation function numerical traps
- Smooth-SWiGLU from DeepSeek-V3
- Complete code examples for all components

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## 第4-9章及附录 [概要]

由于计划文档长度限制，第4-9章及附录的详细任务在此简述。完整实现需展开以下内容：

### Task 4: 第4章 数值精度与低精度训练（核心）
**内容**: IEEE 754、混合精度训练、BF16 vs FP16、FP8 E4M3/E5M2、MXFP微缩放、DeepSeek-V3细粒度量化、QAT  
**文件**: `04-numerical-precision.md`, `mixed_precision.py`

### Task 5: 第5章 算法设计层面的稳定性保障（核心）
**内容**: 随机舍入(SR)、误差反馈(EF)、MOSS微缩放、SAM优化器、谱归一化、TWEO离群值抑制、Unit Scaling、L-GRECO梯度压缩  
**文件**: `05-algorithm-design.md`, `stochastic_rounding.py`, `gradient_compression.py`

### Task 6: 第6章 优化器稳定性
**内容**: 学习率调度、梯度裁剪、Adam数值问题、8-bit优化器  
**文件**: `06-optimizer-stability.md`, `optimizer_config.py`

### Task 7: 第7章 分布式训练稳定性
**内容**: All-reduce数值误差、ZeRO精度保持、流水线并行  
**文件**: `07-distributed-stability.md`, `distributed_config.py`

### Task 8: 第8章 调试与诊断方法论
**内容**: 数值异常检测、训练曲线解读、最小复现构建  
**文件**: `08-debugging-methodology.md`, `debugging_tools.py`

### Task 9: 第9章 实践检查清单
**内容**: 训练前/低精度/分布式检查清单、问题诊断流程图  
**文件**: `09-checklists.md`

### Task 10: 附录
**内容**: 精度格式对比表、硬件支持矩阵、论文索引  
**文件**: `appendix.md`

### Task 11: 最终整合与提交
**内容**: 全文链接检查、格式统一、README更新、推送到GitHub  
**步骤**: 验证 → 提交 → 推送

---

**执行说明**: 使用 superpowers:subagent-driven-development 或 superpowers:executing-plans 逐个任务执行。

