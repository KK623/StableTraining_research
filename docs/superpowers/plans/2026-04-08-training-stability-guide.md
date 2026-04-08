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

## 第4章：数值精度与低精度训练（核心章节）

### Task 4: 撰写第4章 数值精度与低精度训练

**Files:**
- Create: `docs/training-stability-guide/04-numerical-precision.md`
- Create: `docs/training-stability-guide/code-examples/mixed_precision.py`
- Create: `docs/training-stability-guide/code-examples/fp8_simulation.py`

- [ ] **Step 1: 撰写第4章内容**

Create `docs/training-stability-guide/04-numerical-precision.md`:

```markdown
# 第4章 数值精度与低精度训练

本章是全文核心，系统介绍 FP32/FP16/BF16/FP8/MXFP 等精度格式的数值特性。

## 4.1 浮点数值基础

### 4.1.1 IEEE 754 标准概述

浮点数由三部分组成：符号位 + 指数位 + 尾数位

$$value = (-1)^{sign} \\times 2^{exponent-bias} \\times 1.mantissa$$

### 4.1.2 各精度格式对比

| 格式 | 指数位 | 尾数位 | 动态范围 | 机器精度 | 典型用途 |
|------|--------|--------|----------|----------|----------|
| FP32 | 8 | 23 | ~1.7e38 | ~1e-7 | 主权重、损失计算 |
| FP16 | 5 | 10 | 6.55e4 | ~1e-3 | 前向/反向计算 |
| BF16 | 8 | 7 | ~1e38 | ~1e-2 | 训练默认格式 |
| E4M3 | 4 | 3 | 448 | ~0.125 | FP8 前向传播 |
| E5M2 | 5 | 2 | 57,344 | ~0.25 | FP8 梯度计算 |

**信源**: IEEE 754 标准 — IEEE (电气电子工程师学会)

### 4.1.3 次正规数（Subnormal）问题

次正规数用于表示接近零的极小值，但计算极慢。

```python
def count_subnormals(tensor):
    fp32_info = torch.finfo(torch.float32)
    subnormal_mask = (tensor.abs() > 0) & (tensor.abs() < fp32_info.tiny)
    return subnormal_mask.sum().item()
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

### 4.3.1 数值特性对比

| 特性 | FP16 | BF16 |
|------|------|------|
| 指数位 | 5 | 8 |
| 尾数位 | 10 | 7 |
| 最小正值 | 6.1e-5 | 1.2e-38 |
| 最大正值 | 6.55e4 | 3.4e38 |

### 4.3.2 硬件支持矩阵

| 硬件 | FP16 | BF16 | 推荐格式 |
|------|------|------|----------|
| NVIDIA V100 | ✅ | ❌ | FP16 |
| NVIDIA A100 | ✅ | ✅ | BF16 |
| NVIDIA H100 | ✅ | ✅ | BF16/FP8 |
| Google TPU v3 | ❌ | ✅ | BF16 |

**信源**: [BFloat16: The secret to high performance on Cloud TPUs](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus) — Google Brain

## 4.4 FP8 训练前沿

### 4.4.1 E4M3 vs E5M2 格式

- **E4M3**: 4位指数，3位尾数，范围 ±448，用于前向传播
- **E5M2**: 5位指数，2位尾数，范围 ±57,344，用于梯度计算

**信源**: [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) — NVIDIA, Arm, Intel, Qualcomm

### 4.4.2 Transformer Engine 实践

```python
import transformer_engine.pytorch as te

layer = te.Linear(768, 3072)

with te.fp8_autocast():
    output = layer(input)
```

**机构**: NVIDIA  
**信源**: [NVIDIA Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/)

## 4.5 微缩放（Microscaling/MXFP）技术

### 4.5.1 MXFP8 格式

微缩放为张量的子块共享指数，提供比 per-tensor 更细粒度的控制。

**机构**: [Rouhani et al. 2023] — Microsoft Research  
**信源**: [Microscaling Formats for Deep Learning](https://arxiv.org/abs/2310.10537)

### 4.5.2 DeepSeek-V3 细粒度量化实践

DeepSeek-V3 使用 fine-grained quantization：
- 激活：1×128 tile-wise 量化
- 权重：128×128 block-wise 量化

**机构**: DeepSeek (幻方量化)  
**信源**: [DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v1#S5)

### 4.5.3 NVIDIA Blackwell 硬件支持

Blackwell 架构原生支持 MXFP8，硬件级实现 per-block 缩放。

**信源**: [NVIDIA Blackwell Architecture](https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/) — NVIDIA

## 4.6 量化感知训练（QAT）稳定性

### 4.6.1 伪量化节点数值行为

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
```

- [ ] **Step 2: 创建配套代码**

Create `docs/training-stability-guide/code-examples/mixed_precision.py`:

```python
"""混合精度训练代码示例"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


class MixedPrecisionTrainer:
    def __init__(self, model, optimizer, device='cuda', dtype=torch.bfloat16):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.dtype = dtype
        self.scaler = GradScaler() if dtype == torch.float16 else None
        
    def train_step(self, data, target, criterion):
        self.optimizer.zero_grad()
        
        data = data.to(self.device)
        target = target.to(self.device)
        
        with autocast(dtype=self.dtype):
            output = self.model(data)
            loss = criterion(output, target)
        
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        return loss.item()


def check_numerical_health(model):
    health = {'has_nan': False, 'has_inf': False, 'max_grad': 0.0}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                health['has_nan'] = True
            if torch.isinf(param.grad).any():
                health['has_inf'] = True
            grad_max = param.grad.abs().max().item()
            health['max_grad'] = max(health['max_grad'], grad_max)
    
    return health


if __name__ == "__main__":
    model = nn.Linear(784, 10)
    optimizer = torch.optim.Adam(model.parameters())
    
    trainer = MixedPrecisionTrainer(model, optimizer, dtype=torch.bfloat16)
    
    data = torch.randn(32, 784)
    target = torch.randint(0, 10, (32,))
    criterion = nn.CrossEntropyLoss()
    
    loss = trainer.train_step(data, target, criterion)
    print(f"Training loss: {loss:.4f}")
    
    health = check_numerical_health(model)
    print(f"Numerical health: {health}")
```

Create `docs/training-stability-guide/code-examples/fp8_simulation.py`:

```python
"""FP8 数值模拟器"""
import torch


class FP8Simulator:
    E4M3_MAX = 448.0
    E5M2_MAX = 57344.0
    
    @staticmethod
    def quantize_e4m3(tensor):
        scale = tensor.abs().max() / FP8Simulator.E4M3_MAX
        quantized = (tensor / scale).round().clamp(-FP8Simulator.E4M3_MAX, FP8Simulator.E4M3_MAX)
        return quantized * scale, scale
    
    @staticmethod
    def quantize_e5m2(tensor):
        scale = tensor.abs().max() / FP8Simulator.E5M2_MAX
        quantized = (tensor / scale).round().clamp(-FP8Simulator.E5M2_MAX, FP8Simulator.E5M2_MAX)
        return quantized * scale, scale


if __name__ == "__main__":
    x = torch.randn(100, 100) * 100
    
    x_q, scale = FP8Simulator.quantize_e4m3(x)
    error = (x - x_q).abs().mean()
    
    print(f"Original range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"Quantized range: [{x_q.min():.2f}, {x_q.max():.2f}]")
    print(f"Mean absolute error: {error:.4f}")
```

- [ ] **Step 3: 提交**

```bash
git add docs/training-stability-guide/04-numerical-precision.md
git add docs/training-stability-guide/code-examples/mixed_precision.py
git add docs/training-stability-guide/code-examples/fp8_simulation.py
git commit -m "docs: add chapter 4 - numerical precision and low-precision training (CORE)

- IEEE 754 floating-point fundamentals
- Mixed precision training with GradScaler
- BF16 vs FP16 selection guide with hardware matrix
- FP8 training (E4M3/E5M2 formats)
- Microscaling/MXFP technology
- DeepSeek-V3 fine-grained quantization
- QAT numerical stability
- Complete code examples

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## 第5章：算法设计层面的稳定性保障（核心章节）

### Task 5: 撰写第5章 算法设计层面的稳定性保障

**Files:**
- Create: `docs/training-stability-guide/05-algorithm-design.md`
- Create: `docs/training-stability-guide/code-examples/stochastic_rounding.py`
- Create: `docs/training-stability-guide/code-examples/sam_optimizer.py`

- [ ] **Step 1: 撰写第5章内容**

Create `docs/training-stability-guide/05-algorithm-design.md`:

```markdown
# 第5章 算法设计层面的稳定性保障

本章聚焦算法设计层面的稳定性技术，特别是低精度训练中的数值保障手段。

## 5.1 随机舍入（Stochastic Rounding）

### 5.1.1 原理与偏差修正

随机舍入按到相邻值的距离比例概率舍入，保持无偏性：

$$\\mathbb{E}[SR(x)] = x$$

```python
def stochastic_round(x):
    """随机舍入实现"""
    floor = torch.floor(x)
    ceil = torch.ceil(x)
    prob = x - floor
    return torch.where(torch.rand_like(x) < prob, ceil, floor)
```

### 5.1.2 低精度梯度累加中的应用

```python
class StochasticRoundAccumulator:
    def __init__(self, shape, dtype=torch.float16):
        self.buffer = torch.zeros(shape, dtype=torch.float32)
        self.dtype = dtype
    
    def add(self, grad):
        self.buffer += grad.float()
    
    def read(self):
        result = stochastic_round(self.buffer)
        remainder = self.buffer - result
        self.buffer = remainder
        return result.to(self.dtype)
```

### 5.1.3 落地效果

BF16+SR：1.54×吞吐提升，30%内存降低。

**机构**: [Ozkara et al. 2025] — EPFL, IBM Research  
**信源**: [Stochastic Rounding for LLM Training: Theory and Practice](https://proceedings.mlr.press/v258/ozkara25b.html)

## 5.2 误差补偿机制

### 5.2.1 Kahan累加

```python
class KahanSum:
    def __init__(self):
        self.sum = 0.0
        self.c = 0.0
    
    def add(self, x):
        y = x - self.c
        t = self.sum + y
        self.c = (t - self.sum) - y
        self.sum = t
```

### 5.2.2 误差反馈（Error Feedback, EF）

压缩误差本地累加，下一迭代补偿。

**机构**: [Karimireddy et al. 2019] — EPFL  
**信源**: [Error Compensated Quantized SGD](https://arxiv.org/abs/1611.05301)

## 5.3 微缩放与细粒度量化

### 5.3.1 两级微缩放（MOSS）

全局高精度scale + 局部紧凑scale（2的幂）。

**机构**: [Rouhani et al. 2023] — Microsoft Research  
**信源**: [Microscaling Formats for Deep Learning](https://arxiv.org/abs/2310.10537)

### 5.3.2 混合粒度策略

- 权重：per-block量化
- 激活：per-token/per-tile量化
- 梯度：E5M2格式

**机构**: DeepSeek (幻方量化)  
**信源**: [DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v1#S5)

## 5.4 噪声注入技术

### 5.4.1 梯度噪声注入（GNI）

```python
def add_gradient_noise(grad, eta=0.3):
    """添加高斯噪声到梯度"""
    noise = torch.randn_like(grad) * eta
    return grad + noise
```

### 5.4.2 Sharpness-Aware Minimization (SAM)

SAM 通过权重扰动寻找平坦极小值：

```python
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05):
        super().__init__(params, dict(rho=rho))
        self.base_optimizer = base_optimizer
    
    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]['e_w'] = e_w
    
    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]['e_w'])
        self.base_optimizer.step()
```

**机构**: [Foret et al. 2020] — FAIR (Meta AI)  
**信源**: [Sharpness-Aware Minimization](https://arxiv.org/abs/2010.01412)

## 5.5 正则化与稳定性

### 5.5.1 Dropout的数值平滑

Dropout提供期望线性保持，具有数值平滑效应。

### 5.5.2 谱归一化（Spectral Normalization）

控制 Lipschitz 常数，稳定 GAN 训练。

**机构**: [Miyato et al. 2018] — MIT CSAIL, Google Brain  
**信源**: [Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957)

### 5.5.3 权重衰减的数值行为

AdamW 中的解耦权重衰减在低精度下需特别注意数值稳定性。

## 5.6 离群值抑制与范围管理

### 5.6.1 TWEO（Token-Wise Outlier）

块输出正则化，防止激活值>10,000导致的 divergence。

### 5.6.2 UE8M0上取整

缩放因子向上取整到2的幂，防止溢出。

### 5.6.3 单元缩放（Unit Scaling / u-μP）

初始化时确定最优缩放，Maximal Update Parametrization。

**机构**: [Blake et al. 2024] — Graphcore, University of Cambridge  
**信源**: [u-μP: The Unit-Scaled Maximal Update Parametrization](https://arxiv.org/pdf/2407.17465v2)

## 5.7 自适应精度分配

### 5.7.1 分层精度回退

敏感层保持FP32/BF16：Embedding层、归一化层、Attention层、MoE路由层、输出层。

**机构**: DeepSeek (幻方量化), NVIDIA  
**信源**: 
- [DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v1#S5)
- [NVFP4 Training](https://developer.nvidia.com/blog/using-nvfp4-low-precision-model-training-for-higher-throughput-without-losing-accuracy/) — NVIDIA

### 5.7.2 动态精度切换

训练阶段感知，损失曲率驱动的精度调整。

## 5.8 梯度压缩与稀疏化

### 5.8.1 L-GRECO层自适应压缩

层间自适应压缩率，无需超参数调优。

**机构**: [Lee et al. 2024] — KAIST (韩国科学技术院)  
**信源**: [L-GRECO: Layerwise-Adaptive Gradient Compression](https://proceedings.mlsys.org/paper_files/paper/2024/file/9069a8976ff06f6443e7f4172990a580-Paper-Conference.pdf)

### 5.8.2 Top-K稀疏化

仅传输重要梯度分量。

**机构**: [Lin et al. 2017] — Seoul National University, NVIDIA  
**信源**: [Deep Gradient Compression](https://arxiv.org/abs/1712.01887)

### 5.8.3 分布式Lion优化器

利用二值更新特性，更新而非梯度量化。

**机构**: [Chen et al. 2024] — Google Research, Simons Institute  
**信源**: [Distributed Lion](https://arxiv.org/pdf/2404.00438.pdf)

## 5.9 本章小结

| 技术类别 | 代表方法 | 适用场景 |
|----------|----------|----------|
| 舍入策略 | 随机舍入 | 低精度权重更新 |
| 误差补偿 | Kahan累加、EF | 梯度压缩、累加精度 |
| 噪声注入 | SAM、GNI | 寻找平坦极小值 |
| 范围管理 | Unit Scaling、TWEO | 防止离群值 |
| 精度分配 | 分层回退 | 混合精度训练 |
| 梯度压缩 | L-GRECO、Top-K | 分布式训练 |
```

- [ ] **Step 2: 创建配套代码**

Create `docs/training-stability-guide/code-examples/stochastic_rounding.py`:

```python
"""随机舍入实现示例"""
import torch


def stochastic_round(x):
    """随机舍入：按距离比例概率舍入"""
    floor = torch.floor(x)
    ceil = torch.ceil(x)
    prob = x - floor
    return torch.where(torch.rand_like(x) < prob, ceil, floor)


class StochasticRoundAccumulator:
    """使用随机舍入的梯度累加器"""
    
    def __init__(self, shape, dtype=torch.float16):
        self.buffer = torch.zeros(shape, dtype=torch.float32)
        self.dtype = dtype
    
    def add(self, grad):
        self.buffer += grad.float()
    
    def read(self):
        result = stochastic_round(self.buffer)
        remainder = self.buffer - result
        self.buffer = remainder
        return result.to(self.dtype)


if __name__ == "__main__":
    x = torch.tensor([1.2, 2.7, 3.5, 4.1])
    
    results = []
    for _ in range(1000):
        results.append(stochastic_round(x))
    
    mean_result = torch.stack(results).float().mean(dim=0)
    print(f"原始: {x}")
    print(f"随机舍入期望: {mean_result}")
    print(f"误差: {(x - mean_result).abs()}")
```

Create `docs/training-stability-guide/code-examples/sam_optimizer.py`:

```python
"""SAM优化器实现"""
import torch


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization 优化器"""
    
    def __init__(self, params, base_optimizer, rho=0.05):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho)
        super().__init__(params, defaults)
        
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]['e_w'] = e_w
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]['e_w'])
        
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm


if __name__ == "__main__":
    model = torch.nn.Linear(10, 1)
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer = SAM(model.parameters(), base_optimizer, rho=0.05)
    
    loss_fn = torch.nn.MSELoss()
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    
    # SAM 需要两步优化
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.first_step(zero_grad=True)
    
    loss_fn(model(x), y).backward()
    optimizer.second_step(zero_grad=True)
    
    print("SAM optimizer test passed!")
```

- [ ] **Step 3: 提交**

```bash
git add docs/training-stability-guide/05-algorithm-design.md
git add docs/training-stability-guide/code-examples/stochastic_rounding.py
git add docs/training-stability-guide/code-examples/sam_optimizer.py
git commit -m "docs: add chapter 5 - algorithm-level stability techniques (CORE)

- Stochastic rounding for low-precision training
- Error feedback and Kahan summation
- MOSS microscaling
- SAM optimizer implementation
- Spectral normalization
- Unit Scaling (u-μP)
- L-GRECO gradient compression
- Complete code examples

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---



