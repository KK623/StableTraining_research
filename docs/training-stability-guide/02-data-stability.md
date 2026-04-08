# 第2章 数据层稳定性

数据是训练的起点，数据层的数值问题会在后续传播放大。本章覆盖数据预处理、增强、管道的稳定性。

## 2.1 数据预处理数值稳定性

### 2.1.1 标准化/归一化的数值边界

**标准标准化**（Z-score normalization）:

$$
x' = \frac{x - \mu}{\sigma}
$$

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

**Z-score 方法**：

```python
def detect_outliers_zscore(data, threshold=3.0):
    """Z-score 方法检测异常值"""
    z_scores = torch.abs((data - data.mean()) / data.std())
    return z_scores > threshold
```

**IQR 方法**：

```python
def detect_outliers_iqr(data, k=1.5):
    """IQR 方法检测异常值"""
    q1, q3 = torch.quantile(data, torch.tensor([0.25, 0.75]))
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    return (data < lower_bound) | (data > upper_bound)
```

**处理策略**：

```python
def handle_outliers(data, method='clip', lower=None, upper=None):
    """处理异常值"""
    if method == 'clip':
        return torch.clamp(data, lower, upper)
    elif method == 'remove':
        mask = (data >= lower) & (data <= upper)
        return data[mask]
    elif method == 'replace':
        median = data.median()
        data = data.clone()
        data[(data < lower) | (data > upper)] = median
        return data
```

### 2.1.3 数据类型转换陷阱

```python
from PIL import Image
import numpy as np
import torch

# 危险：默认转换可能丢失精度
image = Image.open("image.jpg")  # 8-bit RGB
image_tensor = torch.tensor(np.array(image))  # uint8

# 正确做法：先转 float 再处理
image_tensor = image_tensor.float()

# 更危险：直接除 255 在 FP16 中
# image_fp16 = image_tensor.half() / 255.0  # 风险：中间结果可能下溢

# 安全做法：在 FP32 中计算，再转 FP16
image_safe = image_tensor.float() / 255.0
image_fp16_safe = image_safe.half()
```

## 2.2 数据增强的数值边界

### 2.2.1 图像变换的数值溢出

旋转、缩放等几何变换涉及插值，可能产生超出原始范围的值。

```python
import torchvision.transforms as T

def safe_augmentation_pipeline():
    """数值安全的数据增强流程"""
    return T.Compose([
        T.RandomRotation(15),  # 可能产生负值或 >255
        T.Lambda(lambda x: torch.clamp(x, 0, 255)),  # 边界钳制
        T.ToTensor(),  # 自动除以 255，转为 [0, 1]
    ])
```

### 2.2.2 Mixup/CutMix 的数值稳定性

**Mixup** 的 lambda 采样需要数值稳定：

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

**CutMix** 实现：

```python
def cutmix_data(x, y, alpha=1.0):
    """CutMix 数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    
    # 生成随机裁剪框
    W = x.size()[2]
    H = x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # 调整 lambda 以匹配实际像素比例
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x, y, y[index], lam
```

## 2.3 数据管道中的精度保持

### 2.3.1 DataLoader 的默认 dtype 问题

```python
from torch.utils.data import DataLoader

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
    
    def __len__(self):
        return len(self.data)
```

### 2.3.3 多进程数据加载的数值一致性

```python
# 设置随机种子确保可复现性
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    worker_init_fn=worker_init_fn
)
```

## 2.4 特定数据类型的数值问题

### 2.4.1 图像数据 (0-255)

```python
class ImagePreprocessor:
    """图像预处理数值安全类"""
    
    @staticmethod
    def to_float(image):
        """uint8 [0,255] -> float32 [0, 1]"""
        return image.float() / 255.0
    
    @staticmethod
    def normalize(image, mean, std):
        """标准化，防止除零"""
        std = torch.where(std < 1e-8, torch.ones_like(std), std)
        return (image - mean) / std
    
    @staticmethod
    def denormalize(image, mean, std):
        """反标准化"""
        return image * std + mean
```

### 2.4.2 时序数据

```python
def normalize_timeseries(data, method='zscore'):
    """时序数据标准化"""
    if method == 'zscore':
        mean = data.mean(dim=-1, keepdim=True)
        std = data.std(dim=-1, keepdim=True)
        std = torch.clamp(std, min=1e-8)
        return (data - mean) / std
    elif method == 'minmax':
        min_val = data.min(dim=-1, keepdim=True)[0]
        max_val = data.max(dim=-1, keepdim=True)[0]
        range_val = torch.clamp(max_val - min_val, min=1e-8)
        return (data - min_val) / range_val
```

### 2.4.3 文本/嵌入数据

```python
def safe_embedding_lookup(embeddings, indices, padding_idx=None):
    """安全的嵌入查找"""
    # 检查索引越界
    if indices.max() >= embeddings.num_embeddings:
        raise ValueError(f"Index {indices.max()} out of range")
    
    output = embeddings(indices)
    
    # 处理 padding
    if padding_idx is not None:
        mask = (indices == padding_idx).unsqueeze(-1)
        output = output.masked_fill(mask, 0.0)
    
    return output
```

## 2.5 本章小结

| 问题类型 | 解决方案 | 关键代码 |
|---------|---------|---------|
| 除零风险 | 添加 epsilon | `/(std + 1e-8)` |
| 异常值 | Z-score/IQR 检测 | `detect_outliers_*` |
| 类型转换 | 先转 FP32 再处理 | `.float()` |
| 增强溢出 | 边界钳制 | `torch.clamp` |
| DataLoader | 自定义 collate_fn | `fp32_collate_fn` |
| 多进程 | 设置随机种子 | `worker_init_fn` |

---

**上一章**: [第1章 引言](./01-introduction.md) | **下一章**: [第3章 模型架构稳定性](./03-architecture-stability.md)
