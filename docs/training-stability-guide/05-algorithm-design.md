# 第5章 算法设计层面的稳定性保障

本章聚焦算法设计层面的稳定性技术，特别是低精度训练中的数值保障手段。

## 5.1 随机舍入（Stochastic Rounding）

### 5.1.1 原理与偏差修正

随机舍入按到相邻值的距离比例概率舍入，保持无偏性：

$$\mathbb{E}[SR(x)] = x$$

```python
def stochastic_round(x):
    floor = torch.floor(x)
    ceil = torch.ceil(x)
    prob = x - floor
    return torch.where(torch.rand_like(x) < prob, ceil, floor)
```

### 5.1.2 落地效果

BF16+SR：1.54×吞吐提升，30%内存降低。

**机构**: [Ozkara et al. 2025] — EPFL, IBM Research  
**信源**: [Stochastic Rounding for LLM Training](https://proceedings.mlr.press/v258/ozkara25b.html)

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

**机构**: [Karimireddy et al. 2019] — EPFL  
**信源**: [Error Compensated Quantized SGD](https://arxiv.org/abs/1611.05301)

## 5.3 微缩放与细粒度量化

### 5.3.1 两级微缩放（MOSS）

**机构**: [Rouhani et al. 2023] — Microsoft Research  
**信源**: [Microscaling Formats for Deep Learning](https://arxiv.org/abs/2310.10537)

### 5.3.2 混合粒度策略

**机构**: DeepSeek (幻方量化)  
**信源**: [DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v1#S5)

## 5.4 噪声注入技术

### 5.4.1 梯度噪声注入（GNI）

```python
def add_gradient_noise(grad, eta=0.3):
    noise = torch.randn_like(grad) * eta
    return grad + noise
```

### 5.4.2 Sharpness-Aware Minimization (SAM)

**机构**: [Foret et al. 2020] — FAIR (Meta AI)  
**信源**: [Sharpness-Aware Minimization](https://arxiv.org/abs/2010.01412)

## 5.5 正则化与稳定性

### 5.5.2 谱归一化（Spectral Normalization）

**机构**: [Miyato et al. 2018] — MIT CSAIL, Google Brain  
**信源**: [Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957)

## 5.6 离群值抑制与范围管理

### 5.6.3 单元缩放（Unit Scaling / u-μP）

**机构**: [Blake et al. 2024] — Graphcore, University of Cambridge  
**信源**: [u-μP](https://arxiv.org/pdf/2407.17465v2)

## 5.7 自适应精度分配

### 5.7.1 分层精度回退

**机构**: DeepSeek (幻方量化), NVIDIA  
**信源**: [DeepSeek-V3](https://arxiv.org/html/2412.19437v1#S5), [NVFP4](https://developer.nvidia.com/blog/)

## 5.8 梯度压缩与稀疏化

### 5.8.1 L-GRECO层自适应压缩

**机构**: [Lee et al. 2024] — KAIST  
**信源**: [L-GRECO](https://proceedings.mlsys.org/)

### 5.8.2 Top-K稀疏化

**机构**: [Lin et al. 2017] — Seoul National University, NVIDIA  
**信源**: [Deep Gradient Compression](https://arxiv.org/abs/1712.01887)

### 5.8.3 分布式Lion优化器

**机构**: [Chen et al. 2024] — Google Research  
**信源**: [Distributed Lion](https://arxiv.org/pdf/2404.00438.pdf)

## 5.9 本章小结

| 技术类别 | 代表方法 | 适用场景 |
|----------|----------|----------|
| 舍入策略 | 随机舍入 | 低精度权重更新 |
| 误差补偿 | Kahan累加、EF | 梯度压缩 |
| 噪声注入 | SAM、GNI | 寻找平坦极小值 |
| 范围管理 | Unit Scaling、TWEO | 防止离群值 |
| 精度分配 | 分层回退 | 混合精度训练 |
| 梯度压缩 | L-GRECO、Top-K | 分布式训练 |

---

**上一章**: [第4章 数值精度与低精度训练](./04-numerical-precision.md) | **下一章**: [第6章 优化器稳定性](./06-optimizer-stability.md)
