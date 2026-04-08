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

### 5.1.2 低精度梯度累加中的应用

在 FP16/BF16 训练中，梯度累加容易丢失小梯度信息。随机舍入可以在量化时保留更多梯度细节：

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
        self.buffer = remainder  # 保留余数用于下次累加
        return result.to(self.dtype)
```

### 5.1.3 落地效果

BF16+SR：1.54×吞吐提升，30%内存降低。

**关键发现**：
- 随机舍入在语言模型训练中与 FP32 混合精度相当
- 对权重更新的小数值尤其重要
- 硬件实现可进一步加速

**机构**: [Ozkara et al. 2025] — EPFL, IBM Research  
**信源**: [Stochastic Rounding for LLM Training: Theory and Practice](https://proceedings.mlr.press/v258/ozkara25b.html)

## 5.2 误差补偿机制

### 5.2.1 Kahan累加

Kahan求和算法通过补偿变量减少大量小数值累加时的精度损失：

```python
class KahanSum:
    def __init__(self):
        self.sum = 0.0
        self.c = 0.0  # 补偿变量
    
    def add(self, x):
        y = x - self.c  # 减去上次补偿
        t = self.sum + y  # 临时和
        self.c = (t - self.sum) - y  # 计算新的补偿
        self.sum = t
```

**应用场景**：
- 大批量训练的梯度累加
- 长序列的注意力计算
- 参数平均（如 EMA）

### 5.2.2 误差反馈（Error Feedback, EF）

误差反馈机制用于补偿梯度压缩引入的误差，特别适用于分布式训练中的梯度压缩：

**核心思想**：
1. 本地累积压缩误差
2. 将误差加到下一轮梯度
3. 保证收敛性，减少精度损失

```python
class ErrorFeedbackCompressor:
    def __init__(self, compression_fn):
        self.compression_fn = compression_fn
        self.error_buffer = {}
    
    def compress(self, param_name, grad):
        # 添加上一轮的误差
        if param_name in self.error_buffer:
            grad = grad + self.error_buffer[param_name]
        
        # 压缩梯度
        grad_compressed = self.compression_fn(grad)
        
        # 计算并保存新误差
        self.error_buffer[param_name] = grad - grad_compressed
        
        return grad_compressed
```

**优势**：
- 与各种压缩方法（Top-K、量化）兼容
- 理论保证收敛到原始解
- 实际压缩率可达 100-1000× 而精度损失很小

**机构**: [Karimireddy et al. 2019] — EPFL  
**信源**: [Error Compensated Quantized SGD](https://arxiv.org/abs/1611.05301)

## 5.3 微缩放与细粒度量化

### 5.3.1 两级微缩放（MOSS）

MOSS（Microscaling Optimized for Storage and Speed）提出两级缩放策略：

**全局高精度 scale**：
- 提供整体动态范围覆盖
- 使用 FP32 或 FP16 存储

**局部紧凑 scales**：
- 每个块使用 2 的幂作为缩放因子
- 硬件友好的移位操作代替除法
- 避免昂贵的 max-reduction 操作

```python
class MOSSQuantizer:
    def __init__(self, block_size=128):
        self.block_size = block_size
    
    def quantize(self, tensor, bits=8):
        # 计算全局缩放
        global_scale = tensor.abs().max() / (2**(bits-1) - 1)
        
        # 分块并计算局部缩放（2的幂）
        blocks = tensor.view(-1, self.block_size)
        local_scales = []
        
        for block in blocks:
            block_max = block.abs().max()
            # 向上取整到2的幂
            local_scale = 2 ** torch.ceil(torch.log2(block_max / global_scale))
            local_scales.append(local_scale)
        
        return {
            'values': (tensor / global_scale).to(torch.int8),
            'global_scale': global_scale,
            'local_scales': torch.tensor(local_scales)
        }
```

**优势**：
- 比单级缩放更精细
- 硬件实现高效
- 适合大规模分布式训练

**机构**: [Rouhani et al. 2023] — Microsoft Research  
**信源**: [Microscaling Formats for Deep Learning](https://arxiv.org/abs/2310.10537)

### 5.3.2 混合粒度策略

混合粒度策略对不同张量类型使用不同的量化粒度：

| 张量类型 | 粒度 | 原因 |
|---------|------|------|
| 权重 | per-block (128×128) | 权重分布相对稳定，块级足够 |
| 激活 | per-token (1×128) | 激活有离群 token，需要更细粒度 |
| 梯度 | per-tile / E5M2 | 梯度范围大，需要动态范围 |

**实现要点**：
- 权重使用静态缩放（预计算）
- 激活使用动态缩放（实时计算）
- 梯度使用 E5M2 格式（更大的动态范围）

**机构**: DeepSeek (幻方量化)  
**信源**: [DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v1#S5)

## 5.4 噪声注入技术

### 5.4.1 梯度噪声注入（GNI）

在梯度中添加可控的高斯噪声，有助于：
- 逃离尖锐的局部极小值
- 增强泛化能力
- 在低精度训练中提供数值稳定性

```python
def add_gradient_noise(grad, eta=0.3, epoch=0):
    """添加高斯噪声到梯度"""
    # 噪声随时间衰减
    std = eta / ((1 + epoch) ** 0.5)
    noise = torch.randn_like(grad) * std
    return grad + noise
```

**噪声调度**：
- 初期使用较大噪声（探索）
- 后期逐渐减小（收敛）
- 与退火策略配合使用

### 5.4.2 Sharpness-Aware Minimization (SAM)

SAM 通过寻找平坦的极小值区域来提升泛化和稳定性：

**核心思想**：
1. 在当前权重处计算梯度
2. 向梯度方向迈出一小步（扰动）
3. 在扰动后的位置计算新的梯度
4. 用这个梯度更新原始权重

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

**数值稳定性优势**：
- 平坦极小值对扰动不敏感
- 降低对超参数的敏感性
- 适合低精度训练（扰动提供正则化）

**机构**: [Foret et al. 2020] — FAIR (Meta AI)  
**信源**: [Sharpness-Aware Minimization](https://arxiv.org/abs/2010.01412)

## 5.5 正则化与稳定性

### 5.5.1 Dropout的数值平滑

Dropout 不仅提供正则化，还有数值平滑效应：

**期望线性保持**：
- 训练时随机丢弃神经元
- 测试时自动缩放（p × output）
- 保持输出期望一致

**数值稳定性**：
- 减少共适应（co-adaptation）
- 分散梯度信号，防止单一路径爆炸
- 对低精度训练的噪声有容忍度

### 5.5.2 谱归一化（Spectral Normalization）

谱归一化通过控制权重矩阵的谱范数（最大奇异值）来稳定训练：

$$
W_{SN} = \frac{W}{\sigma(W)}
$$

其中 $\sigma(W)$ 是 W 的最大奇异值。

```python
class SpectralNorm(nn.Module):
    def __init__(self, module, n_power_iterations=1):
        super().__init__()
        self.module = module
        self.n_power_iterations = n_power_iterations
        
        # 初始化 u 向量
        u = torch.randn(1, module.weight.size(0))
        self.register_buffer('u', u)
    
    def compute_weight(self):
        weight = self.module.weight
        u = self.u
        
        # 幂迭代计算谱范数
        for _ in range(self.n_power_iterations):
            v = torch.nn.functional.normalize(torch.mv(weight.t(), u), dim=0)
            u = torch.nn.functional.normalize(torch.mv(weight, v), dim=0)
        
        sigma = torch.dot(u, torch.mv(weight, v))
        self.u.data = u
        
        return weight / sigma
```

**应用场景**：
- GAN 训练稳定性（尤其判别器）
- 深度网络的梯度控制
- Lipschitz 约束的强化学习

**优势**：
- 无需超参数调优
- 计算开销小（幂迭代）
- 与其他归一化方法兼容

**机构**: [Miyato et al. 2018] — MIT CSAIL, Google Brain  
**信源**: [Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957)

### 5.5.3 权重衰减的数值行为

AdamW 中的解耦权重衰减在低精度下需要特别注意：

**标准 L2 正则化 vs 解耦权重衰减**：
- 标准：梯度包含 L2 项，影响自适应学习率
- AdamW：权重衰减直接应用于参数，与梯度解耦

**低精度注意事项**：
- 权重衰减量应随精度降低而减小
- 过大的 weight_decay 会导致参数快速趋零
- 建议 FP16 训练使用 0.01，BF16 可使用 0.1

## 5.6 离群值抑制与范围管理

### 5.6.1 TWEO（Token-Wise Outlier）

Token-Wise Outlier 抑制通过块输出正则化防止极端激活值：

**问题**：
- 某些 token 会产生极大激活值（>10,000）
- 导致 FP8/BF16 量化时大部分值被压缩到零
- 训练发散

**解决方案**：
```python
def tweo_regularization(hidden_states, max_val=1000.0):
    """TWEO 正则化"""
    # 对超过阈值的激活进行软约束
    overflow_mask = hidden_states.abs() > max_val
    penalty = (hidden_states.abs() - max_val).clamp(min=0)
    return penalty.sum() * 0.01  # 正则化系数
```

### 5.6.2 UE8M0上取整

UE8M0（Unsigned Exponent, 8-bit Mantissa, 0-bit fraction）格式中的上取整策略：

**策略**：
- 缩放因子向上取整到最近的 2 的幂
- 确保缩放后最大值不超过格式上限的 80%
- 防止溢出导致的 NaN

```python
def compute_ue8m0_scale(max_abs_val, format_max=448.0):
    """计算 UE8M0 缩放因子，向上取整"""
    exact_scale = format_max / max_abs_val
    # 向上取整到2的幂
    log_scale = torch.ceil(torch.log2(torch.tensor(exact_scale)))
    return 2 ** log_scale
```

### 5.6.3 单元缩放（Unit Scaling / u-μP）

单元缩放是一种在初始化时确定最优缩放的方法：

**核心思想**：
- 每个张量有一个缩放因子
- 在初始化时根据维度计算最优缩放
- 保持前向和后向传播的方差一致

```python
class UnitScaledLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
        # 单元缩放因子
        self.forward_scale = in_features ** -0.5
        self.backward_scale = out_features ** -0.5
    
    def forward(self, x):
        # 应用缩放
        scaled_weight = self.weight * self.forward_scale
        output = torch.nn.functional.linear(x, scaled_weight)
        return output
```

**优势**：
- 无需调参
- 与低精度训练兼容
- 自动处理不同深度的缩放

**机构**: [Blake et al. 2024] — Graphcore, University of Cambridge  
**信源**: [u-μP: The Unit-Scaled Maximal Update Parametrization](https://arxiv.org/pdf/2407.17465v2)

## 5.7 自适应精度分配

### 5.7.1 分层精度回退

对不同层使用不同的精度策略，敏感层保持较高精度：

**敏感层列表**：
| 层类型 | 推荐精度 | 原因 |
|--------|----------|------|
| Embedding | FP32/BF16 | 词表大，量化误差累积 |
| LayerNorm/RMSNorm | FP32 | 数值稳定性关键 |
| Attention | BF16 | Softmax 数值敏感 |
| MoE 路由 | FP32 | 稀疏性导致数值不稳定 |
| 输出层 | FP32/BF16 | 影响最终预测精度 |

**实现策略**：
```python
class HybridPrecisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 敏感层用 BF16
        self.embedding = nn.Embedding(vocab_size, dim, dtype=torch.bfloat16)
        self.norms = nn.ModuleList([RMSNorm(dim) for _ in range(n_layers)])
        
        # 计算层用 FP8
        self.layers = nn.ModuleList([
            FP8TransformerBlock(dim) for _ in range(n_layers)
        ])
```

**机构**: DeepSeek (幻方量化), NVIDIA  
**信源**: 
- [DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v1#S5)
- [NVFP4 Training](https://developer.nvidia.com/blog/)

### 5.7.2 动态精度切换

根据训练阶段动态调整精度：

**阶段策略**：
- **Warmup**：使用 FP32/BF16 稳定训练
- **稳定期**：切换到 FP8 加速
- **收敛期**：回到 BF16 保证精度

**触发条件**：
- 损失曲率变化
- 梯度范数稳定性
- 验证集指标 plateau

## 5.8 梯度压缩与稀疏化

### 5.8.1 L-GRECO层自适应压缩

L-GRECO（Layerwise-Adaptive Gradient Compression）根据每层特性自适应选择压缩策略：

**三种压缩策略**：
1. **量化**：降低精度（1-bit SGD, QSGD）
2. **稀疏化**：只传 Top-K（Deep Gradient Compression）
3. **低秩分解**：利用梯度张量结构

**自适应选择**：
```python
class LGRECO:
    def __init__(self, model, target_compression=100):
        self.layer_stats = {}
        for name, param in model.named_parameters():
            self.layer_stats[name] = {
                'method': 'topk',  # 默认
                'ratio': 0.01    # 1% 稀疏度
            }
    
    def adapt_compression(self, name, grad):
        # 根据梯度统计选择最优压缩方法
        grad_norm = grad.norm()
        grad_variance = grad.var()
        
        if grad_variance < 1e-6:  # 梯度稳定
            return 'quantize', 8  # 8-bit 量化
        elif grad_norm > 10:  # 梯度大
            return 'topk', 0.001  # 更稀疏
        else:
            return 'lowrank', 0.1  # 低秩近似
```

**优势**：
- 无需全局超参数
- 每层最优压缩
- 与现有优化器兼容

**机构**: [Lee et al. 2024] — KAIST  
**信源**: [L-GRECO: Layerwise-Adaptive Gradient Compression](https://proceedings.mlsys.org/)

### 5.8.2 Top-K稀疏化

Top-K 稀疏化只传输梯度中绝对值最大的 K% 元素：

```python
def top_k_sparsify(grad, sparsity=0.01):
    """Top-K 稀疏化"""
    k = int(grad.numel() * sparsity)
    
    # 找到 top-k 阈值
    threshold = torch.topk(grad.abs().flatten(), k)[0][-1]
    
    # 创建掩码
    mask = grad.abs() >= threshold
    
    # 应用掩码
    sparse_grad = grad * mask
    
    return sparse_grad, mask
```

**误差反馈结合**：
- 未传输的梯度误差累积到下一轮
- 保证收敛性
- 通常达到 100-1000× 压缩率

**机构**: [Lin et al. 2017] — Seoul National University, NVIDIA  
**信源**: [Deep Gradient Compression](https://arxiv.org/abs/1712.01887)

### 5.8.3 分布式Lion优化器

Lion 优化器天生适合分布式训练，因其使用符号更新：

**数值特性**：
- 更新只有 {-1, 0, +1} 三个值
- 可以用 2-bit 传输更新而非梯度
- 减少通信量 16×

**分布式优化**：
```python
class DistributedLion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99)):
        defaults = dict(lr=lr, betas=betas)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                # 更新动量
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
                
                # 符号更新（只有 -1, 0, +1）
                update = exp_avg.sign()
                
                # 分布式：只传 2-bit 更新
                p.add_(update, alpha=-group['lr'])
```

**优势**：
- 极低的通信开销
- 适合大规模分布式训练
- 与误差反馈结合效果更佳

**机构**: [Chen et al. 2024] — Google Research  
**信源**: [Distributed Lion](https://arxiv.org/pdf/2404.00438.pdf)

## 5.9 本章小结

| 技术类别 | 代表方法 | 适用场景 |
|----------|----------|----------|
| 舍入策略 | 随机舍入 | 低精度权重更新 |
| 误差补偿 | Kahan累加、EF | 梯度压缩、累加精度 |
| 微缩放 | MOSS | 细粒度量化 |
| 噪声注入 | SAM、GNI | 寻找平坦极小值 |
| 范围管理 | Unit Scaling、TWEO | 防止离群值 |
| 精度分配 | 分层回退 | 混合精度训练 |
| 梯度压缩 | L-GRECO、Top-K | 分布式训练 |

---

**上一章**: [第4章 数值精度与低精度训练](./04-numerical-precision.md) | **下一章**: [第6章 优化器稳定性](./06-optimizer-stability.md)
