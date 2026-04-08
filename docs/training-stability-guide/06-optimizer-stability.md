# 第6章 优化器稳定性

优化器的选择和配置直接影响训练的数值稳定性，特别是在低精度环境下。

## 6.1 学习率调度与数值稳定性

### 6.1.1 Warmup 的必要性

**为什么需要 Warmup：**

1. **防止早期梯度爆炸**：初始化后的权重尚未稳定，大学习率会导致梯度范数激增
2. **稳定自适应优化器状态**：Adam 等优化器需要积累梯度统计量，初期统计量不稳定
3. **避免位置编码过度更新**：Transformer 的位置编码在早期容易因大学习率而发散

**数学原理：**

$$\eta_t = \eta_{max} \times \min\left(\frac{t}{T_w}, 1.0\right)$$

其中 $T_w$ 是 warmup 步数，通常占总步数的 1-10%。

**推荐配置：**

| 训练规模 | Warmup 比例 | 典型步数 |
|---------|-------------|---------|
| 小型 (<1B) | 5-10% | 500-1000 |
| 中型 (1-10B) | 2-5% | 1000-2000 |
| 大型 (>10B) | 1-2% | 2000-5000 |

```python
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

**信源**: [Gotmare et al. 2018] — Singapore University of Technology and Design  
**信源**: [A Closer Look at Deep Learning Heuristics with Learning Rate Warmups](https://arxiv.org/abs/1810.13243)

## 6.2 梯度裁剪策略

### 6.2.1 Norm Clipping vs Value Clipping

```python
# Norm Clipping（推荐）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Value Clipping
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

### 6.2.2 自适应梯度裁剪

```python
class AdaptiveGradientClipper:
    def __init__(self, initial_max_norm=1.0, target_norm=0.5, adaptation_rate=0.01):
        self.max_norm = initial_max_norm
        self.target_norm = target_norm
        self.adaptation_rate = adaptation_rate
        self.gradient_norms = []
    
    def clip(self, parameters):
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_norm)
        self.gradient_norms.append(total_norm.item())
        if len(self.gradient_norms) >= 100:
            avg_norm = sum(self.gradient_norms[-100:]) / 100
            if avg_norm > self.target_norm * 1.5:
                self.max_norm *= (1 - self.adaptation_rate)
            elif avg_norm < self.target_norm * 0.5:
                self.max_norm *= (1 + self.adaptation_rate)
            self.gradient_norms = []
```

## 6.3 自适应优化器的数值问题

### 6.3.1 Adam 的 epsilon 选择

```python
# 标准 Adam
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)

# FP16 训练建议增大 epsilon
optimizer_fp16 = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-4)

# AdamW
optimizer_adamw = torch.optim.AdamW(model.parameters(), lr=1e-3, eps=1e-8, weight_decay=0.01)
```

### 6.3.2 Lion 优化器的数值特性

**Lion (EvoLved Sign Momentum) 的核心特点：**

Lion 只使用梯度的符号（sign）进行更新，摒弃了梯度幅值信息：

$$\theta_t = \theta_{t-1} - \eta \cdot \text{sign}(\beta_1 m_{t-1} + (1-\beta_1) g_t)$$

其中 $m_t$ 是动量，更新规则为 $m_t = \beta_2 m_{t-1} + (1-\beta_2) g_t$。

**数值稳定性优势：**

1. **更新幅值恒定**：每次更新只有 {-1, 0, +1} 三个离散值，避免了梯度爆炸
2. **对异常值鲁棒**：离群梯度不会导致过大参数更新
3. **低精度友好**：符号操作在 FP16/BF16 下依然稳定
4. **分布式优势**：可用 2-bit 传输更新，通信量降低 16×

**与 Adam 的数值对比：**

| 特性 | Adam | Lion |
|-----|------|------|
| 二阶矩估计 | 需要 | 不需要 |
| 内存占用 | 2×参数 | 1×参数 |
| 更新值域 | 连续值 | {-1, 0, +1} |
| 对异常值敏感 | 是 | 否 |
| epsilon 敏感 | 是 | 否 |

**实现代码：**

```python
class Lion(torch.optim.Optimizer):
    """Lion 优化器实现
    
    数值特性：
    - 使用符号更新，避免梯度幅值异常
    - 无需要 epsilon 防止除零
    - 适合低精度训练
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # 初始化动量
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                # 权重衰减（解耦）
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # 更新动量（EMA of gradients）
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
                
                # 核心：符号更新，只取 {-1, 0, +1}
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(update.sign(), alpha=-group['lr'])
```

**低精度训练建议：**

```python
# Lion 在 FP16/BF16 下表现优异，无需特殊 epsilon
optimizer = Lion(model.parameters(), lr=3e-4, betas=(0.9, 0.99))

# 配合梯度裁剪（可选，Lion 本身较稳定）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**信源**: [Chen et al. 2023] — Google Brain  
**信源**: [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675)

## 6.4 低精度优化器状态

### 6.4.1 8-bit Adam 实践

```python
import bitsandbytes as bnb
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-3)
```

**机构**: [Dettmers et al. 2022] — University of Washington, Meta AI  
**信源**: [8-bit Optimizers](https://arxiv.org/abs/2110.02861)

### 6.4.2 优化器状态分片（ZeRO）

**ZeRO (Zero Redundancy Optimizer) 数值稳定性考虑：**

ZeRO 通过分片优化器状态降低内存占用，但引入了通信精度和数值一致性挑战。

**三阶段分片策略：**

| Stage | 分片内容 | 内存节省 | 通信量 | 数值风险 |
|-------|---------|---------|--------|---------|
| Stage 1 | 优化器状态 | 4× | 不变 | 低 |
| Stage 2 | 优化器状态 + 梯度 | 8× | 不变 | 中 |
| Stage 3 | 优化器状态 + 梯度 + 参数 | 与并行度线性相关 | 1.5× | 高 |

**关键数值稳定性问题：**

1. **分片精度损失**：参数分片在 GPU 间传输时需保持 FP32
2. **更新同步误差**：ZeRO-3 的参数更新需要 All-gather，需防止精度损失
3. **CPU Offloading 精度**：offload 到 CPU/NVMe 时保持 FP32

**稳定配置示例：**

```python
# DeepSpeed ZeRO-2 稳定配置
deepspeed_config = {
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,  # 动态损失缩放
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "gradient_clipping": 1.0  # 自动梯度裁剪
}
```

**ZeRO-3 参数更新数值保障：**

```python
class StableZeRO3Update:
    """确保 ZeRO-3 参数更新的数值稳定性"""
    
    def __init__(self, model, dtype=torch.float32):
        self.model = model
        self.master_dtype = dtype
        self.param_copies = {}
    
    def pre_update(self):
        """更新前：确保参数在 FP32"""
        for name, param in self.model.named_parameters():
            if param.dtype != self.master_dtype:
                self.param_copies[name] = param.data.clone()
                param.data = param.data.to(self.master_dtype)
    
    def post_update(self):
        """更新后：安全转回计算精度"""
        for name, param in self.model.named_parameters():
            if name in self.param_copies:
                # 检查更新是否产生 NaN
                if torch.isnan(param).any():
                    # 回退到更新前
                    param.data.copy_(self.param_copies[name])
                del self.param_copies[name]
```

**混合精度注意事项：**

```python
# FSDP (PyTorch 原生 ZeRO) 配置
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

model = FSDP(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy,
    mixed_precision=torch.bfloat16,
    device_id=torch.cuda.current_device(),
    limit_all_gathers=True,  # 限制并发通信，减少内存压力
)
```

**机构**: [Rajbhandari et al. 2020] — Microsoft Research  
**信源**: [ZeRO](https://arxiv.org/abs/1910.02054)

## 6.5 本章小结

| 主题 | 关键建议 |
|------|----------|
| 学习率调度 | 使用 Warmup，避免早期过大学习率 |
| 梯度裁剪 | 推荐 Norm Clipping，max_norm=1.0 |
| Adam 数值 | FP16 训练增大 epsilon 到 1e-4 |
| 低精度优化 | 主权重保持 FP32，或使用 8-bit 优化器 |

---

**上一章**: [第5章 算法设计层面的稳定性保障](./05-algorithm-design.md) | **下一章**: [第7章 分布式训练稳定性](./07-distributed-stability.md)
