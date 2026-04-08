# 第6章 优化器稳定性

优化器的选择和配置直接影响训练的数值稳定性，特别是在低精度环境下。

## 6.1 学习率调度与数值稳定性

### 6.1.1 Warmup 的必要性

```python
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

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

**信源**: [Chen et al. 2023] — Google Brain

## 6.4 低精度优化器状态

### 6.4.1 8-bit Adam 实践

```python
import bitsandbytes as bnb
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-3)
```

**机构**: [Dettmers et al. 2022] — University of Washington, Meta AI  
**信源**: [8-bit Optimizers](https://arxiv.org/abs/2110.02861)

### 6.4.2 优化器状态分片（ZeRO）

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
