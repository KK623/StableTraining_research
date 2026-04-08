# 第8章 调试与诊断方法论

系统化的调试方法是快速定位和解决稳定性问题的关键。

## 8.1 数值异常检测工具

### 8.1.1 NaN/Inf 监控钩子

```python
class NanInfMonitor:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.violations = []
        self.register_hooks()

    def register_hooks(self):
        def check_nan_inf(module, input, output):
            if isinstance(output, torch.Tensor):
                has_nan = torch.isnan(output).any()
                has_inf = torch.isinf(output).any()
                if has_nan or has_inf:
                    self.violations.append({
                        'module': module.__class__.__name__,
                        'has_nan': has_nan.item(),
                        'has_inf': has_inf.item()
                    })
        for module in self.model.modules():
            if len(list(module.children())) == 0:
                self.hooks.append(module.register_forward_hook(check_nan_inf))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
```

### 8.1.2 梯度范数追踪

```python
class GradientTracker:
    def __init__(self):
        self.history = []

    def track(self, model, step):
        stats = {'step': step, 'total_norm': 0.0, 'has_nan': False, 'has_inf': False}
        for name, param in model.named_parameters():
            if param.grad is not None:
                stats['total_norm'] += param.grad.norm().item() ** 2
                stats['has_nan'] |= torch.isnan(param.grad).any().item()
                stats['has_inf'] |= torch.isinf(param.grad).any().item()
        stats['total_norm'] = stats['total_norm'] ** 0.5
        self.history.append(stats)
        return stats
```

## 8.2 训练曲线解读

### 8.2.1 损失震荡的数值原因

| 曲线特征 | 可能原因 | 解决方案 |
|---------|---------|---------|
| 突然 NaN | 梯度爆炸、学习率过大 | 梯度裁剪、降低学习率 |
| 持续震荡 | 学习率过高、批次噪声 | 降低学习率、增大 batch size |
| plateau | 学习率过低、陷入平坦区域 | 增大学习率、使用学习率重启 |

### 8.2.2 梯度范数异常模式

```python
def analyze_gradient_patterns(gradient_history):
    norms = [h['total_norm'] for h in gradient_history]
    return {
        'exploding': any(n > 100 for n in norms),
        'vanishing': all(n < 1e-7 for n in norms[-10:]),
    }
```

## 8.3 最小可复现问题构建

### 8.3.1 问题隔离方法

```python
class ProblemIsolator:
    @staticmethod
    def isolate_data_issue(model, data, target):
        random_data = torch.randn_like(data)
        random_target = torch.randint(0, 10, target.shape)
        output = model(random_data)
        loss = nn.CrossEntropyLoss()(output, random_target)
        return not torch.isnan(loss).any()
```

## 8.4 推荐工具

### 8.4.1 PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(input)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### 8.4.2 Weights & Biases 监控

```python
import wandb
wandb.init(project="training-stability")
wandb.log({'loss': loss.item(), 'grad_norm': grad_norm})
```

## 8.5 本章小结

| 工具 | 用途 |
|------|------|
| NaN/Inf 监控 | 实时检测数值异常 |
| 梯度追踪 | 监控梯度范数变化 |
| PyTorch Profiler | 性能分析 |
| W&B | 实验跟踪 |

---

**上一章**: [第7章 分布式训练稳定性](./07-distributed-stability.md) | **下一章**: [第9章 实践检查清单](./09-checklists.md)
