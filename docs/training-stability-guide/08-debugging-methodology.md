# 第8章 调试与诊断方法论

系统化的调试方法是快速定位和解决稳定性问题的关键。本章提供可复现的诊断流程和工具。

## 8.1 数值异常检测工具

### 8.1.1 NaN/Inf 监控钩子

**实时监控钩子实现：**

```python
class NanInfMonitor:
    """全面的数值异常监控器
    
    监控位置：
    - 前向传播输出
    - 反向传播梯度
    - 优化器状态
    """
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
                    violation = {
                        'module': module.__class__.__name__,
                        'has_nan': has_nan.item(),
                        'has_inf': has_inf.item(),
                        'output_range': (output.min().item(), output.max().item()) if not has_nan else None
                    }
                    self.violations.append(violation)
                    # 打印即时警告
                    if has_nan:
                        print(f"[WARNING] NaN detected in {module.__class__.__name__}")
                    if has_inf:
                        print(f"[WARNING] Inf detected in {module.__class__.__name__}")
                        
        for module in self.model.modules():
            if len(list(module.children())) == 0:  # 叶子模块
                self.hooks.append(module.register_forward_hook(check_nan_inf))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def check_gradients(self):
        """检查梯度异常"""
        violations = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                has_nan = torch.isnan(param.grad).any()
                has_inf = torch.isinf(param.grad).any()
                grad_norm = param.grad.norm().item()
                
                if has_nan or has_inf or grad_norm > 1000 or (grad_norm < 1e-10 and grad_norm > 0):
                    violations.append({
                        'param': name,
                        'has_nan': has_nan.item(),
                        'has_inf': has_inf.item(),
                        'grad_norm': grad_norm
                    })
        return violations
```

### 8.1.2 梯度范数追踪

**历史趋势分析：**

```python
class GradientTracker:
    """梯度范数历史追踪器"""
    def __init__(self, max_history=10000):
        self.history = []
        self.max_history = max_history

    def track(self, model, step):
        """记录当前梯度统计信息"""
        stats = {
            'step': step,
            'total_norm': 0.0,
            'has_nan': False,
            'has_inf': False,
            'layer_norms': {}
        }
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                layer_norm = param.grad.norm().item()
                stats['total_norm'] += layer_norm ** 2
                stats['has_nan'] |= torch.isnan(param.grad).any().item()
                stats['has_inf'] |= torch.isinf(param.grad).any().item()
                stats['layer_norms'][name] = layer_norm
                
        stats['total_norm'] = stats['total_norm'] ** 0.5
        self.history.append(stats)
        
        # 限制历史长度
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            
        return stats

    def detect_anomalies(self, window=100):
        """检测异常模式"""
        if len(self.history) < window * 2:
            return None
            
        recent = self.history[-window:]
        previous = self.history[-2*window:-window]
        
        recent_norms = [h['total_norm'] for h in recent if h['total_norm'] > 0]
        previous_norms = [h['total_norm'] for h in previous if h['total_norm'] > 0]
        
        if not recent_norms or not previous_norms:
            return None
            
        avg_recent = sum(recent_norms) / len(recent_norms)
        avg_previous = sum(previous_norms) / len(previous_norms)
        
        # 检测突然变化
        if avg_recent > avg_previous * 10:
            return {'type': 'explosion', 'ratio': avg_recent / avg_previous}
        elif avg_recent < avg_previous * 0.1 and avg_previous > 1e-6:
            return {'type': 'vanishing', 'ratio': avg_recent / avg_previous}
            
        return None
```

### 8.1.3 激活值分布监控

```python
class ActivationMonitor:
    """监控激活值分布，帮助发现数值问题"""
    
    def __init__(self):
        self.activation_stats = {}
        
    def hook_fn(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activation_stats[name] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'sparsity': (output == 0).float().mean().item()
                }
        return hook
    
    def register(self, model):
        """为所有层注册监控"""
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子模块
                module.register_forward_hook(self.hook_fn(name))
    
    def check_distributions(self):
        """检查分布异常"""
        warnings = []
        for name, stats in self.activation_stats.items():
            # 检测死神经元（全0输出）
            if stats['sparsity'] > 0.9:
                warnings.append(f"{name}: {stats['sparsity']*100:.1f}% zero outputs")
            # 检测极端值
            if abs(stats['max']) > 10000 or abs(stats['min']) > 10000:
                warnings.append(f"{name}: extreme values [{stats['min']:.2f}, {stats['max']:.2f}]")
            # 检测方差坍塌
            if stats['std'] < 1e-6 and stats['mean'] != 0:
                warnings.append(f"{name}: collapsed variance (std={stats['std']:.2e})")
        return warnings
```

## 8.2 训练曲线解读

### 8.2.1 损失震荡的数值原因

| 曲线特征 | 可能原因 | 诊断方法 | 解决方案 |
|---------|---------|---------|---------|
| 突然 NaN | 梯度爆炸、学习率过大 | 检查梯度范数峰值 | 梯度裁剪、降低学习率 |
| 持续震荡 | 学习率过高、批次噪声 | 观察loss波动幅度 | 降低学习率、增大 batch size |
| 缓慢上升后 NaN | 累积数值误差 | 监控中间层激活 | 使用 BF16、增大 eps |
| plateau | 学习率过低、梯度消失 | 检查梯度范数 | 增大学习率、检查初始化 |
| 周期性震荡 | batch size 与数据分布不匹配 | 分析数据 shuffle | 调整 batch size、数据重采样 |

**损失震荡诊断代码：**

```python
def diagnose_loss_instability(loss_history, window=50):
    """诊断损失不稳定性"""
    if len(loss_history) < window:
        return "Insufficient data"
        
    recent_losses = loss_history[-window:]
    
    # 计算变异系数
    mean_loss = sum(recent_losses) / len(recent_losses)
    variance = sum((l - mean_loss) ** 2 for l in recent_losses) / len(recent_losses)
    cv = (variance ** 0.5) / mean_loss if mean_loss != 0 else float('inf')
    
    # 检测趋势
    first_half = sum(recent_losses[:window//2]) / (window//2)
    second_half = sum(recent_losses[window//2:]) / (window//2)
    
    diagnosis = {
        'coefficient_of_variation': cv,
        'trend': 'increasing' if second_half > first_half * 1.1 else 'decreasing' if second_half < first_half * 0.9 else 'stable',
        'stability': 'stable' if cv < 0.1 else 'moderate' if cv < 0.3 else 'unstable'
    }
    
    if cv > 0.5:
        diagnosis['recommendation'] = 'High instability detected. Consider reducing learning rate or increasing batch size.'
    elif second_half > first_half * 1.2:
        diagnosis['recommendation'] = 'Loss increasing. Check for gradient explosion or data issues.'
        
    return diagnosis
```

### 8.2.2 梯度范数异常模式

```python
def analyze_gradient_patterns(gradient_history):
    """分析梯度范数历史，识别异常模式"""
    norms = [h['total_norm'] for h in gradient_history if h['total_norm'] > 0]
    
    if len(norms) < 10:
        return {'status': 'insufficient_data'}
    
    analysis = {
        'max_norm': max(norms),
        'min_norm': min(norms),
        'mean_norm': sum(norms) / len(norms),
        'exploding': max(norms) > 100,
        'vanishing': all(n < 1e-7 for n in norms[-10:]),
        'increasing_trend': norms[-1] > norms[0] * 10 if len(norms) > 1 else False,
    }
    
    # 检测梯度尖峰
    mean_norm = analysis['mean_norm']
    spikes = [n for n in norms if n > mean_norm * 10]
    analysis['spike_frequency'] = len(spikes) / len(norms)
    
    if analysis['spike_frequency'] > 0.1:
        analysis['warning'] = 'Frequent gradient spikes detected. Consider gradient clipping or check for data outliers.'
        
    return analysis
```

## 8.3 最小可复现问题构建

### 8.3.1 问题隔离方法

```python
class ProblemIsolator:
    """系统化问题隔离工具"""
    
    @staticmethod
    def isolate_data_issue(model, data, target, criterion):
        """检查问题是否来自数据"""
        # 使用随机数据测试
        random_data = torch.randn_like(data)
        random_target = torch.randint(0, 10, target.shape)
        
        try:
            output = model(random_data)
            loss = criterion(output, random_target)
            has_nan = torch.isnan(loss).any()
            return {
                'is_data_issue': not has_nan,  # 随机数据正常则原数据有问题
                'random_data_loss': loss.item() if not has_nan else float('nan')
            }
        except Exception as e:
            return {'is_data_issue': False, 'error': str(e)}
    
    @staticmethod
    def isolate_model_issue(model, data, target, criterion):
        """检查问题是否来自模型结构"""
        # 使用简单线性模型对比
        simple_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(data.numel() // data.size(0), 10)
        )
        
        try:
            output = simple_model(data)
            loss = criterion(output, target)
            return {
                'is_model_issue': not torch.isnan(loss).any(),
                'simple_model_works': True
            }
        except:
            return {'is_model_issue': False, 'simple_model_works': False}
    
    @staticmethod
    def isolate_layer_issue(model, data):
        """逐层检查定位问题层"""
        x = data
        layer_outputs = []
        
        for name, module in model.named_children():
            try:
                x = module(x)
                has_nan = torch.isnan(x).any().item()
                layer_outputs.append({
                    'layer': name,
                    'output_shape': x.shape,
                    'has_nan': has_nan,
                    'output_range': (x.min().item(), x.max().item()) if not has_nan else None
                })
                if has_nan:
                    break
            except Exception as e:
                layer_outputs.append({
                    'layer': name,
                    'error': str(e)
                })
                break
                
        return layer_outputs
```

### 8.3.2 二分法定位问题

```python
def bisect_locate_nan(model, data):
    """二分法快速定位产生 NaN 的层"""
    layers = list(model.children())
    if not layers:
        return model.__class__.__name__
    
    left, right = 0, len(layers) - 1
    problem_layer = None
    
    while left <= right:
        mid = (left + right) // 2
        
        # 构建部分模型
        partial_model = nn.Sequential(*layers[:mid+1])
        
        try:
            output = partial_model(data)
            if torch.isnan(output).any():
                problem_layer = mid
                right = mid - 1  # 问题在前半段
            else:
                left = mid + 1   # 问题在后半段
        except Exception as e:
            problem_layer = mid
            break
    
    return layers[problem_layer].__class__.__name__ if problem_layer is not None else None
```

## 8.4 推荐工具

### 8.4.1 PyTorch Profiler

**性能与数值联合分析：**

```python
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

def profile_with_numeric_check(model, input_tensor):
    """性能分析同时监控数值"""
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        output = model(input_tensor)
        
        # 检查输出
        if torch.isnan(output).any():
            print("NaN detected during profiling!")
            
    # 输出统计
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # 保存详细 trace
    prof.export_chrome_trace("trace.json")
```

### 8.4.2 Weights & Biases 监控

```python
import wandb

def setup_wandb_logging(model, project="training-stability"):
    """配置 W&B 监控"""
    wandb.init(project=project)
    
    # 自动监控梯度
    wandb.watch(model, log="all", log_freq=100)
    
    return wandb

def log_training_step(wandb_run, step, loss, grad_norm, model, lr):
    """记录训练指标"""
    # 基本指标
    log_dict = {
        'train/loss': loss,
        'train/grad_norm': grad_norm,
        'train/lr': lr,
        'train/step': step
    }
    
    # 记录异常
    if torch.isnan(loss):
        log_dict['alerts/nan_loss'] = 1
    if grad_norm > 100:
        log_dict['alerts/large_grad'] = 1
        
    # 权重分布
    for name, param in model.named_parameters():
        if param.grad is not None:
            log_dict[f'gradients/{name}'] = wandb.Histogram(param.grad.cpu().numpy())
            
    wandb_run.log(log_dict)
```

### 8.4.3 TensorBoard 可视化

```python
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    """TensorBoard 数值监控"""
    
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        
    def log_histograms(self, model, step):
        """记录权重和梯度分布"""
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'weights/{name}', param, step)
            if param.grad is not None:
                self.writer.add_histogram(f'gradients/{name}', param.grad, step)
                self.writer.add_scalar(f'grad_norm/{name}', param.grad.norm(), step)
    
    def log_layer_stats(self, activation_monitor, step):
        """记录激活统计"""
        for name, stats in activation_monitor.activation_stats.items():
            self.writer.add_scalars(f'activations/{name}', {
                'mean': stats['mean'],
                'std': stats['std'],
                'sparsity': stats['sparsity']
            }, step)
```

## 8.5 调试决策树

```
训练出现问题
│
├─ 出现 NaN/Inf
│  ├─ 第一步就出现 → 检查初始化、数据预处理
│  ├─ 训练中途出现 → 检查学习率、梯度裁剪
│  └─ 特定 batch 出现 → 检查数据异常值
│
├─ 损失不下降
│  ├─ 梯度范数 ≈ 0 → 梯度消失 → 检查激活、初始化
│  ├─ 梯度范数正常 → 学习率问题 → 调整学习率/调度
│  └─ 梯度范数极大 → 梯度爆炸 → 启用裁剪
│
├─ 损失震荡
│  ├─ 高频小幅度 → batch size 太小或学习率偏高
│  └─ 低频大幅度 → 数据分布问题或学习率过高
│
└─ 精度不达预期
   ├─ 训练损失低但验证高 → 过拟合
   └─ 两者都高 → 欠拟合或数值精度问题
```

## 8.6 本章小结

| 工具 | 用途 | 使用时机 |
|------|------|---------|
| NaN/Inf 监控 | 实时检测数值异常 | 所有训练 |
| 梯度追踪 | 监控梯度范数趋势 | 训练不稳定时 |
| 激活监控 | 发现分布异常 | 调试特定层 |
| PyTorch Profiler | 性能分析 | 性能调优 |
| W&B / TensorBoard | 实验跟踪 | 长期训练 |
| 问题隔离器 | 快速定位问题源 | 出现故障时 |

**调试最佳实践：**

1. **从小规模开始**：先用小模型、小数据验证pipeline
2. **增加监控**：尽早安装 NaN/Inf 监控钩子
3. **记录一切**：保存完整的超参数、随机种子、代码版本
4. **对比基线**：保留已知稳定的配置作为对比

---

**上一章**: [第7章 分布式训练稳定性](./07-distributed-stability.md) | **下一章**: [第9章 实践检查清单](./09-checklists.md)
