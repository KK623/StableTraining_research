# 第4章 数值精度与低精度训练

本章是全文核心，系统介绍 FP32/FP16/BF16/FP8/MXFP 等精度格式的数值特性，以及混合精度训练的原理与实践。

## 4.1 浮点数值基础

### 4.1.1 IEEE 754 标准的完整数学结构

IEEE 754 标准定义了浮点数的二进制表示格式。一个浮点数由三个部分组成：

$$
\text{value} = (-1)^{s} \times 2^{e - \text{bias}} \times m
$$

其中各部分的数学定义如下：

**符号位（Sign Bit）**：
- 1 位，$s \in \{0, 1\}$
- $s = 0$ 表示正数，$s = 1$ 表示负数
- 数学上：$(-1)^s$

**指数位（Exponent Bits）**：
- $w_e$ 位，存储无符号整数 $E \in [0, 2^{w_e} - 1]$
- **偏置值（Bias）**：$\text{bias} = 2^{w_e - 1} - 1$
  - FP32（8位指数）：$\text{bias} = 127$
  - FP16（5位指数）：$\text{bias} = 15$
  - BF16（8位指数）：$\text{bias} = 127$
- 实际指数：$e = E - \text{bias}$

偏置值的作用是将有符号的指数映射到无符号整数存储，使得：
- 指数 $e = 0$ 对应存储值 $E = \text{bias}$
- 可表示的指数范围：$e \in [1 - \text{bias}, \text{bias}]$（规范化数）

**尾数位（Mantissa/Significand Bits）**：
- $w_m$ 位，存储小数部分
- **规范化数（Normalized Numbers）**：隐含前导 1
  $$
  m = 1 + \sum_{i=1}^{w_m} b_i \cdot 2^{-i} = 1.f_1 f_2 \ldots f_{w_m}
  $$
  其中 $b_i$ 是第 $i$ 位尾数的值（0 或 1）

**完整数学表示**：

$$
\text{value} = (-1)^s \times 2^{E - \text{bias}} \times \left(1 + \sum_{i=1}^{w_m} b_i \cdot 2^{-i}\right)
$$

### 4.1.2 次正规数（Subnormal/Denormalized Numbers）的深入分析

当指数位全为 0 时，表示次正规数，其数学形式不同于规范化数：

$$
\text{value}_{\text{subnormal}} = (-1)^s \times 2^{1 - \text{bias}} \times \sum_{i=1}^{w_m} b_i \cdot 2^{-i}
$$

**关键区别**：

| 特性 | 规范化数 | 次正规数 |
|------|----------|----------|
| 指数位 | $1 \leq E \leq 2^{w_e} - 2$ | $E = 0$ |
| 隐含位 | 1 | 0 |
| 最小值 | $2^{-\text{bias} + 1}$ | $2^{-\text{bias} + 1 - w_m}$ |
| 间距 | 均匀的相对间距 | 变化的绝对间距 |

**为什么次正规数计算慢（硬件原因）**：

1. **流水线停顿**：大多数浮点单元针对规范化数优化，次正规数需要特殊处理路径
2. **微码介入**：某些架构（如 x86）对次正规数操作触发微码异常处理
3. **额外周期**：次正规数乘法可能需要 10-100 倍更多周期

```python
import torch
import time

def benchmark_denormals():
    """演示次正规数对性能的影响"""
    
    # 正常范围的数值
    normal_tensor = torch.randn(1000000, dtype=torch.float32) * 1e-10
    
    # 包含次正规数的张量（小于 FP32 最小规范化数 ~1.18e-38）
    # 通过除法产生次正规数
    denormal_tensor = torch.randn(1000000, dtype=torch.float32) * 1e-10
    denormal_tensor = denormal_tensor / 1e30  # 产生次正规数
    
    # 预热
    for _ in range(100):
        _ = normal_tensor * 2.0
        _ = denormal_tensor * 2.0
    
    # 测试正常数性能
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(1000):
        result = normal_tensor * 2.0
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    normal_time = time.perf_counter() - start
    
    # 测试次正规数性能
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(1000):
        result = denormal_tensor * 2.0
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    denormal_time = time.perf_counter() - start
    
    print(f"正常数计算时间: {normal_time:.4f}s")
    print(f"次正规数计算时间: {denormal_time:.4f}s")
    print(f"性能下降倍数: {denormal_time / normal_time:.1f}x")
    
    return normal_time, denormal_time

# 禁用次正规数以提升性能
def disable_denormals():
    """
    禁用次正规数（Flush to Zero）
    小于最小规范化数的值会被置为 0
    """
    import ctypes
    # x86 架构：设置 MXCSR 寄存器的 DAZ (Denormals Are Zero) 和 FTZ (Flush To Zero) 位
    # 这在 PyTorch 中可以通过以下方式实现：
    torch.set_flush_denormal(True)
    print("次正规数已禁用（Flush to Zero 模式已启用）")

# 检测次正规数
def count_subnormals(tensor):
    """
    统计张量中次正规数的数量
    
    次正规数定义：0 < |x| < 最小规范化数
    FP32 最小规范化数 = 2^(-126) ≈ 1.17549435e-38
    """
    fp32_info = torch.finfo(torch.float32)
    # tiny 表示最小正规范化数
    subnormal_mask = (tensor.abs() > 0) & (tensor.abs() < fp32_info.tiny)
    return subnormal_mask.sum().item()
```

**禁用次正规数对数值稳定性的影响**：

1. **梯度下溢风险**：极小的梯度会被直接置零，导致某些参数不更新
2. **数值连续性破坏**：在 0 附近失去平滑过渡
3. **训练初期影响**：随机初始化的小权重可能受影响

**建议**：
- 大多数深度学习训练可以安全禁用次正规数
- 对于需要极高数值精度的科学计算应谨慎

### 4.1.3 各精度格式的误差分析

**机器 Epsilon（Machine Epsilon）**：

机器 epsilon 定义为 1.0 与下一个可表示浮点数之间的距离：

$$
\epsilon = 2^{-w_m}
$$

具体数值：
- FP32：$\epsilon = 2^{-23} \approx 1.19 \times 10^{-7}$
- FP16：$\epsilon = 2^{-10} \approx 9.77 \times 10^{-4}$
- BF16：$\epsilon = 2^{-7} \approx 7.81 \times 10^{-3}$

**舍入误差模型**：

对于浮点运算 $\circ \in \{+, -, \times, /\}$，IEEE 754 保证：

$$
\text{fl}(a \circ b) = (a \circ b) \cdot (1 + \delta), \quad |\delta| \leq \epsilon
$$

其中 $\text{fl}(\cdot)$ 表示浮点运算结果，$\delta$ 是舍入误差。

**相对误差与绝对误差界限**：

对于数值 $x$ 的浮点表示 $\hat{x}$：

$$
\hat{x} = x \cdot (1 + \delta), \quad |\delta| \leq \epsilon
$$

相对误差：$\left|\frac{\hat{x} - x}{x}\right| \leq \epsilon$

绝对误差：$|\hat{x} - x| \leq \epsilon |x|$

**舍入误差累积的数学模型**：

考虑 $n$ 个数的求和 $S = \sum_{i=1}^{n} x_i$，递归求和的误差分析：

$$
\hat{S}_n = \sum_{i=1}^{n} x_i \prod_{j=i}^{n-1} (1 + \delta_j), \quad |\delta_j| \leq \epsilon
$$

误差上界：

$$
\left|\frac{\hat{S}_n - S_n}{S_n}\right| \leq (n-1)\epsilon \cdot \frac{\sum_{i=1}^{n} |x_i|}{\left|\sum_{i=1}^{n} x_i\right|}
$$

**条件数（Condition Number）**：

问题的数值稳定性与条件数相关：

$$
\kappa = \frac{|x \cdot f'(x)|}{|f(x)|}
$$

条件数大表示问题是病态的，小的输入误差会被放大。

```python
import torch
import numpy as np

def analyze_precision_formats():
    """分析不同精度格式的数值特性"""
    
    formats = {
        'FP32': torch.float32,
        'FP16': torch.float16,
        'BF16': torch.bfloat16
    }
    
    for name, dtype in formats.items():
        info = torch.finfo(dtype)
        print(f"\n{name}:")
        print(f"  指数位数: {info.bits - info.nmant - 1}")
        print(f"  尾数位数: {info.nmant}")
        print(f"  机器 epsilon: {info.eps:.2e}")
        print(f"  最大可表示值: {info.max:.2e}")
        print(f"  最小规范化正数: {info.tiny:.2e}")
        print(f"  动态范围（数量级）: {np.log10(info.max / info.tiny):.1f}")

def demonstrate_rounding_error():
    """演示舍入误差累积"""
    
    # 理论：n 个数求和的误差上界约为 n * epsilon
    n = 10000
    epsilon_fp32 = torch.finfo(torch.float32).eps
    epsilon_fp16 = torch.finfo(torch.float16).eps
    
    # 生成测试数据：大量小数值求和
    small_values = torch.ones(n, dtype=torch.float32) * 0.0001
    
    # FP32 求和
    sum_fp32 = small_values.sum()
    
    # FP16 求和
    sum_fp16 = small_values.to(torch.float16).sum()
    
    # 理论误差界限
    theoretical_error_fp32 = n * epsilon_fp32 * sum_fp32
    theoretical_error_fp16 = n * epsilon_fp16 * sum_fp16
    
    print(f"FP32 求和结果: {sum_fp32:.6f}")
    print(f"FP16 求和结果: {sum_fp16:.6f}")
    print(f"理论误差界限 (FP32): ±{theoretical_error_fp32:.6e}")
    print(f"理论误差界限 (FP16): ±{theoretical_error_fp16:.6e}")
    
    # Kahan 求和算法（补偿求和）
    def kahan_sum(arr):
        """Kahan 补偿求和算法，减少舍入误差"""
        sum_val = 0.0
        compensation = 0.0  # 补偿项
        
        for x in arr:
            y = x - compensation  # 减去之前丢失的低阶位
            t = sum_val + y       # 求和
            compensation = (t - sum_val) - y  # 计算本次丢失的部分
            sum_val = t
        
        return sum_val
    
    sum_kahan = kahan_sum(small_values.tolist())
    print(f"Kahan 求和结果: {sum_kahan:.6f}")
```

### 4.1.4 各精度格式对比

| 格式 | 指数位 | 尾数位 | 偏置值 | 动态范围 | 机器 epsilon | 最小规范化数 | 典型用途 |
|------|--------|--------|--------|----------|--------------|--------------|----------|
| FP32 | 8 | 23 | 127 | $\sim 1.7 \times 10^{38}$ | $\sim 1.19 \times 10^{-7}$ | $1.18 \times 10^{-38}$ | 主权重、损失计算 |
| FP16 | 5 | 10 | 15 | $6.55 \times 10^{4}$ | $\sim 9.77 \times 10^{-4}$ | $6.10 \times 10^{-5}$ | 前向/反向计算 |
| BF16 | 8 | 7 | 127 | $\sim 1.7 \times 10^{38}$ | $\sim 7.81 \times 10^{-3}$ | $1.18 \times 10^{-38}$ | 训练默认格式 |
| E4M3 | 4 | 3 | 7 | 448 | $\sim 0.125$ | $2^{-6} = 0.0156$ | FP8 前向传播 |
| E5M2 | 5 | 2 | 15 | 57,344 | $\sim 0.25$ | $2^{-14} \approx 6.1 \times 10^{-5}$ | FP8 梯度计算 |

**信源**: IEEE 754 标准 — IEEE (电气电子工程师学会)

## 4.2 混合精度训练机制

### 4.2.1 GradScaler 的数学原理

混合精度训练的核心问题是**梯度下溢（Gradient Underflow）**：当梯度值小于 FP16 可表示的最小值时，会被置为零，导致参数无法更新。

**损失缩放因子的选择理论**：

设损失缩放因子为 $\lambda$，则：

$$
\tilde{L} = \lambda \cdot L
$$

反向传播时：

$$
\tilde{g} = \frac{\partial \tilde{L}}{\partial w} = \lambda \cdot \frac{\partial L}{\partial w} = \lambda \cdot g
$$

权重更新：

$$
w_{t+1} = w_t - \eta \cdot \frac{\tilde{g}}{\lambda} = w_t - \eta \cdot g
$$

**梯度下溢的数学条件**：

梯度下溢发生的条件：

$$
|g| < \frac{\text{FP16}_{\min}}{\lambda}
$$

其中 $\text{FP16}_{\min} \approx 6 \times 10^{-5}$ 是 FP16 最小规范化正数。

为防止下溢，需要：

$$
\lambda > \frac{\text{FP16}_{\min}}{|g_{\min}|}
$$

其中 $|g_{\min}|$ 是模型中最小梯度的绝对值。

**动态缩放的更新策略推导**：

动态损失缩放根据梯度是否溢出调整 $\lambda$：

1. **溢出检测**：检查梯度中是否存在 Inf 或 NaN
   - 若 $\exists g_i : |g_i| > \frac{\text{FP16}_{\max}}{\lambda}$，则发生上溢

2. **缩放因子更新规则**：
   $$
   \lambda_{t+1} = \begin{cases}
   \lambda_t \cdot \text{factor} & \text{if 无溢出} \\
   \max(\lambda_t / \text{factor}, \lambda_{\min}) & \text{if 溢出}
   \end{cases}
   $$

   其中 factor 通常为 2，$\lambda_{\min}$ 防止缩放过小。

3. **间隔控制**：每 $N$ 步检查一次，避免频繁调整

```python
import torch
from torch.cuda.amp import autocast, GradScaler

class GradScalerMath:
    """
    GradScaler 的数学原理实现
    展示损失缩放的核心计算逻辑
    """
    
    def __init__(self, init_scale=2.**16, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000, enabled=True):
        """
        Args:
            init_scale: 初始损失缩放因子 λ
            growth_factor: 无溢出时的增长因子
            backoff_factor: 溢出时的回退因子
            growth_interval: 连续无溢出步数后增长缩放因子
            enabled: 是否启用损失缩放
        """
        self._scale = torch.tensor(init_scale, dtype=torch.float32)
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._enabled = enabled
        
        # 计数器
        self._growth_tracker = 0
        self._per_optimizer_states = {}
    
    def scale(self, loss):
        """
        应用损失缩放: L̃ = λ · L
        
        数学原理：
        1. 将损失乘以缩放因子，使梯度放大 λ 倍
        2. 放大后的梯度在 FP16 范围内可表示
        3. 权重更新时再除以 λ 恢复正确比例
        """
        if not self._enabled:
            return loss
        
        # 损失缩放：将损失乘以缩放因子
        return loss * self._scale.to(loss.device, loss.dtype)
    
    def unscale_(self, optimizer):
        """
        反缩放梯度：g = ḡ / λ
        
        在裁剪梯度前需要先反缩放，使梯度回到正确比例
        """
        if not self._enabled:
            return
        
        # 对每个参数组的梯度进行反缩放
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    # 反缩放：除以缩放因子
                    param.grad.data.div_(self._scale.to(param.grad.device))
    
    def step(self, optimizer):
        """
        执行优化器步进，检查梯度是否溢出
        
        溢出检测条件：
        - 梯度包含 Inf 或 NaN
        - 数学上：|g| > FP16_max / λ 时会发生溢出
        """
        if not self._enabled:
            optimizer.step()
            return
        
        # 检查梯度是否包含 Inf 或 NaN
        found_inf = self._check_grads_for_overflow(optimizer)
        
        if found_inf:
            # 溢出：跳过权重更新，回退缩放因子
            self._update_scale(False)
            # 清零梯度
            optimizer.zero_grad()
        else:
            # 无溢出：正常更新权重
            optimizer.step()
            self._update_scale(True)
    
    def _check_grads_for_overflow(self, optimizer):
        """检查梯度是否溢出（包含 Inf 或 NaN）"""
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                        return True
        return False
    
    def _update_scale(self, no_overflow):
        """
        更新损失缩放因子
        
        数学规则：
        - 无溢出：增长计数器，达到间隔后 λ ← λ × growth_factor
        - 溢出：λ ← λ × backoff_factor，重置计数器
        """
        if no_overflow:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                # 连续无溢出，增加缩放因子
                self._scale *= self._growth_factor
                self._growth_tracker = 0
        else:
            # 溢出，回退缩放因子
            self._scale = torch.max(
                self._scale * self._backoff_factor,
                torch.tensor(1.0)  # 最小缩放因子
            )
            self._growth_tracker = 0
    
    def get_scale(self):
        return self._scale.item()


# PyTorch 官方 GradScaler 使用示例
def mixed_precision_training_example(model, dataloader, optimizer, criterion):
    """
    混合精度训练完整示例
    """
    scaler = GradScaler()
    
    for data, target in dataloader:
        optimizer.zero_grad()
        
        # autocast 上下文：自动在适当操作使用 FP16/BF16
        with autocast(dtype=torch.float16):  # 或 torch.bfloat16
            output = model(data)
            loss = criterion(output, target)
        
        # scale(loss)：应用损失缩放
        # backward()：在缩放后的损失上计算梯度
        scaler.scale(loss).backward()
        
        # step()：内部处理梯度反缩放和溢出检查
        scaler.step(optimizer)
        
        # update()：更新缩放因子
        scaler.update()
```

### 4.2.2 数值误差传播分析

**前向传播的精度损失**：

考虑一层线性变换 $y = Wx + b$，在 FP16 中计算：

$$
\hat{y} = \text{fl}(\text{fl}(W \cdot x) + b)
$$

误差分析：

$$
\hat{y}_i = \sum_{j} W_{ij} x_j (1 + \delta_{ij}) + b_i(1 + \delta_i), \quad |\delta| \leq \epsilon_{\text{FP16}}
$$

累积误差：

$$
|\hat{y}_i - y_i| \leq \epsilon_{\text{FP16}} \left( \sum_{j} |W_{ij} x_j| + |b_i| \right)
$$

**反向传播的梯度误差累积**：

对于损失函数 $L$，梯度计算：

$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot x^T
$$

在 FP16 中的误差：

$$
\left|\widehat{\frac{\partial L}{\partial W}} - \frac{\partial L}{\partial W}\right| \leq \epsilon_{\text{FP16}} \cdot \left|\frac{\partial L}{\partial y}\right| \cdot |x|^T
$$

**权重更新的精度保持**：

FP32 主权重更新：

$$
W_{\text{FP32}}^{(t+1)} = W_{\text{FP32}}^{(t)} - \eta \cdot \frac{\partial L}{\partial W_{\text{FP16}}}
$$

关键：使用 FP32 存储主权重，避免小梯度更新的精度损失。

```python
def analyze_gradient_flow(model, sample_input):
    """
    分析梯度流动和数值精度
    """
    model.train()
    
    # 前向传播
    with autocast(dtype=torch.float16):
        output = model(sample_input)
        loss = output.mean()
    
    # 反向传播
    scaler = GradScaler()
    scaler.scale(loss).backward()
    
    # 分析各层梯度统计
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach()
            stats = {
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'min': grad.min().item(),
                'max': grad.max().item(),
                'fp16_min': torch.finfo(torch.float16).tiny,
                'underflow_ratio': (grad.abs() < torch.finfo(torch.float16).tiny).float().mean().item()
            }
            grad_stats[name] = stats
            
            print(f"{name}:")
            print(f"  范围: [{stats['min']:.2e}, {stats['max']:.2e}]")
            print(f"  下溢比例: {stats['underflow_ratio']:.2%}")
    
    return grad_stats

def check_gradient_underflow(model):
    """
    检测模型中是否存在严重的梯度下溢问题
    
    返回：
        underflow_ratio: 下溢梯度元素的比例
        problematic_layers: 存在严重下溢的层名列表
    """
    underflow_count = 0
    total_params = 0
    fp16_min = torch.finfo(torch.float16).tiny  # ~6.1e-5
    
    problematic_layers = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            layer_total = param.grad.numel()
            layer_underflow = (param.grad.abs() < fp16_min).sum().item()
            
            underflow_count += layer_underflow
            total_params += layer_total
            
            layer_ratio = layer_underflow / layer_total
            if layer_ratio > 0.01:  # 超过 1% 的下溢
                problematic_layers.append((name, layer_ratio))
    
    underflow_ratio = underflow_count / total_params if total_params > 0 else 0
    
    return {
        'underflow_ratio': underflow_ratio,
        'problematic_layers': problematic_layers,
        'needs_loss_scaling': underflow_ratio > 0.01
    }
```

**机构**: [Micikevicius et al. 2018] — NVIDIA, Baidu Research, UC Berkeley  
**信源**: [Mixed Precision Training](https://arxiv.org/abs/1710.03740)

## 4.3 BF16 vs FP16 的数学对比

### 4.3.1 详细对比的数学基础

**动态范围计算（指数位数影响）**：

动态范围由指数位数决定。对于 $w_e$ 位指数：

$$
\text{Dynamic Range} = 20 \cdot \log_{10}\left(\frac{\text{max}}{\text{min}}\right) \text{ dB}
$$

或表示为数量级：

$$
\text{Orders of Magnitude} = \log_{10}(2^{2 \cdot (2^{w_e-1} - 1) - 1}) \approx 0.6 \cdot (2^{w_e} - 3)
$$

具体计算：

| 格式 | 指数位 | 最大指数 | 最小指数 | 动态范围（数量级） |
|------|--------|----------|----------|-------------------|
| FP16 | 5 | $2^{4} - 1 = 15$ | $1 - 15 = -14$ | $\approx 10^{-5}$ 到 $10^{4}$ |
| BF16 | 8 | $2^{7} - 1 = 127$ | $1 - 127 = -126$ | $\approx 10^{-38}$ 到 $10^{38}$ |

BF16 的动态范围与 FP32 相同，因为它使用相同的 8 位指数。

**精度计算（尾数位数影响）**：

相对精度由尾数位数决定：

$$
\text{Relative Precision} = 2^{-w_m}
$$

有效十进制位数：

$$
\text{Decimal Digits} = w_m \cdot \log_{10}(2) \approx 0.301 \cdot w_m
$$

| 格式 | 尾数位数 | 相对精度 | 有效十进制位数 |
|------|----------|----------|----------------|
| FP32 | 23 | $\approx 1.2 \times 10^{-7}$ | 6-9 位 |
| FP16 | 10 | $\approx 9.8 \times 10^{-4}$ | 3-4 位 |
| BF16 | 7 | $\approx 7.8 \times 10^{-3}$ | 2-3 位 |

**舍入误差对比**：

对于数值 $x$ 的舍入：

$$
\hat{x} = x \cdot (1 + \delta), \quad |\delta| \leq \epsilon = 2^{-w_m-1}
$$

| 格式 | 最大舍入误差 | 典型场景误差 |
|------|-------------|-------------|
| FP32 | $\pm 0.5 \times 10^{-7}$ | 可忽略 |
| FP16 | $\pm 0.5 \times 10^{-3}$ | 需注意累积 |
| BF16 | $\pm 0.5 \times 10^{-2}$ | 大数值计算稳定 |

### 4.3.2 场景化选择的理论依据

**选择 BF16 的场景**：

1. **梯度范围大**：当梯度跨越多个数量级时
   - 数学条件：$\max(|g|) / \min(|g|) > 10^4$
   
2. **避免损失缩放**：BF16 的动态范围足够大，通常不需要损失缩放
   - 简化训练流程，减少超参数调优

3. **大规模训练**：LLM 训练中激活值和梯度范围变化大

**选择 FP16 的场景**：

1. **精度敏感任务**：需要更高精度的科学计算
   - 数学条件：要求的相对误差 $< 10^{-3}$

2. **旧硬件支持**：V100 等不支持 BF16

3. **显存受限**：FP16 的精度损失可接受时

```python
def compare_bf16_fp16_numerics():
    """
    对比 BF16 和 FP16 的数值特性
    """
    # 创建测试数据
    x = torch.linspace(-100, 100, 1000)
    
    # 转换为不同精度
    x_fp32 = x.to(torch.float32)
    x_fp16 = x.to(torch.float16)
    x_bf16 = x.to(torch.bfloat16)
    
    # 计算精度损失
    error_fp16 = (x_fp16.to(torch.float32) - x_fp32).abs()
    error_bf16 = (x_bf16.to(torch.float32) - x_fp32).abs()
    
    print("精度对比:")
    print(f"FP16 最大绝对误差: {error_fp16.max():.6f}")
    print(f"BF16 最大绝对误差: {error_bf16.max():.6f}")
    print(f"FP16 平均相对误差: {(error_fp16 / x_fp32.abs()).mean():.6%}")
    print(f"BF16 平均相对误差: {(error_bf16 / x_fp32.abs()).mean():.6%}")
    
    # 动态范围对比
    print("\n动态范围对比:")
    info_fp16 = torch.finfo(torch.float16)
    info_bf16 = torch.finfo(torch.bfloat16)
    
    print(f"FP16: [{info_fp16.min:.2e}, {info_fp16.max:.2e}]")
    print(f"BF16: [{info_bf16.min:.2e}, {info_bf16.max:.2e}]")
    
    # 小数值表示能力
    small_values = torch.tensor([1e-10, 1e-20, 1e-30, 1e-40], dtype=torch.float32)
    print(f"\n小数值表示:")
    print(f"原始值: {small_values}")
    print(f"FP16: {small_values.to(torch.float16)}")
    print(f"BF16: {small_values.to(torch.bfloat16)}")

def recommend_precision_format(model_info):
    """
    根据模型信息推荐精度格式
    
    Args:
        model_info: 包含以下字段的字典
            - hardware: 硬件类型 ('V100', 'A100', 'H100', 'TPU')
            - model_depth: 模型深度
            - task_type: 任务类型 ('nlp', 'vision', 'scientific')
    """
    hardware = model_info.get('hardware', 'A100')
    task_type = model_info.get('task_type', 'nlp')
    
    # 硬件支持检查
    bf16_supported = hardware in ['A100', 'H100', 'TPU', 'MI200']
    
    if not bf16_supported:
        return {
            'recommendation': 'FP16',
            'reason': f'{hardware} 不支持 BF16',
            'needs_loss_scaling': True
        }
    
    # 任务类型分析
    if task_type == 'scientific':
        return {
            'recommendation': 'FP16',
            'reason': '科学计算需要更高精度',
            'needs_loss_scaling': True
        }
    
    # 默认推荐 BF16
    return {
        'recommendation': 'BF16',
        'reason': '动态范围大，通常不需要损失缩放',
        'needs_loss_scaling': False
    }
```

### 4.3.3 硬件支持矩阵

| 硬件 | FP16 TensorCore | BF16 | FP8 | MXFP8 |
|------|-----------------|------|-----|-------|
| V100 | ✅ | ❌ | ❌ | ❌ |
| A100 | ✅ | ✅ | ❌ | ❌ |
| H100 | ✅ | ✅ | ✅ | ❌ |
| B200 | ✅ | ✅ | ✅ | ✅ |

**信源**: [BFloat16: The secret to high performance on Cloud TPUs](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus) — Google Brain

## 4.4 FP8 训练前沿

### 4.4.1 E4M3 和 E5M2 的数值特性

FP8 格式有两种变体，针对深度学习不同阶段的特性设计：

**E4M3 格式（4 位指数，3 位尾数）**：

数学表示：
$$
\text{value} = (-1)^s \times 2^{E - 7} \times (1.m_1 m_2 m_3)_2
$$

数值特性计算：
- 偏置值：$\text{bias} = 2^{4-1} - 1 = 7$
- 最大可表示值：$\approx 2^{8} \times (2 - 2^{-3}) = 448$
- 最小规范化正数：$2^{-6} = 0.015625$
- 机器 epsilon：$2^{-3} = 0.125$（约 1 位有效数字）

**E5M2 格式（5 位指数，2 位尾数）**：

数学表示：
$$
\text{value} = (-1)^s \times 2^{E - 15} \times (1.m_1 m_2)_2
$$

数值特性计算：
- 偏置值：$\text{bias} = 2^{5-1} - 1 = 15$
- 最大可表示值：$\approx 2^{16} \times (2 - 2^{-2}) = 57344$
- 最小规范化正数：$2^{-14} \approx 6.1 \times 10^{-5}$
- 机器 epsilon：$2^{-2} = 0.25$（约 0.5 位有效数字）

**为什么 E4M3 用于前向、E5M2 用于反向**：

| 特性 | 前向传播（E4M3） | 反向传播（E5M2） |
|------|-----------------|-----------------|
| 主要需求 | 精度 | 动态范围 |
| 数值特点 | 激活值范围相对集中 | 梯度范围变化大 |
| 格式选择 | 更多尾数位（3位） | 更多指数位（5位） |
| 可表示范围 | 小但足够（±448） | 大（±57344） |

数学依据：
1. **前向传播**：激活值经过 LayerNorm 等操作后范围相对可控，需要更多精度
2. **反向传播**：梯度可能因链式法则累积而变得极小或极大，需要更大动态范围

### 4.4.2 缩放因子管理的数学

FP8 训练需要精细的缩放因子管理：

**缩放因子计算**：

对于张量 $X$，缩放因子 $s$ 的计算：

$$
s = \frac{\text{max}_{\text{representable}}}{\max(|X|) \cdot \text{margin}}
$$

其中 margin（通常为 1.25）为安全余量，防止溢出。

量化过程：

$$
X_{\text{quantized}} = \text{round}\left(\frac{X \cdot s}{\text{max}_{\text{representable}}} \times 127\right)
$$

反量化：

$$
\hat{X} = X_{\text{quantized}} \times \frac{\text{max}_{\text{representable}}}{s \times 127}
$$

```python
import torch
import transformer_engine.pytorch as te

class FP8Scaler:
    """
    FP8 缩放因子管理的数学实现
    """
    
    def __init__(self):
        self.forward_scale = 1.0   # E4M3 缩放因子
        self.backward_scale = 1.0  # E5M2 缩放因子
        self.history = {'forward': [], 'backward': []}
    
    def compute_scaling_factor(self, tensor, format='e4m3', margin=1.25):
        """
        计算最优缩放因子
        
        数学推导：
        1. 找到张量绝对值的最大值: max_val = max(|tensor|)
        2. 目标：将 max_val 映射到可表示范围的 margin 处
        3. 缩放因子：scale = max_representable / (max_val * margin)
        
        Args:
            tensor: 输入张量
            format: 'e4m3' 或 'e5m2'
            margin: 安全余量，防止溢出
        
        Returns:
            scale: 缩放因子
        """
        max_val = tensor.abs().max().item()
        
        # E4M3 和 E5M2 的最大可表示值
        max_representable = 448.0 if format == 'e4m3' else 57344.0
        
        if max_val == 0:
            return 1.0
        
        # 计算缩放因子
        scale = max_representable / (max_val * margin)
        
        return scale
    
    def quantize_to_fp8(self, tensor, scale, format='e4m3'):
        """
        将张量量化到 FP8
        
        量化公式：
        q = round(tensor * scale / max_repr * 127)
        
        Args:
            tensor: 输入张量（FP32/FP16）
            scale: 缩放因子
            format: 'e4m3' 或 'e5m2'
        
        Returns:
            quantized: 量化后的 FP8 张量
            scale: 使用的缩放因子（用于反量化）
        """
        max_representable = 448.0 if format == 'e4m3' else 57344.0
        
        # 应用缩放并量化
        scaled = tensor * scale
        
        # 限制到可表示范围
        scaled = torch.clamp(scaled, -max_representable, max_representable)
        
        # 转换为 FP8（PyTorch 2.1+ 支持）
        if format == 'e4m3':
            fp8_dtype = torch.float8_e4m3fn
        else:
            fp8_dtype = torch.float8_e5m2
        
        quantized = scaled.to(fp8_dtype)
        
        return quantized, scale
    
    def update_scale_history(self, forward_scale=None, backward_scale=None):
        """更新缩放因子历史，用于动态调整"""
        if forward_scale is not None:
            self.history['forward'].append(forward_scale)
            # 保持最近 100 个历史值
            self.history['forward'] = self.history['forward'][-100:]
        
        if backward_scale is not None:
            self.history['backward'].append(backward_scale)
            self.history['backward'] = self.history['backward'][-100:]
    
    def get_smoothed_scale(self, window=10):
        """
        获取平滑后的缩放因子
        
        使用移动平均减少缩放因子的抖动
        """
        if len(self.history['forward']) >= window:
            forward_smooth = sum(self.history['forward'][-window:]) / window
        else:
            forward_smooth = self.forward_scale
        
        if len(self.history['backward']) >= window:
            backward_smooth = sum(self.history['backward'][-window:]) / window
        else:
            backward_smooth = self.backward_scale
        
        return forward_smooth, backward_smooth


# Transformer Engine FP8 训练示例
def fp8_training_with_transformer_engine():
    """
    使用 Transformer Engine 进行 FP8 训练
    """
    # 创建 FP8 线性层
    layer = te.Linear(768, 3072)
    
    # 在 FP8 自动转换上下文中执行
    with te.fp8_autocast():
        # 前向传播自动使用 E4M3
        output = layer(input_tensor)
        
        # 反向传播自动使用 E5M2
        loss = criterion(output, target)
        loss.backward()
    
    # Transformer Engine 自动管理缩放因子
```

**信源**: [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) — NVIDIA, Arm, Intel, Qualcomm

## 4.5 微缩放（Microscaling/MXFP）技术

### 4.5.1 MXFP8 的数学原理

MXFP8（Microscaling FP8）通过引入两级缩放机制，在保持硬件效率的同时提高量化精度。

**块级缩放的误差分析**：

传统 per-tensor 量化的误差：

$$
\text{MSE}_{\text{tensor}} = \frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{x}_i)^2
$$

其中所有元素共享同一个缩放因子，受离群值影响大。

MXFP8 块级量化的误差：

将张量划分为 $B$ 个块，每块 $K$ 个元素：

$$
\text{MSE}_{\text{block}} = \frac{1}{B} \sum_{b=1}^{B} \frac{1}{K} \sum_{i \in \text{block}_b} (x_i - \hat{x}_i^{(b)})^2
$$

每块有独立的缩放因子 $s_b$，更好地适应局部数据分布。

**两级缩放的数学优势**：

MXFP8 采用两级缩放：

$$
\hat{x}_i = \text{dequantize}(\text{quantize}(x_i \cdot s_b \cdot S))
$$

其中：
- $S$：全局缩放因子（per-tensor）
- $s_b$：局部块缩放因子（per-block）
- 块内使用 E4M3 格式

误差分析：

设原始值 $x_i$ 在块 $b$ 中，块均值为 $\mu_b$，标准差为 $\sigma_b$。

per-tensor 量化的信噪比：

$$
\text{SNR}_{\text{tensor}} \approx \frac{\sigma^2}{\sigma^2 + \frac{\Delta^2}{12}}
$$

per-block 量化的信噪比：

$$
\text{SNR}_{\text{block}} \approx \frac{1}{B} \sum_{b=1}^{B} \frac{\sigma_b^2}{\sigma_b^2 + \frac{\Delta_b^2}{12}}
$$

当数据分布不均匀时，$\text{SNR}_{\text{block}} \gg \text{SNR}_{\text{tensor}}$。

```python
class MXFP8Quantizer:
    """
    MXFP8 微缩放量化器的数学实现
    """
    
    def __init__(self, block_size=128):
        """
        Args:
            block_size: 块大小，通常是 128 或 256
        """
        self.block_size = block_size
    
    def compute_block_scales(self, tensor):
        """
        计算每个块的缩放因子
        
        数学原理：
        对于每个块 b，缩放因子 s_b = max_repr / max(|x_b|)
        
        Args:
            tensor: 输入张量，形状为 [..., D]
        
        Returns:
            scales: 块级缩放因子，形状为 [..., D/block_size]
        """
        # 将最后一维分成块
        original_shape = tensor.shape
        D = original_shape[-1]
        num_blocks = D // self.block_size
        
        # reshape 为 [..., num_blocks, block_size]
        tensor_reshaped = tensor.reshape(*original_shape[:-1], num_blocks, self.block_size)
        
        # 计算每块的最大绝对值
        block_max = tensor_reshaped.abs().max(dim=-1, keepdim=True)[0]
        
        # E4M3 最大可表示值
        max_representable = 448.0
        
        # 计算缩放因子（添加小值防止除零）
        scales = max_representable / (block_max + 1e-10)
        
        return scales.squeeze(-1)
    
    def quantize(self, tensor, scales):
        """
        应用 MXFP8 量化
        
        量化过程：
        1. 扩展 scales 到与 tensor 相同形状
        2. 应用块级缩放
        3. 量化到 E4M3
        
        Args:
            tensor: 输入张量
            scales: 块级缩放因子
        
        Returns:
            quantized: 量化后的 FP8 张量
            scales: 缩放因子（用于反量化）
        """
        original_shape = tensor.shape
        D = original_shape[-1]
        num_blocks = D // self.block_size
        
        # reshape
        tensor_reshaped = tensor.reshape(*original_shape[:-1], num_blocks, self.block_size)
        
        # 扩展 scales 用于广播
        scales_expanded = scales.unsqueeze(-1)
        
        # 应用缩放
        scaled = tensor_reshaped * scales_expanded
        
        # 限制范围并量化
        scaled = torch.clamp(scaled, -448.0, 448.0)
        quantized = scaled.to(torch.float8_e4m3fn)
        
        # 恢复形状
        quantized = quantized.reshape(original_shape)
        
        return quantized, scales
    
    def dequantize(self, quantized, scales):
        """
        反量化
        
        Args:
            quantized: 量化后的 FP8 张量
            scales: 块级缩放因子
        
        Returns:
            dequantized: 反量化后的张量
        """
        original_shape = quantized.shape
        D = original_shape[-1]
        num_blocks = D // self.block_size
        
        # reshape
        quantized_reshaped = quantized.reshape(*original_shape[:-1], num_blocks, self.block_size)
        scales_expanded = scales.unsqueeze(-1)
        
        # 反量化：除以缩放因子
        dequantized = quantized_reshaped.to(torch.float32) / scales_expanded
        
        return dequantized.reshape(original_shape)


def analyze_mxfp8_vs_per_tensor():
    """
    对比 MXFP8 块级量化与 per-tensor 量化的误差
    """
    # 创建具有离群值的测试数据
    torch.manual_seed(42)
    tensor = torch.randn(1024, 1024)
    # 添加离群值
    tensor[0, :] *= 100  # 第一行有较大值
    
    # Per-tensor 量化
    tensor_max = tensor.abs().max()
    scale_tensor = 448.0 / tensor_max
    quantized_tensor = (tensor * scale_tensor).clamp(-448, 448).to(torch.float8_e4m3fn)
    dequantized_tensor = quantized_tensor.to(torch.float32) / scale_tensor
    error_tensor = (tensor - dequantized_tensor).pow(2).mean()
    
    # MXFP8 块级量化
    mxfp8 = MXFP8Quantizer(block_size=128)
    scales_block = mxfp8.compute_block_scales(tensor)
    quantized_block, _ = mxfp8.quantize(tensor, scales_block)
    dequantized_block = mxfp8.dequantize(quantized_block, scales_block)
    error_block = (tensor - dequantized_block).pow(2).mean()
    
    print(f"Per-tensor 量化 MSE: {error_tensor:.6f}")
    print(f"MXFP8 块级量化 MSE: {error_block:.6f}")
    print(f"误差改善: {error_tensor / error_block:.2f}x")
```

### 4.5.2 DeepSeek-V3 细粒度量化策略的理论基础

DeepSeek-V3 采用了细粒度的 FP8 量化策略，结合了 tile-wise 和 block-wise 量化的优势。

**激活量化（1×128 tile-wise）的数学原理**：

对于激活 $A \in \mathbb{R}^{B \times D}$（批次大小 × 隐藏维度）：

$$
A_{\text{quant}}[b, d] = \text{round}\left(\frac{A[b, d]}{s_{b, \lfloor d/128 \rfloor}} \times 448\right)
$$

其中 $s_{b, t}$ 是第 $b$ 个样本、第 $t$ 个 tile 的缩放因子：

$$
s_{b, t} = \frac{\max_{d \in \text{tile}_t} |A[b, d]|}{448}
$$

数学优势：
1. **离群值隔离**：每个 token 的离群值只影响其所在 tile
2. **局部适应性**：不同 tile 可根据自身分布选择最优缩放

**权重量化（128×128 block-wise）的数学原理**：

对于权重 $W \in \mathbb{R}^{D_{\text{out}} \times D_{\text{in}}}$：

$$
W_{\text{quant}}[i, j] = \text{round}\left(\frac{W[i, j]}{s_{\lfloor i/128 \rfloor, \lfloor j/128 \rfloor}} \times 448\right)
$$

块级缩放因子：

$$
s_{bi, bj} = \frac{\max_{i \in \text{block}_{bi}, j \in \text{block}_{bj}} |W[i, j]|}{448}
$$

**在线计算缩放的数学优势**：

传统方法使用历史统计量：

$$
s_t = \alpha \cdot s_{t-1} + (1 - \alpha) \cdot \frac{\max(|X_t|)}{448}
$$

在线计算方法：

$$
s_t = \frac{\max(|X_t|)}{448}
$$

优势分析：
1. **无滞后**：立即响应当前数据分布
2. **无超参数**：无需调整平滑系数 $\alpha$
3. **离群值鲁棒**：不会被历史离群值持续影响

```python
class DeepSeekFP8Linear(nn.Module):
    """
    DeepSeek-V3 风格的 FP8 线性层实现
    
    数学要点：
    1. 激活：1×128 tile-wise 量化
    2. 权重：128×128 block-wise 量化
    3. 在线计算缩放因子
    """
    
    def __init__(self, in_features, out_features, block_size=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        
        # 权重量化为 FP8 E4M3
        self.register_buffer('weight', torch.empty(
            out_features, in_features, dtype=torch.float8_e4m3fn
        ))
        
        # 权重缩放因子：每个 128×128 块一个
        self.weight_scale = nn.Parameter(torch.ones(
            out_features // block_size, 
            in_features // block_size
        ))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化权重并计算初始缩放因子"""
        # FP32 初始化
        weight_fp32 = torch.empty(self.out_features, self.in_features)
        nn.init.kaiming_uniform_(weight_fp32, a=math.sqrt(5))
        
        # 计算块级缩放并量化
        self.weight_scale.data = self._compute_weight_scales(weight_fp32)
        self.weight = self._quantize_weight(weight_fp32, self.weight_scale)
    
    def _compute_weight_scales(self, weight):
        """
        计算权重的块级缩放因子
        
        Args:
            weight: FP32 权重，形状 [out_features, in_features]
        
        Returns:
            scales: 缩放因子，形状 [out_features/block_size, in_features/block_size]
        """
        out_blocks = self.out_features // self.block_size
        in_blocks = self.in_features // self.block_size
        
        # reshape 为 [out_blocks, block_size, in_blocks, block_size]
        w_reshaped = weight.reshape(
            out_blocks, self.block_size,
            in_blocks, self.block_size
        ).permute(0, 2, 1, 3)  # [out_blocks, in_blocks, block_size, block_size]
        
        # 每块的最大绝对值
        block_max = w_reshaped.abs().max(dim=(2, 3))[0]  # [out_blocks, in_blocks]
        
        # E4M3 最大可表示值为 448
        scales = block_max / 448.0 + 1e-10
        
        return scales
    
    def _quantize_weight(self, weight, scales):
        """量化权重到 FP8"""
        out_blocks = self.out_features // self.block_size
        in_blocks = self.in_features // self.block_size
        
        w_reshaped = weight.reshape(
            out_blocks, self.block_size,
            in_blocks, self.block_size
        ).permute(0, 2, 1, 3)
        
        # 应用缩放
        scales_expanded = scales.unsqueeze(2).unsqueeze(3)
        w_scaled = w_reshaped / scales_expanded
        
        # 量化
        w_quant = w_scaled.clamp(-448, 448).to(torch.float8_e4m3fn)
        
        # 恢复形状
        w_quant = w_quant.permute(0, 2, 1, 3).reshape(
            self.out_features, self.in_features
        )
        
        return w_quant
    
    def quantize_activation(self, x, tile_size=128):
        """
        激活的 tile-wise 量化（1×128）
        
        Args:
            x: 输入激活，形状 [batch, seq_len, hidden_dim]
            tile_size: tile 大小，默认 128
        
        Returns:
            x_fp8: 量化后的 FP8 激活
            x_scale: tile 级缩放因子
        """
        original_shape = x.shape
        hidden_dim = original_shape[-1]
        num_tiles = hidden_dim // tile_size
        
        # reshape 为 [..., num_tiles, tile_size]
        x_reshaped = x.reshape(*original_shape[:-1], num_tiles, tile_size)
        
        # 计算每 tile 的最大绝对值（在线计算）
        tile_max = x_reshaped.abs().max(dim=-1, keepdim=True)[0]
        
        # 缩放因子
        scale = tile_max / 448.0 + 1e-10
        
        # 量化
        x_scaled = x_reshaped / scale
        x_quantized = x_scaled.clamp(-448, 448).to(torch.float8_e4m3fn)
        
        # 恢复形状
        x_fp8 = x_quantized.reshape(original_shape)
        
        return x_fp8, scale.squeeze(-1)
    
    def forward(self, x):
        """
        前向传播
        
        使用 torch._scaled_mm 进行 FP8 矩阵乘法
        """
        # 量化激活
        x_fp8, x_scale = self.quantize_activation(x)
        
        # FP8 矩阵乘法（需要 PyTorch 2.1+）
        # output = torch._scaled_mm(x_fp8, self.weight.t(), 
        #                          scale_a=x_scale, scale_b=self.weight_scale)
        
        # 简化版本：反量化后计算
        x_dequant = x_fp8.to(torch.float32) * x_scale.unsqueeze(-1)
        
        # 反量化权重
        w_dequant = self.weight.to(torch.float32) * self.weight_scale.repeat_interleave(
            self.block_size, dim=0
        ).repeat_interleave(self.block_size, dim=1)
        
        output = torch.matmul(x_dequant, w_dequant.t())
        
        return output
```

**机构**: DeepSeek (幻方量化)  
**信源**: [DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v1#S5)

**信源**: [Rouhani et al. 2023] — Microsoft Research  
**信源**: [Microscaling Formats for Deep Learning](https://arxiv.org/abs/2310.10537)

## 4.6 量化感知训练（QAT）稳定性

### 4.6.1 伪量化的数学原理

量化感知训练通过在训练时模拟量化效果，使模型适应低精度推理。

**均匀量化的数学定义**：

对于 $b$ 位有符号整数量化，量化步长为：

$$
\Delta = \frac{\max(|x|) - \min(|x|)}{2^b - 1}
$$

量化函数：

$$
Q(x) = \text{round}\left(\frac{x}{\Delta}\right) \cdot \Delta
$$

**伪量化（Fake Quantization）**：

前向传播时模拟量化/反量化：

$$
y = Q(x) = \text{round}\left(\text{clip}\left(\frac{x}{\Delta}, -2^{b-1}, 2^{b-1}-1\right)\right) \cdot \Delta
$$

反向传播时，梯度通过 STE 传播。

**量化噪声分析**：

量化引入的误差可建模为均匀分布噪声：

$$
\epsilon = Q(x) - x, \quad \epsilon \sim U\left(-\frac{\Delta}{2}, \frac{\Delta}{2}\right)
$$

量化噪声方差：

$$
\text{Var}(\epsilon) = \frac{\Delta^2}{12}
$$

### 4.6.2 直通估计器（STE）的收敛性分析

**STE 的数学定义**：

由于量化函数 $Q(x)$ 的导数几乎处处为 0，需要近似梯度：

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial Q} \cdot \frac{\partial Q}{\partial x}
$$

STE 近似：

$$
\frac{\partial Q}{\partial x} \approx 1
$$

即：

$$
\frac{\partial L}{\partial x} \approx \frac{\partial L}{\partial Q}
$$

**收敛性分析**：

STE 引入的梯度偏差：

$$
\mathbb{E}\left[\frac{\partial L}{\partial x} - \frac{\partial L}{\partial Q}\right] = \mathbb{E}\left[\frac{\partial L}{\partial Q} \cdot \left(\frac{\partial Q}{\partial x} - 1\right)\right]
$$

当量化步长 $\Delta \to 0$ 时，STE 偏差趋于 0。

**改进的 STE 变体**：

1. **Clip STE**：考虑裁剪边界
   $$
   \frac{\partial Q}{\partial x} = \mathbf{1}_{|x| \leq \text{threshold}}
   $$

2. **Sigmoid STE**：使用 sigmoid 近似
   $$
   \frac{\partial Q}{\partial x} = \sigma'(x)
   $$

```python
class FakeQuantize(nn.Module):
    """
    伪量化模块的数学实现
    
    数学原理：
    1. 前向：模拟量化-反量化过程
    2. 反向：使用 STE 传播梯度
    """
    
    def __init__(self, num_bits=8, symmetric=True):
        super().__init__()
        self.num_bits = num_bits
        self.symmetric = symmetric
        
        # 量化范围
        if symmetric:
            self.qmin = -(2 ** (num_bits - 1))
            self.qmax = 2 ** (num_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** num_bits - 1
        
        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))
    
    def forward(self, x):
        if not self.training:
            # 推理时直接量化
            return self._quantize(x)
        
        # 训练时：伪量化 + STE
        # 前向：量化后反量化
        x_quant = self._quantize(x)
        x_dequant = self._dequantize(x_quant)
        
        # STE：前向用反量化值，反向传原始梯度
        # 数学表达：output = x_dequant + stop_gradient(x - x_dequant)
        return x_dequant + (x - x_dequant).detach()
    
    def _quantize(self, x):
        """量化到整数范围"""
        if self.symmetric:
            x_scaled = x / self.scale
        else:
            x_scaled = x / self.scale + self.zero_point
        
        x_clipped = torch.clamp(x_scaled, self.qmin, self.qmax)
        x_rounded = torch.round(x_clipped)
        
        return x_rounded
    
    def _dequantize(self, x_quant):
        """反量化回浮点"""
        if self.symmetric:
            return x_quant * self.scale
        else:
            return (x_quant - self.zero_point) * self.scale
    
    def compute_scale(self, x, method='max'):
        """
        计算最优缩放因子
        
        Args:
            x: 输入张量
            method: 'max' 或 'mse'
        
        Returns:
            scale: 计算得到的缩放因子
        """
        if method == 'max':
            # 基于最大值的缩放
            x_max = x.abs().max()
            scale = x_max / self.qmax
        elif method == 'mse':
            # 基于 MSE 最优的缩放
            # 通过网格搜索找到最小化 MSE 的缩放因子
            x_max = x.abs().max()
            candidate_scales = torch.linspace(0.1, 1.0, 100) * x_max / self.qmax
            
            best_scale = candidate_scales[0]
            best_mse = float('inf')
            
            for s in candidate_scales:
                x_quant = torch.round(torch.clamp(x / s, self.qmin, self.qmax))
                x_dequant = x_quant * s
                mse = (x - x_dequant).pow(2).mean()
                
                if mse < best_mse:
                    best_mse = mse
                    best_scale = s
            
            scale = best_scale
        
        return scale


class StraightThroughEstimator(torch.autograd.Function):
    """
    自定义 STE 实现
    
    数学原理：
    前向：y = quantize(x)
    反向：∂L/∂x = ∂L/∂y（忽略量化函数的零梯度）
    """
    
    @staticmethod
    def forward(ctx, x, scale, qmin, qmax):
        # 保存用于反向传播
        ctx.save_for_backward(x)
        ctx.scale = scale
        ctx.qmin = qmin
        ctx.qmax = qmax
        
        # 量化
        x_scaled = x / scale
        x_clipped = torch.clamp(x_scaled, qmin, qmax)
        x_rounded = torch.round(x_clipped)
        
        return x_rounded * scale
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        scale = ctx.scale
        qmin, qmax = ctx.qmin, ctx.qmax
        
        # STE：梯度直接传递
        # 可选：考虑裁剪边界
        grad_input = grad_output.clone()
        
        # 在裁剪边界外梯度为 0（Clip STE）
        x_scaled = x / scale
        mask = (x_scaled >= qmin) & (x_scaled <= qmax)
        grad_input = grad_input * mask.float()
        
        return grad_input, None, None, None


def quantization_aware_training_step(model, data, target, criterion, optimizer, num_bits=8):
    """
    QAT 训练步骤
    
    数学流程：
    1. 在前向传播中插入伪量化节点
    2. 使用 STE 计算梯度
    3. 更新 FP32 权重
    """
    optimizer.zero_grad()
    
    # 前向传播（自动应用伪量化）
    output = model(data)
    loss = criterion(output, target)
    
    # 反向传播（STE 自动处理）
    loss.backward()
    
    # 更新权重
    optimizer.step()
    
    return loss.item()
```

**数值稳定性要点**：
- 缩放因子需要足够大以覆盖张量范围，但不能过大导致精度损失
- 训练初期使用较大的 batch size 稳定统计量
- 逐步降低精度（FP32 → FP16 → INT8）进行渐进式量化

**与低精度训练的区别**：
- QAT 主要用于推理优化，训练时仍使用较高精度
- FP8/BF16 训练是真正的低精度训练，需要处理数值稳定性问题

## 4.7 本章小结

| 主题 | 关键要点 |
|------|----------|
| IEEE 754 结构 | 符号位、指数位（含偏置）、尾数位；规范化数与次正规数 |
| 误差分析 | 机器 epsilon $\epsilon = 2^{-w_m}$；舍入误差累积模型 |
| 次正规数 | 硬件计算慢，可禁用但需注意梯度下溢 |
| GradScaler | 损失缩放防止梯度下溢；动态调整策略 |
| BF16 vs FP16 | BF16 动态范围大（同 FP32），FP16 精度高；现代硬件首选 BF16 |
| FP8 格式 | E4M3 用于前向（精度优先），E5M2 用于反向（范围优先） |
| 微缩放 | MXFP8 块级缩放减少量化误差；DeepSeek-V3 细粒度策略 |
| QAT | 伪量化模拟推理；STE 实现梯度传播 |

---

**上一章**: [第3章 模型架构稳定性](./03-architecture-stability.md) | **下一章**: [第5章 算法设计层面的稳定性保障](./05-algorithm-design.md)
