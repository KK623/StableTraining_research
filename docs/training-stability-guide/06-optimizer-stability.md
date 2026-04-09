# 第6章 优化器稳定性

优化器的选择和配置直接影响训练的数值稳定性，特别是在低精度环境下。本章深入分析主流优化器的数值特性，从理论层面解释其稳定性机制。

## 6.1 Adam优化器的数值分析

### 6.1.1 二阶矩估计的偏差分析

Adam优化器维护两个指数移动平均（EMA）状态：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

其中 $g_t = \nabla_\theta \mathcal{L}(\theta_t)$ 是时间步 $t$ 的梯度。

**偏差问题的数学根源：**

由于初始化 $m_0 = 0, v_0 = 0$，早期估计存在系统性偏差：

$$E[m_t] = E\left[\sum_{i=1}^{t} (1-\beta_1)\beta_1^{t-i} g_i\right] = (1-\beta_1^t) E[g_t] \neq E[g_t]$$

同理：

$$E[v_t] = (1-\beta_2^t) E[g_t^2] \neq E[g_t^2]$$

### 6.1.2 偏差修正的推导

为获得无偏估计，需进行偏差修正：

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

**修正后的无偏性证明：**

$$E[\hat{m}_t] = E\left[\frac{\sum_{i=1}^{t} (1-\beta_1)\beta_1^{t-i} g_i}{1-\beta_1^t}\right] = \frac{1-\beta_1}{1-\beta_1^t} \sum_{i=1}^{t} \beta_1^{t-i} E[g_i]$$

假设梯度平稳 $E[g_i] = E[g]$：

$$E[\hat{m}_t] = \frac{1-\beta_1}{1-\beta_1^t} \cdot \frac{1-\beta_1^t}{1-\beta_1} \cdot E[g] = E[g]$$

**Warmup与偏差修正的协同效应：**

在训练初期（$t$ 较小），$\beta_1^t \approx 1$，偏差修正因子 $\frac{1}{1-\beta_1^t}$ 很大，放大了早期不稳定的梯度统计量。这就是为什么Warmup在Adam中尤为关键——它通过限制学习率来抵消偏差修正带来的放大效应。

### 6.1.3 Epsilon参数对数值稳定性的影响

Adam的更新公式为：

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Epsilon的数值作用：**

1. **防止除零**：当 $\hat{v}_t \to 0$ 时，避免数值溢出
2. **控制更新幅值上限**：$\left|\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}\right| \leq \frac{|\hat{m}_t|}{\epsilon}$

**FP16训练需要更大Epsilon的理论依据：**

FP16的表示范围为 $[5.96 \times 10^{-8}, 65504]$，精度为 $2^{-10} \approx 9.77 \times 10^{-4}$。

当使用 $\epsilon = 1\times 10^{-8}$ 时：
- 在FP32中：$\sqrt{\hat{v}_t} + 10^{-8}$ 可精确表示
- 在FP16中：$10^{-8}$ 下溢为0，失去保护作用

**推荐的Epsilon配置：**

| 精度 | 推荐Epsilon | 理论依据 |
|------|-------------|----------|
| FP32 | $1\times 10^{-8}$ | 远小于典型梯度方差 |
| TF32 | $1\times 10^{-7}$ | 10-bit尾数精度限制 |
| FP16 | $1\times 10^{-4}$ ~ $1\times 10^{-3}$ | 避免下溢，匹配精度 |
| BF16 | $1\times 10^{-8}$ ~ $1\times 10^{-7}$ | 与FP32相同指数范围 |

```python
# 标准 Adam (FP32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)

# FP16 训练建议增大 epsilon
optimizer_fp16 = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-4)

# BF16 训练可使用标准 epsilon
optimizer_bf16 = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)
```

**信源**: [Kingma & Ba 2014] — University of Amsterdam, Google  
**信源**: [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

## 6.2 Lion优化器的深入分析

### 6.2.1 符号更新的理论依据

Lion (EvoLved Sign Momentum) 的核心更新规则：

$$c_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$\theta_t = \theta_{t-1} - \eta \cdot \text{sign}(c_t)$$
$$m_t = \beta_2 m_{t-1} + (1-\beta_2) g_t$$

**与SignedSGD的理论联系：**

SignedSGD的更新为 $\theta_{t+1} = \theta_t - \eta \cdot \text{sign}(g_t)$。Lion可视为SignedSGD的动量版本：

$$\text{sign}(\beta_1 m_{t-1} + (1-\beta_1) g_t) = \text{sign}\left(\frac{\beta_1}{1-\beta_1} m_{t-1} + g_t\right)$$

当 $\beta_1 = 0.9$ 时，历史梯度权重是当期梯度的9倍，形成强平滑效果。

### 6.2.2 为什么Lion不需要Epsilon

**数学分析：**

Lion的更新仅依赖符号函数：

$$\text{sign}(x) = \begin{cases} +1 & x > 0 \\ 0 & x = 0 \\ -1 & x < 0 \end{cases}$$

符号函数对输入缩放不敏感：$\text{sign}(\alpha x) = \text{sign}(x)$ 对于 $\alpha > 0$。这意味着：

1. **无需除法归一化**：不像Adam需要 $\frac{m}{\sqrt{v}}$ 来自适应调整学习率
2. **无除零风险**：没有分母，自然不需要epsilon保护
3. **对梯度幅值不敏感**：只关心方向，不关心大小

### 6.2.3 分布式训练中的优势

**2-bit传输的理论基础：**

Lion的更新 $\text{sign}(c_t) \in \{-1, 0, +1\}$ 可用2-bit编码：
- 00: 0 (不更新)
- 01: +1
- 10: -1
- 11: 保留/未使用

相比Adam的FP32更新（32-bit），通信量降低16×。

**量化友好的深层原因：**

Adam的更新 $\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$ 是连续值，量化会引入误差：

$$\text{Quantize}\left(\frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}\right) = \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon} + \delta$$

而Lion的符号更新天然离散，量化无损失：

$$\text{Quantize}(\text{sign}(c_t)) = \text{sign}(c_t)$$

### 6.2.4 收敛性分析（与Adam比较）

**收敛速率：**

对于凸函数，Adam的收敛速率为 $O(1/\sqrt{T})$，Lion同样达到 $O(1/\sqrt{T})$。

**泛化性能差异：**

Lion的符号更新引入了隐式梯度裁剪：

$$\|\theta_{t+1} - \theta_t\|_\infty = \eta$$

每个参数的最大更新量被严格限制为学习率，这起到了类似于梯度裁剪的正则化效果。

**与Adam的数值对比：**

| 特性 | Adam | Lion |
|-----|------|------|
| 二阶矩估计 | 需要 | 不需要 |
| 内存占用 | 2×参数 | 1×参数 |
| 更新值域 | 连续值 | {-1, 0, +1} |
| 对异常值敏感 | 是 | 否 |
| epsilon 敏感 | 是 | 否 |
| 通信带宽 | 32-bit/参数 | 2-bit/参数 |
| 低精度稳定性 | 需调整epsilon | 天然稳定 |

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

## 6.3 权重衰减的数值行为

### 6.3.1 L2正则化 vs 解耦权重衰减

**L2正则化（传统Adam）：**

目标函数：$\mathcal{L}'(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2}\|\theta\|^2$

梯度：$\nabla_\theta \mathcal{L}' = \nabla_\theta \mathcal{L} + \lambda \theta$

代入Adam更新：

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t + \lambda \theta_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**数值问题**：权重衰减被自适应学习率除，导致大梯度参数衰减慢，小梯度参数衰减快。

**解耦权重衰减（AdamW）：**

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_t$$

**数学差异分析：**

设参数 $i$ 的更新为：

- L2正则化：$\Delta \theta_i^{(L2)} = -\eta \frac{m_i + \lambda \theta_i}{\sqrt{v_i} + \epsilon}$
- AdamW：$\Delta \theta_i^{(W)} = -\eta \left(\frac{m_i}{\sqrt{v_i} + \epsilon} + \lambda \theta_i\right)$

差异：

$$\Delta \theta_i^{(W)} - \Delta \theta_i^{(L2)} = -\eta \lambda \theta_i \left(1 - \frac{1}{\sqrt{v_i} + \epsilon}\right)$$

当 $\sqrt{v_i} \gg 1$ 时，L2正则化的有效衰减被抑制；AdamW保持恒定的衰减率。

### 6.3.2 AdamW的更新公式推导

完整AdamW算法：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_t = \theta_{t-1} - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right)$$

### 6.3.3 低精度下的数值问题

**FP16中的权重衰减问题：**

当参数 $\theta$ 较小时，FP16的精度限制导致：

$$\text{FP16}(\theta - \eta \lambda \theta) = \text{FP16}(\theta(1 - \eta \lambda))$$

若 $\eta \lambda < 2^{-10} \approx 9.77 \times 10^{-4}$，在FP16中 $1 - \eta \lambda = 1$，权重衰减失效。

**解决方案：**

```python
# 主权重保持FP32
master_weight = param.float()
master_weight = master_weight - lr * weight_decay * master_weight
param.copy_(master_weight)
```

```python
# AdamW 实现
optimizer_adamw = torch.optim.AdamW(
    model.parameters(), 
    lr=1e-3, 
    eps=1e-8, 
    weight_decay=0.01
)
```

**信源**: [Loshchilov & Hutter 2017] — University of Freiburg  
**信源**: [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)

## 6.4 8-bit优化器的深入分析

### 6.4.1 分块量化的理论基础

8-bit Adam将优化器状态从FP32量化到8-bit，核心挑战是保持数值稳定性。

**分块量化的数学原理：**

将张量分块为大小为 $B$ 的块，每块独立量化：

$$Q(x) = \text{round}\left(\frac{x - z}{s}\right) \cdot s + z$$

其中：
- $s = \frac{\max(x) - \min(x)}{2^8 - 1}$ 是缩放因子
- $z = \min(x)$ 是零点

**为什么选择128个元素一块：**

1. **统计代表性**：128个样本足以估计分布的极值
2. **内存对齐**：128×4字节 = 512字节，符合GPU缓存行
3. **量化误差控制**：块越大，极值差异越大，量化误差越大；块越小，元数据开销越高

**量化误差分析：**

对于均匀分布 $x \sim U[a, b]$，块大小为 $B$ 的量化误差：

$$E[|x - Q(x)|^2] \approx \frac{(b-a)^2}{12 \cdot 256^2}$$

分块后，每块的动态范围 $(b-a)$ 减小，误差降低。

### 6.4.2 量化误差对优化过程的影响

**动量项的量化误差累积：**

$$m_t^{\text{quant}} = \beta_2 m_{t-1}^{\text{quant}} + (1-\beta_2) g_t + \delta_t$$

其中 $\delta_t$ 是量化误差。误差累积：

$$m_t^{\text{quant}} = \sum_{i=1}^{t} (1-\beta_2)\beta_2^{t-i} g_i + \sum_{i=1}^{t} \beta_2^{t-i} \delta_i$$

由于 $|\delta_i| \leq \frac{s}{2}$，累积误差有界：

$$\left|\sum_{i=1}^{t} \beta_2^{t-i} \delta_i\right| \leq \frac{s}{2(1-\beta_2)}$$

### 6.4.3 与FP32优化器的收敛性比较

**理论保证：**

在适当条件下，8-bit Adam保持与FP32 Adam相同的收敛速率，但常数因子增大：

$$\mathbb{E}[\|\nabla \mathcal{L}(\theta_T)\|^2] \leq O\left(\frac{1}{\sqrt{T}}\right) + O(\delta^2)$$

其中 $\delta$ 是最大量化误差。

**实践建议：**

```python
import bitsandbytes as bnb

# 8-bit Adam
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-3)

# 8-bit AdamW
optimizer = bnb.optim.AdamW8bit(
    model.parameters(), 
    lr=1e-3,
    optim_bits=8,
    block_size=128  # 默认分块大小
)
```

**信源**: [Dettmers et al. 2022] — University of Washington, Meta AI  
**信源**: [8-bit Optimizers](https://arxiv.org/abs/2110.02861)

## 6.5 学习率调度的数值影响

### 6.5.1 Warmup的数学原理

**线性Warmup：**

$$\eta_t = \eta_{max} \times \min\left(\frac{t}{T_w}, 1.0\right)$$

其中 $T_w$ 是warmup步数。

**余弦Warmup（更平滑的过渡）：**

$$\eta_t = \eta_{max} \times \frac{1}{2}\left(1 - \cos\left(\frac{\pi \cdot t}{T_w}\right)\right), \quad t \leq T_w$$

**为什么Warmup能防止早期梯度爆炸：**

初始化后的梯度分布：$g \sim \mathcal{N}(0, \sigma^2 I)$，其中 $\sigma$ 较大。

参数更新：$\theta_1 = \theta_0 - \eta g_0$

参数变化范数：$\|\theta_1 - \theta_0\| = \eta \|g_0\|$

使用Warmup，早期 $\eta_t \ll \eta_{max}$，限制参数变化：

$$\|\theta_t - \theta_0\| \leq \sum_{i=1}^{t} \eta_i \|g_i\| \approx \eta_{max} \frac{t^2}{2T_w} \bar{g}$$

相比无Warmup的线性增长 $t \cdot \eta_{max} \cdot \bar{g}$，Warmup将早期累积变化降低为二次增长。

### 6.5.2 学习率突变对优化器状态的影响

**自适应优化器的状态耦合：**

Adam的二阶矩 $v_t$ 对学习率敏感：

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) (\eta_t \cdot g_t)^2$$

当学习率突变时，梯度统计量需要重新适应。

**余弦退火的数学推导：**

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{\pi \cdot (t - T_w)}{T - T_w}\right)\right)$$

其中 $T$ 是总训练步数。

**平滑性分析：**

余弦退火的一阶导数连续：

$$\frac{d\eta}{dt} = -\frac{\pi(\eta_{max} - \eta_{min})}{2(T-T_w)} \sin\left(\frac{\pi(t-T_w)}{T-T_w}\right)$$

在 $t=T_w$ 和 $t=T$ 处导数为0，实现平滑过渡。

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

## 6.6 梯度裁剪策略

### 6.6.1 Norm Clipping vs Value Clipping

```python
# Norm Clipping（推荐）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Value Clipping
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

**Norm Clipping的数学原理：**

设梯度向量 $g$，裁剪阈值 $\tau$：

$$g_{\text{clip}} = \begin{cases} g & \|g\| \leq \tau \\ \tau \frac{g}{\|g\|} & \|g\| > \tau \end{cases}$$

保持梯度方向，限制幅值。

### 6.6.2 自适应梯度裁剪

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

## 6.7 低精度优化器状态

### 6.7.1 优化器状态分片（ZeRO）

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

## 6.8 本章小结

| 主题 | 关键建议 |
|------|----------|
| Adam数值分析 | 理解偏差修正的必要性，FP16训练增大epsilon到1e-4 |
| Lion优化器 | 利用符号更新的数值稳定性，适合低精度和分布式训练 |
| 权重衰减 | 使用AdamW的解耦权重衰减，低精度保持主权重FP32 |
| 8-bit优化器 | 分块量化平衡精度与内存，128元素块为经验最优 |
| 学习率调度 | Warmup防止早期不稳定，余弦退火实现平滑衰减 |
| 梯度裁剪 | 推荐Norm Clipping，max_norm=1.0 |
| 低精度优化 | 主权重保持FP32，或使用8-bit优化器 |

---

**上一章**: [第5章 算法设计层面的稳定性保障](./05-algorithm-design.md) | **下一章**: [第7章 分布式训练稳定性](./07-distributed-stability.md)
