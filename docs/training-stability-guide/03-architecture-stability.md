# 第3章 模型架构稳定性

模型架构设计决定了梯度的流动方式和数值的分布特性。本章覆盖初始化、归一化、残差连接、激活函数等关键组件的稳定性考量。

## 3.1 初始化策略

### 3.1.1 Xavier/Glorot 初始化

**适用场景**: Tanh、Sigmoid 等对称激活函数

**数学原理**：

$$
W_{ij} \sim U\left[-\frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}\right]
$$

#### 3.1.1.1 方差保持的数学条件

Xavier/Glorot 初始化的核心目标是保持前向传播和后向传播中信号方差的一致性。考虑一个全连接层：

$$
y_i = \sum_{j=1}^{n_{in}} W_{ij} x_j
$$

假设输入 $x_j$ 独立同分布，均值为0，方差为 $\text{Var}(x)$；权重 $W_{ij}$ 独立同分布，均值为0，方差为 $\text{Var}(W)$。

**前向传播方差分析**：

输出 $y_i$ 的方差为：

$$
\text{Var}(y_i) = \text{Var}\left(\sum_{j=1}^{n_{in}} W_{ij} x_j\right) = \sum_{j=1}^{n_{in}} \text{Var}(W_{ij} x_j)
$$

由于 $W$ 和 $x$ 独立：

$$
\text{Var}(y_i) = \sum_{j=1}^{n_{in}} \text{Var}(W_{ij}) \cdot \text{Var}(x_j) = n_{in} \cdot \text{Var}(W) \cdot \text{Var}(x)
$$

为了保持方差不变（$\text{Var}(y) = \text{Var}(x)$），需要：

$$
n_{in} \cdot \text{Var}(W) = 1 \implies \text{Var}(W) = \frac{1}{n_{in}}
$$

**后向传播方差分析**：

梯度反向传播时，对于损失函数 $L$，输入梯度为：

$$
\frac{\partial L}{\partial x_j} = \sum_{i=1}^{n_{out}} W_{ij} \frac{\partial L}{\partial y_i}
$$

类似地，梯度方差为：

$$
\text{Var}\left(\frac{\partial L}{\partial x}\right) = n_{out} \cdot \text{Var}(W) \cdot \text{Var}\left(\frac{\partial L}{\partial y}\right)
$$

为了保持梯度方差不变，需要：

$$
n_{out} \cdot \text{Var}(W) = 1 \implies \text{Var}(W) = \frac{1}{n_{out}}
$$

#### 3.1.1.2 前向和后向传播的方差分析

前向传播要求 $\text{Var}(W) = \frac{1}{n_{in}}$，而后向传播要求 $\text{Var}(W) = \frac{1}{n_{out}}$。这两个条件通常无法同时满足。

Xavier/Glorot 初始化采用折中方案，取两者的调和平均：

$$
\text{Var}(W) = \frac{2}{n_{in} + n_{out}}
$$

#### 3.1.1.3 为什么分母是 $\sqrt{n_{in} + n_{out}}$

对于均匀分布 $U[-a, a]$，方差为 $\frac{a^2}{3}$。令：

$$
\frac{a^2}{3} = \frac{2}{n_{in} + n_{out}}
$$

解得：

$$
a = \sqrt{\frac{6}{n_{in} + n_{out}}} = \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}
$$

对于正态分布初始化，标准差为：

$$
\sigma = \sqrt{\frac{2}{n_{in} + n_{out}}}
$$

```python
import torch
import torch.nn as nn
import math

def xavier_uniform_manual(tensor, gain=1.0):
    """
    手动实现 Xavier/Glorot 均匀初始化
    
    参数:
        tensor: 待初始化的权重张量
        gain: 缩放因子，用于调整激活函数的影响
    
    数学推导:
        fan_in = n_in (输入维度)
        fan_out = n_out (输出维度)
        limit = sqrt(6 / (fan_in + fan_out)) * gain
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    
    # 计算均匀分布的边界
    # Var(W) = 2 / (n_in + n_out)
    # 对于 U[-a, a]: Var = a^2 / 3
    # 因此: a = sqrt(6 / (n_in + n_out))
    limit = gain * math.sqrt(6.0 / (fan_in + fan_out))
    
    with torch.no_grad():
        return tensor.uniform_(-limit, limit)

def xavier_normal_manual(tensor, gain=1.0):
    """
    手动实现 Xavier/Glorot 正态初始化
    
    标准差: sigma = gain * sqrt(2 / (n_in + n_out))
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    
    # 正态分布的标准差
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    
    with torch.no_grad():
        return tensor.normal_(0, std)

# PyTorch 内置实现
layer = nn.Linear(784, 256)
nn.init.xavier_uniform_(layer.weight)
nn.init.zeros_(layer.bias)
```

**机构**: [Glorot & Bengio 2010] — University of Toronto  
**信源**: [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html)

### 3.1.2 Kaiming/He 初始化

**适用场景**: ReLU 及其变体

#### 3.1.2.1 考虑 ReLU 对方差的影响

ReLU 激活函数定义为 $\text{ReLU}(x) = \max(0, x)$。对于以0为中心对称分布的输入，ReLU 会将大约一半的神经元输出置为0。

考虑一个经过 ReLU 的层：

$$
y = \text{ReLU}(Wx)
$$

对于前向传播，假设 $z = Wx$ 服从均值为0的对称分布，则：

$$
\mathbb{E}[y] = \mathbb{E}[\max(0, z)] = \frac{1}{2} \mathbb{E}[|z|]
$$

$$
\mathbb{E}[y^2] = \frac{1}{2} \mathbb{E}[z^2] = \frac{1}{2} \text{Var}(z)
$$

因此，经过 ReLU 后的方差约为输入方差的一半。

#### 3.1.2.2 考虑负值截断的修正因子

为了保持方差不变，需要在初始化时补偿 ReLU 造成的方差损失。设修正因子为 $c$：

$$
\text{Var}(y) = c \cdot n_{in} \cdot \text{Var}(W) \cdot \text{Var}(x)
$$

对于 ReLU，$c = \frac{1}{2}$（因为大约一半输出为0）。因此：

$$
\frac{1}{2} \cdot n_{in} \cdot \text{Var}(W) = 1 \implies \text{Var}(W) = \frac{2}{n_{in}}
$$

对于正态分布初始化：

$$
W_{ij} \sim N\left(0, \frac{2}{n_{in}}\right)
$$

对于均匀分布初始化：

$$
W_{ij} \sim U\left[-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right]
$$

```python
def kaiming_normal_manual(tensor, mode='fan_in', nonlinearity='relu'):
    """
    手动实现 Kaiming/He 正态初始化
    
    参数:
        mode: 'fan_in' 保持前向传播方差，'fan_out' 保持反向传播方差
        nonlinearity: 激活函数类型，影响修正因子
    
    数学推导:
        对于 ReLU: 修正因子 a = sqrt(2)
        对于 leaky_relu: 修正因子 a = sqrt(2 / (1 + negative_slope^2))
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    
    # 选择计算模式
    if mode == 'fan_in':
        fan = fan_in
    elif mode == 'fan_out':
        fan = fan_out
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    # 计算修正因子
    if nonlinearity == 'relu':
        # ReLU 将负值置为0，方差减半
        # 因此需要放大 sqrt(2) 倍来补偿
        gain = math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        negative_slope = 0.01  # 默认 leaky_relu 斜率
        gain = math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        gain = 1.0
    
    # 标准差计算
    # Var(W) = gain^2 / fan
    std = gain / math.sqrt(fan)
    
    with torch.no_grad():
        return tensor.normal_(0, std)

# PyTorch 内置实现
layer = nn.Linear(784, 256)
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

**机构**: [He et al. 2015] — Microsoft Research Asia  
**信源**: [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852)

### 3.1.3 Fixup 初始化

**适用场景**: 极深网络（100+ 层）无需归一化层

#### 3.1.3.1 残差分支缩放的理论

Fixup 初始化的核心思想是确保在训练初期，残差分支的贡献接近于零，从而避免梯度爆炸或消失。

考虑一个具有 $L$ 个残差块的深度网络：

$$
x_{l+1} = x_l + F_l(x_l, W_l)
$$

其中 $F_l$ 是第 $l$ 个残差分支。在初始化时，如果 $F_l$ 的输出方差与 $x_l$ 相当，则经过 $L$ 层后，输出方差可能放大 $L$ 倍：

$$
\text{Var}(x_L) \approx (1 + \alpha)^L \cdot \text{Var}(x_0)
$$

Fixup 通过对残差分支进行缩放来解决这个问题。设缩放因子为 $S_l$：

$$
x_{l+1} = x_l + S_l \cdot F_l(x_l, W_l)
$$

Fixup 建议的缩放策略：

$$
S_l = L^{-1/(2m-2)}
$$

其中 $m$ 是残差分支中的层数（通常为2或3）。

#### 3.1.3.2 初始化时的零贡献保证

Fixup 初始化包含以下关键组件：

1. **残差分支最后一层的零初始化**：将每个残差分支的最后一层权重和偏置初始化为零
2 **输入缩放**：对网络输入进行适当缩放
3. **偏置项**：添加可学习的偏置项来调整激活分布

数学上，设第 $l$ 个残差分支为：

$$
F_l(x) = W_l^{(2)} \phi(W_l^{(1)} x + b_l^{(1)}) + b_l^{(2)}
$$

Fixup 初始化设置 $W_l^{(2)} = 0$ 和 $b_l^{(2)} = 0$，确保初始时 $F_l(x) = 0$。

```python
class FixupLayer(nn.Module):
    """
    Fixup 初始化层实现
    
    核心思想:
    1. 残差分支初始贡献为0
    2. 通过可学习缩放因子逐步引入残差贡献
    3. 添加多个偏置项来调整激活分布
    """
    def __init__(self, dim, num_layers):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        # 三个偏置项分别用于:
        # bias1: 残差分支输入调整
        # bias2: 残差分支输出调整  
        # bias3: 最终输出调整
        self.bias1 = nn.Parameter(torch.zeros(dim))
        self.bias2 = nn.Parameter(torch.zeros(dim))
        self.bias3 = nn.Parameter(torch.zeros(dim))
        
        # 计算 Fixup 缩放因子
        # L^(-1/(2m-2))，其中 L 是总层数，m 是残差分支层数
        self.fixup_scale = num_layers ** (-1/4)  # 假设 m=3
    
    def forward(self, x, residual_fn):
        """
        前向传播
        
        参数:
            x: 输入张量
            residual_fn: 残差分支函数
        """
        # 残差分支计算，应用缩放因子
        residual = self.fixup_scale * self.scale * residual_fn(x + self.bias1)
        return x + residual + self.bias2 + self.bias3

class FixupBasicBlock(nn.Module):
    """Fixup 基本残差块（类似 ResNet BasicBlock）"""
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # 关键：最后一层卷积权重初始化为零
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
        # Fixup 缩放因子
        self.scale = nn.Parameter(torch.ones(1))
        self.bias1 = nn.Parameter(torch.zeros(out_channels))
        self.bias2 = nn.Parameter(torch.zeros(out_channels))
        self.bias3 = nn.Parameter(torch.zeros(out_channels))
        
        self.fixup_scale = num_layers ** (-1/4)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x + self.bias1.view(1, -1, 1, 1))
        out = nn.functional.relu(out)
        out = self.conv2(out)
        
        # 应用 Fixup 缩放
        out = self.fixup_scale * self.scale * out
        
        out = out + self.bias2.view(1, -1, 1, 1)
        out = out + identity + self.bias3.view(1, -1, 1, 1)
        out = nn.functional.relu(out)
        
        return out
```

**机构**: [Zhang et al. 2019] — University of Toronto, Vector Institute  
**信源**: [Fixup Initialization](https://arxiv.org/abs/1901.09321)

### 3.1.4 低精度训练中的初始化调整

#### 3.1.4.1 方差限制的数学原因

在 FP16（半精度浮点数）训练中，数值表示范围受限：

- FP16 的指数范围：$2^{-14}$ 到 $2^{15}$（约 $6.1 \times 10^{-5}$ 到 $6.5 \times 10^{4}$）
- FP16 的尾数精度：约 3-4 位有效十进制数字

初始化时的方差过大可能导致：
1. **上溢（Overflow）**：权重值超过 FP16 最大表示范围
2. **梯度爆炸**：反向传播时梯度值过大

初始化时的方差过小可能导致：
1. **下溢（Underflow）**：权重值小于 FP16 最小表示范围，被置为0
2. **梯度消失**：反向传播时梯度值过小

设 FP16 的最小正规格化数为 $N_{min} = 2^{-14} \approx 6.1 \times 10^{-5}$。为了保证初始化后的权重不会下溢，需要：

$$
\sigma \gg N_{min}
$$

同时，为了避免上溢，需要：

$$
3\sigma \ll N_{max} = 2^{15} \approx 32768
$$

实际训练中，通常限制标准差不超过0.01：

```python
def low_precision_init(tensor, gain=1.0, max_std=0.01, dtype=None):
    """
    适合低精度训练（FP16/BF16）的初始化
    
    参数:
        tensor: 待初始化的权重张量
        gain: 基础缩放因子
        max_std: FP16 训练时的最大标准差限制
        dtype: 目标数据类型
    
    数学原理:
        FP16 数值范围: [6.1e-5, 65504]
        为了避免数值问题:
        1. 限制标准差，避免极端值
        2. 对于小维度输入，可能需要增大标准差
        3. 考虑激活函数的修正因子
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    
    # 计算基础标准差（Kaiming 初始化）
    std = gain * math.sqrt(2.0 / fan_in)
    
    # 检测目标数据类型
    target_dtype = dtype or tensor.dtype
    
    if target_dtype == torch.float16:
        # FP16 限制标准差
        # 原因:
        # 1. 避免极端值导致的上溢
        # 2. 保证梯度在合理范围内
        # 3. 与 LayerNorm/Adam 等优化器配合
        std = min(std, max_std)
        
        # 额外安全检查：确保标准差不会导致数值问题
        # 对于极小的 fan_in，可能需要特殊处理
        if fan_in < 10:
            std = max(std, 0.001)  # 避免方差过小
            
    elif target_dtype == torch.bfloat16:
        # BF16 有更大的动态范围，限制可以放宽
        std = min(std, max_std * 2)
    
    with torch.no_grad():
        return tensor.normal_(0, std)

def analyze_init_distribution(fan_in, fan_out, dtype=torch.float16):
    """
    分析初始化分布的数值特性
    
    返回:
        包含均值、标准差、极值概率等统计信息的字典
    """
    std = math.sqrt(2.0 / fan_in)
    
    # FP16 限制
    if dtype == torch.float16:
        std = min(std, 0.01)
    
    # 计算极值概率（假设正态分布）
    # P(|x| > 3*sigma) 的概率
    extreme_prob = 2 * (1 - 0.9987)  # 约 0.26%
    
    # 计算下溢概率（对于 FP16）
    if dtype == torch.float16:
        from scipy import stats
        min_normal = 6.1e-5
        underflow_prob = 2 * stats.norm.cdf(min_normal, 0, std)
    else:
        underflow_prob = 0.0
    
    return {
        'fan_in': fan_in,
        'fan_out': fan_out,
        'std': std,
        'extreme_prob': extreme_prob,
        'underflow_prob': underflow_prob,
        'dtype': str(dtype)
    }
```

## 3.2 归一化层的数值行为

### 3.2.1 BatchNorm 的数值稳定性

#### 3.2.1.1 小 batch 的统计量方差

BatchNorm 对每个特征维度计算 batch 统计量：

$$
\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i, \quad \sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2
$$

其中 $m$ 是 batch size。

**统计量方差分析**：

对于独立同分布的输入 $x_i \sim N(\mu, \sigma^2)$，样本均值的方差为：

$$
\text{Var}(\mu_B) = \frac{\sigma^2}{m}
$$

样本方差的方差为：

$$
\text{Var}(\sigma_B^2) = \frac{2\sigma^4}{m-1}
$$

当 batch size $m$ 较小时，统计量的方差增大，导致：
1. 训练不稳定
2. 测试时使用的移动平均统计量与训练时不一致
3. 梯度估计噪声增大

**小 batch 的数值问题**：

当 $m < 32$ 时，建议增大 `eps` 参数：

$$
y = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \cdot \gamma + \beta
$$

增大 $\epsilon$ 可以：
1. 防止除零
2. 减小小 batch 统计量噪声的影响
3. 提高数值稳定性

#### 3.2.1.2 分布式 BN 的误差累积

在分布式训练中，BatchNorm 需要在多个 GPU 间同步统计量。设 $N$ 个 GPU，每个处理 $m$ 个样本：

**全局统计量计算**：

$$
\mu_{global} = \frac{1}{Nm} \sum_{j=1}^{N} \sum_{i=1}^{m} x_i^{(j)}
$$

$$
\sigma_{global}^2 = \frac{1}{Nm} \sum_{j=1}^{N} \sum_{i=1}^{m} (x_i^{(j)} - \mu_{global})^2
$$

**数值误差来源**：

1. **通信误差**：All-Reduce 操作可能引入舍入误差
2. **数值稳定性**：大 batch 下的求和可能丢失精度
3. **同步延迟**：异步训练时统计量不一致

```python
import torch
import torch.nn as nn
import torch.distributed as dist

class StableBatchNorm(nn.Module):
    """
    数值稳定的 BatchNorm 实现
    
    特性:
    1. 针对小 batch 增大 eps
    2. 分布式训练优化
    3. FP16 安全计算
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, 
                 min_batch_size=32, fp16_eps=1e-3):
        super().__init__()
        self.num_features = num_features
        
        # 根据精度选择 eps
        self.base_eps = eps
        self.fp16_eps = fp16_eps
        self.min_batch_size = min_batch_size
        
        # 可学习参数
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # 运行时统计量
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
    def forward(self, x):
        """
        前向传播，自动处理数值稳定性
        
        参数:
            x: 输入张量 [N, C, ...]
        """
        batch_size = x.size(0)
        
        # 根据 batch size 和数据类型调整 eps
        if x.dtype == torch.float16 and batch_size < self.min_batch_size:
            eps = self.fp16_eps
        else:
            eps = self.base_eps
        
        if self.training:
            # 在 FP32 中计算统计量以提高精度
            x_fp32 = x.float()
            
            # 计算均值和方差
            dims = [0] + list(range(2, x.dim()))
            mean = x_fp32.mean(dim=dims)
            var = x_fp32.var(dim=dims, unbiased=False)
            
            # 分布式同步（如果可用）
            if dist.is_initialized() and dist.get_world_size() > 1:
                # 使用 All-Reduce 同步统计量
                # 注意：这里需要处理数值稳定性
                world_size = dist.get_world_size()
                
                # 同步均值
                mean_all = mean * world_size
                dist.all_reduce(mean_all)
                mean_all = mean_all / world_size
                
                # 同步方差（使用 Welford 算法减少误差）
                var_all = var * world_size
                dist.all_reduce(var_all)
                var_all = var_all / world_size
                
                mean = mean_all
                var = var_all
            
            # 更新运行时统计量
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + \
                                   self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + \
                                  self.momentum * var
                self.num_batches_tracked += 1
            
            # 归一化
            x_norm = (x_fp32 - mean.view(1, -1, *([1] * (x.dim() - 2)))) / \
                     torch.sqrt(var.view(1, -1, *([1] * (x.dim() - 2))) + eps)
            
            # 应用缩放和偏移
            output = x_norm * self.weight.view(1, -1, *([1] * (x.dim() - 2))) + \
                     self.bias.view(1, -1, *([1] * (x.dim() - 2)))
            
            return output.to(x.dtype)
        else:
            # 评估模式：使用运行时统计量
            x_fp32 = x.float()
            x_norm = (x_fp32 - self.running_mean.view(1, -1, *([1] * (x.dim() - 2)))) / \
                     torch.sqrt(self.running_var.view(1, -1, *([1] * (x.dim() - 2))) + eps)
            output = x_norm * self.weight.view(1, -1, *([1] * (x.dim() - 2))) + \
                     self.bias.view(1, -1, *([1] * (x.dim() - 2)))
            return output.to(x.dtype)

# 使用示例
# 默认 eps=1e-5，对于 FP16 可能需要增大
bn = nn.BatchNorm1d(256, eps=1e-3)  # FP16 训练建议

# 评估模式下的数值稳定
bn.eval()
with torch.no_grad():
    output = bn(input)  # 使用运行统计量
```

**机构**: [Ioffe & Szegedy 2015] — Google  
**信源**: [Batch Normalization](https://arxiv.org/abs/1502.03167)

### 3.2.2 LayerNorm 的 FP16 问题

#### 3.2.2.1 均值/方差计算的数值误差

LayerNorm 对每个样本的特征维度进行归一化：

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

其中：

$$
\mu = \frac{1}{H} \sum_{i=1}^{H} x_i, \quad \sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2
$$

$H$ 是特征维度。

**FP16 数值问题**：

1. **求和误差**：大维度求和时的累积舍入误差
2. **方差计算**：$\sigma^2 = \mathbb{E}[x^2] - \mu^2$ 中的灾难性抵消
3. **除法稳定性**：小方差时的数值不稳定

#### 3.2.2.2 为什么需要在 FP32 中计算

在 FP16 中直接计算 LayerNorm 的问题示例：

假设 $x = [1000.0, 1000.1, 999.9]$ 在 FP16 中：
- FP16 只能精确表示整数到 2048
- 1000.1 和 999.9 可能被舍入到 1000
- 计算得到的方差可能为 0（实际应为 0.01）

**Welford 算法**（数值稳定的方差计算）：

```
M_1 = x_1
S_1 = 0
for k = 2 to n:
    M_k = M_{k-1} + (x_k - M_{k-1}) / k
    S_k = S_{k-1} + (x_k - M_{k-1}) * (x_k - M_k)
return M_n, S_n / (n - 1)
```

```python
class StableLayerNorm(nn.Module):
    """
    数值稳定的 LayerNorm 实现
    
    关键优化:
    1. 在 FP32 中计算均值和方差
    2. 使用 Welford 算法进行数值稳定的方差计算
    3. 支持多种方差计算模式
    """
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
    
    def _welford_var(self, x, mean):
        """
        使用 Welford 算法计算方差
        
        比直接计算 E[x^2] - E[x]^2 更数值稳定
        """
        # x: [..., normalized_shape]
        # 计算 (x - mean)^2 的均值
        var = torch.mean((x - mean) ** 2, dim=-1, keepdim=True)
        return var
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，最后维度为 normalized_shape
        """
        # 保存原始数据类型
        original_dtype = x.dtype
        
        # 在 FP32 中计算以提高数值稳定性
        x_fp32 = x.float()
        
        # 计算均值（最后 normalized_shape 维度）
        mean = x_fp32.mean(dim=-len(self.normalized_shape), keepdim=True)
        
        # 计算方差
        # 使用 Welford 风格的计算
        var = self._welford_var(x_fp32, mean)
        
        # 归一化
        x_norm = (x_fp32 - mean) / torch.sqrt(var + self.eps)
        
        # 应用可学习参数
        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias
        
        # 转换回原始数据类型
        return x_norm.to(original_dtype)

class FusedLayerNorm(nn.Module):
    """
    融合优化的 LayerNorm（模拟 apex 或 torch.nn.functional.layer_norm）
    
    优化点:
    1. 融合多个操作减少 kernel 启动开销
    2. 优化的内存访问模式
    3. 自动处理 FP16/FP32 转换
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x):
        # 使用 PyTorch 内置的融合 LayerNorm
        # 内部已经优化了数值稳定性
        return nn.functional.layer_norm(
            x, 
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps
        )
```

### 3.2.3 RMSNorm 的稳定性优势

#### 3.2.3.1 去除均值计算的数学影响

RMSNorm（Root Mean Square Layer Normalization）去除均值中心化步骤：

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \epsilon}
$$

**去除均值的优势**：

1. **计算简化**：省去一次均值计算和减法操作
2. **数值稳定**：避免 $\mathbb{E}[x^2] - \mu^2$ 形式的灾难性抵消
3. **梯度简化**：反向传播计算更简单

#### 3.2.3.2 方差对比分析

对比 LayerNorm 和 RMSNorm 的数值特性：

**LayerNorm**：

$$
y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_i
$$

**RMSNorm**：

$$
y_i = \frac{x_i}{\sqrt{\frac{1}{n}\sum x_j^2 + \epsilon}} \cdot \gamma_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i
$$

**方差分析**：

假设输入 $x$ 已经过某种归一化（如前一层的输出），则：

- LayerNorm 强制输出均值为0
- RMSNorm 保持输入的均值特性

对于以0为中心的输入，两者等价。但对于非零均值输入：

$$
\text{RMS}^2 = \frac{1}{n}\sum x_i^2 = \sigma^2 + \mu^2
$$

因此 RMSNorm 的缩放因子包含了均值信息。

```python
class RMSNorm(nn.Module):
    """
    RMSNorm 实现（LLaMA、T5 等使用）
    
    相比 LayerNorm 的优势:
    1. 去除均值计算，更简单高效
    2. 数值更稳定（无 E[x^2] - E[x]^2 抵消）
    3. 在 Transformer 中表现相当或更好
    
    数学公式:
        RMS(x) = sqrt(mean(x^2) + eps)
        output = x / RMS(x) * gamma
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # 可学习缩放参数 gamma
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量 [..., dim]
        """
        # 保存原始数据类型
        original_dtype = x.dtype
        
        # 在 FP32 中计算范数以保持数值稳定
        x_fp32 = x.float()
        
        # 计算 RMS: sqrt(mean(x^2) + eps)
        # 注意: 这里不需要减去均值
        rms = torch.sqrt(torch.mean(x_fp32 ** 2, dim=-1, keepdim=True) + self.eps)
        
        # 归一化并应用可学习缩放
        x_norm = x_fp32 / rms * self.weight
        
        # 转换回原始数据类型
        return x_norm.to(original_dtype)

class RMSNormWithBias(nn.Module):
    """
    带偏置的 RMSNorm 变体
    
    某些实现（如部分 T5 版本）包含可学习偏置
    """
    def __init__(self, dim, eps=1e-6, use_bias=False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        original_dtype = x.dtype
        x_fp32 = x.float()
        
        # RMS 计算
        rms = torch.sqrt(torch.mean(x_fp32 ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x_fp32 / rms * self.weight
        
        if self.bias is not None:
            x_norm = x_norm + self.bias
        
        return x_norm.to(original_dtype)

# 数值稳定性对比示例
def compare_norm_stability():
    """对比 LayerNorm 和 RMSNorm 的数值稳定性"""
    # 生成测试数据：大数值，小方差
    x = torch.ones(1000) * 1000.0
    x[0] = 1000.1
    x[1] = 999.9
    
    # FP16 测试
    x_fp16 = x.half()
    
    print("输入统计:")
    print(f"  均值: {x.mean().item():.6f}")
    print(f"  方差: {x.var().item():.6f}")
    print(f"  RMS: {torch.sqrt((x ** 2).mean()).item():.6f}")
    
    # LayerNorm 可能遇到问题：1000.1 和 999.9 在 FP16 中可能变成 1000
    # 导致方差被计算为 0
    
    # RMSNorm 更稳定，因为直接使用 x^2
    rms_fp32 = torch.sqrt((x ** 2).mean())
    rms_fp16 = torch.sqrt((x_fp16.float() ** 2).mean())
    
    print(f"\nRMS (FP32): {rms_fp32.item():.6f}")
    print(f"RMS (FP16): {rms_fp16.item():.6f}")
```

**机构**: [Zhang & Sennrich 2019] — University of Edinburgh  
**信源**: [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)

### 3.2.4 GroupNorm 的数值特性

GroupNorm 在小 batch 时比 BatchNorm 更稳定：

```python
# GroupNorm 不受 batch size 影响
gn = nn.GroupNorm(num_groups=32, num_channels=256)

# 适用于小 batch 训练（如检测、分割任务）
```

## 3.3 残差连接与梯度流动

### 3.3.1 残差连接对梯度流的保护

#### 3.3.1.1 梯度保护的数学证明

考虑一个残差块：

$$
y = F(x, \{W_i\}) + x
$$

其中 $F$ 是残差映射（通常包含多个层）。

**梯度分析**：

对损失函数 $L$ 求梯度：

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial L}{\partial y} \cdot \left(\frac{\partial F}{\partial x} + I\right)
$$

其中 $I$ 是单位矩阵。

**关键性质**：

即使 $F$ 的梯度 $\frac{\partial F}{\partial x}$ 很小（梯度消失），+1 项保证梯度不会完全消失：

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} + \frac{\partial L}{\partial y} \cdot \frac{\partial F}{\partial x}
$$

#### 3.3.1.2 雅可比矩阵的特征值分析

考虑残差映射 $F$ 在 $x$ 处的雅可比矩阵 $J_F = \frac{\partial F}{\partial x}$。

残差连接的雅可比矩阵为：

$$
J_{res} = J_F + I
$$

**特征值分析**：

设 $J_F$ 的特征值为 $\lambda_1, \lambda_2, \ldots, \lambda_n$，则 $J_{res}$ 的特征值为 $\lambda_i + 1$。

对于深层网络，梯度传播涉及雅可比矩阵的连乘：

$$
\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial x_L} \cdot \prod_{l=1}^{L} (J_{F_l} + I)
$$

**谱半径分析**：

如果 $J_{F_l}$ 的特征值满足 $|\lambda_i| < 1$，则：

$$
|1 + \lambda_i| > 1 - |\lambda_i| > 0
$$

这保证了梯度不会消失。

如果 $J_{F_l}$ 的特征值很大，$|1 + \lambda_i|$ 可能很大，导致梯度爆炸。因此需要适当的初始化来控制 $J_{F_l}$ 的谱半径。

```python
def analyze_residual_jacobian(residual_block, x, delta=1e-4):
    """
    数值估计残差块的雅可比矩阵并分析特征值
    
    参数:
        residual_block: 残差块函数 F(x)
        x: 输入张量
        delta: 有限差分步长
    
    返回:
        雅可比矩阵的特征值分析结果
    """
    x = x.detach().requires_grad_(True)
    
    # 前向传播
    y = residual_block(x)
    
    # 计算雅可比矩阵（使用自动微分）
    # 对于高维输入，我们采样部分维度
    jac = []
    for i in range(min(100, y.numel())):  # 限制维度
        grad_output = torch.zeros_like(y)
        grad_output.view(-1)[i] = 1.0
        
        grad = torch.autograd.grad(y, x, grad_output, retain_graph=True)[0]
        jac.append(grad.view(-1).detach())
    
    jac_matrix = torch.stack(jac)
    
    # 计算特征值
    eigenvalues = torch.linalg.eigvals(jac_matrix)
    
    # 添加单位矩阵后的特征值
    residual_jac = jac_matrix + torch.eye(jac_matrix.size(0))
    residual_eigenvalues = torch.linalg.eigvals(residual_jac)
    
    return {
        'jacobian_spectral_radius': torch.max(torch.abs(eigenvalues)).item(),
        'residual_spectral_radius': torch.max(torch.abs(residual_eigenvalues)).item(),
        'min_abs_eigenvalue': torch.min(torch.abs(eigenvalues)).item(),
        'min_abs_residual_eigenvalue': torch.min(torch.abs(residual_eigenvalues)).item(),
    }
```

### 3.3.2 Pre-LN vs Post-LN 的梯度流分析

#### 3.3.2.1 梯度传播路径长度

**Post-LN（原始 Transformer）**：

```
x = x + Sublayer(LN(x))
```

梯度路径：

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \left(1 + \frac{\partial \text{Sublayer}}{\partial \text{LN}} \cdot \frac{\partial \text{LN}}{\partial x}\right)
$$

**Pre-LN（现代 Transformer）**：

```
x = LN(x + Sublayer(x))
```

梯度路径：

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial \text{LN}}{\partial z} \cdot \left(1 + \frac{\partial \text{Sublayer}}{\partial x}\right)
$$

其中 $z = x + \text{Sublayer}(x)$。

**路径长度对比**：

- Post-LN：梯度通过 LN 后再通过残差分支
- Pre-LN：梯度直接通过残差分支，LN 的梯度影响较小

#### 3.3.2.2 数值稳定性对比

**Post-LN 的问题**：

1. 深层时，输出幅值可能指数增长
2. 需要精心设计的学习率 warm-up
3. 梯度在早期层可能过大

**Pre-LN 的优势**：

1. 每层输出被 LN 约束，幅值稳定
2. 训练更稳定，可以使用更大的学习率
3. 减少了对 warm-up 的依赖

**数学分析**：

设第 $l$ 层的输入为 $x_l$，输出为 $x_{l+1}$。

Post-LN：

$$
x_{l+1} = x_l + F_l(\text{LN}(x_l))
$$

如果 $F_l$ 的输出方差与输入相当，经过 $L$ 层后：

$$
\text{Var}(x_L) \approx (1 + \alpha)^L \cdot \text{Var}(x_0)
$$

Pre-LN：

$$
x_{l+1} = \text{LN}(x_l + F_l(x_l))
$$

LN 将输出方差归一化为1，防止指数增长。

```python
class PostLNTransformerBlock(nn.Module):
    """
    Post-LN Transformer 块（原始 Transformer 设计）
    
    结构: x = x + Sublayer(LN(x))
    
    特点:
    - 梯度路径短
    - 但数值可能不稳定，需要 careful warm-up
    """
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # Post-LN: 先归一化再计算
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask)
        x = x + attn_out
        
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x

class PreLNTransformerBlock(nn.Module):
    """
    Pre-LN Transformer 块（现代设计，LLaMA、GPT-3 等使用）
    
    结构: x = LN(x + Sublayer(x))
    
    特点:
    - 数值更稳定
    - 可以使用更大的学习率
    - 减少 warm-up 依赖
    """
    def __init__(self, dim, num_heads, dropout=0.1, use_rmsnorm=True):
        super().__init__()
        NormLayer = RMSNorm if use_rmsnorm else nn.LayerNorm
        
        self.norm1 = NormLayer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = NormLayer(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # Pre-LN: 先归一化再计算
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask)
        x = x + attn_out
        
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x

# 现代模型（LLaMA、GPT-3）普遍采用 Pre-LN
class ModernTransformerBlock(nn.Module):
    """
    现代 Transformer 块（LLaMA 风格）
    
    特性:
    - Pre-LN 结构
    - RMSNorm 替代 LayerNorm
    - SwiGLU FFN
    - RoPE 位置编码
    """
    def __init__(self, dim, num_heads, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or int(8 * dim / 3)  # LLaMA 使用 2/3 * 4d
        
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        
        # SwiGLU FFN
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x):
        # Self-attention with residual
        h = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # SwiGLU FFN with residual
        # SwiGLU(x) = (W1(x) * Swish(W3(x))) @ W2
        ffn_out = self.w2(nn.functional.silu(self.w1(self.norm2(h))) * self.w3(self.norm2(h)))
        out = h + ffn_out
        
        return out
```

### 3.3.3 梯度检查点的数值影响

```python
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(self, x):
    """使用梯度检查点节省内存
    
    原理:
    - 前向时不保存中间激活值
    - 反向时重新计算前向传播
    
    数值注意:
    - 重新计算可能引入微小数值差异
    - 对于确定性操作，差异通常在 1e-7 级别
    - 通常可忽略，但在极端情况下可能影响收敛
    """
    # use_reentrant=False 推荐用于更高效的梯度计算
    return checkpoint(self.layer, x, use_reentrant=False)

class CheckpointedTransformer(nn.Module):
    """使用梯度检查点的 Transformer"""
    def __init__(self, num_layers, dim, num_heads):
        super().__init__()
        self.layers = nn.ModuleList([
            PreLNTransformerBlock(dim, num_heads) 
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            # 每层使用检查点
            x = checkpoint(layer, x, use_reentrant=False)
        return x
```

**数值注意**：重新计算可能引入微小数值差异，通常可忽略。

## 3.4 激活函数的数值陷阱

### 3.4.1 GELU/Swish 的数值分析

#### 3.4.1.1 指数函数的溢出风险

GELU（Gaussian Error Linear Unit）定义为：

$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数。

**近似形式**：

$$
\text{GELU}(x) \approx x \cdot \sigma(1.702x) = \frac{x}{1 + e^{-1.702x}}
$$

**数值问题**：

1. **大正数溢出**：当 $x$ 很大时，$e^{1.702x}$ 可能上溢
2. **大负数下溢**：当 $x$ 为很大的负数时，$e^{1.702x}$ 可能下溢为0
3. **中间值精度**：近似计算的误差

#### 3.4.1.2 近似计算的误差

**erf 函数的近似**：

标准 GELU 实现使用 erf 的数值近似：

$$
\text{erf}(x) \approx \text{tanh}\left(\sqrt{\frac{2}{\pi}} x (1 + 0.044715 x^2)\right)
$$

这个近似的误差约为 $10^{-5}$ 级别。

**Swish/SiLU**：

$$
\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

数值问题与 GELU 类似，涉及 sigmoid 的指数运算。

```python
class StableGELU(nn.Module):
    """
    数值稳定的 GELU 实现
    
    优化策略:
    1. 在 FP32 中计算
    2. 使用 PyTorch 内置的 numerically stable 实现
    3. 对大值进行截断处理
    """
    def __init__(self, approximate='tanh'):
        super().__init__()
        self.approximate = approximate
    
    def forward(self, x):
        # 在 FP32 中计算以避免 FP16 溢出
        original_dtype = x.dtype
        x_fp32 = x.float()
        
        # 使用 PyTorch 内置的 stable GELU
        # 内部已经处理了数值稳定性
        output = nn.functional.gelu(x_fp32, approximate=self.approximate)
        
        return output.to(original_dtype)

class StableSwish(nn.Module):
    """
    数值稳定的 Swish/SiLU 实现
    
    Swish(x) = x * sigmoid(x)
    
    数值优化:
    - 对于大负数，sigmoid(x) -> 0，结果 -> 0
    - 对于大正数，sigmoid(x) -> 1，结果 -> x
    - 中间区域需要 careful 计算
    """
    def forward(self, x):
        original_dtype = x.dtype
        x_fp32 = x.float()
        
        # PyTorch 的 silu 已经优化了数值稳定性
        output = nn.functional.silu(x_fp32)
        
        return output.to(original_dtype)

def manual_stable_sigmoid(x):
    """
    手动实现的数值稳定 sigmoid
    
    策略:
    - x >= 0: z = exp(-x), return 1 / (1 + z)
    - x < 0: z = exp(x), return z / (1 + z)
    
    这样避免了对大正数计算 exp(x) 的溢出
    """
    # 在 FP32 中计算
    x = x.float()
    
    # 使用 where 实现条件计算
    z = torch.where(x >= 0, -x, x)
    z = torch.exp(z)
    
    result = torch.where(x >= 0, 1 / (1 + z), z / (1 + z))
    
    return result
```

### 3.4.2 ReLU 的 Dead Neuron 问题

#### 3.4.2.1 概率分析

ReLU 定义为：

$$
\text{ReLU}(x) = \max(0, x)
$$

**Dead Neuron 问题**：

如果一个神经元在训练过程中始终输出负值，则其梯度始终为0，无法更新：

$$
\frac{\partial \text{ReLU}}{\partial x} = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}
$$

**概率分析**：

假设输入 $x \sim N(\mu, \sigma^2)$，则神经元"死亡"的概率为：

$$
P(x \leq 0) = \Phi\left(-\frac{\mu}{\sigma}\right)
$$

其中 $\Phi$ 是标准正态 CDF。

对于初始化时的零均值输入（$\mu = 0$）：

$$
P(x \leq 0) = 0.5
$$

这意味着大约 50% 的神经元在初始化时可能输出0。

**深层网络中的累积效应**：

在深层网络中，如果某层有大量 dead neurons，后续层接收到的有效梯度减少，导致：

1. 梯度稀疏性增加
2. 有效容量降低
3. 训练速度减慢

```python
class StableReLU(nn.Module):
    """
    缓解 Dead Neuron 问题的 ReLU 变体
    """
    pass  # 占位，实际使用 LeakyReLU 或 PReLU

# LeakyReLU 缓解此问题
# f(x) = max(0, x) + negative_slope * min(0, x)
activation = nn.LeakyReLU(negative_slope=0.01)

# 或 PReLU（可学习负斜率）
activation = nn.PReLU()

# 或 SiLU/Swish（平滑替代，无硬阈值）
activation = nn.SiLU()  # x * sigmoid(x)

def analyze_dead_neurons(model, dataloader, device='cuda'):
    """
    分析模型中的 dead neurons
    
    返回每层 ReLU 的激活比例统计
    """
    activation_stats = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
                # 计算激活比例
                active_ratio = (output > 0).float().mean().item()
                if name not in activation_stats:
                    activation_stats[name] = []
                activation_stats[name].append(active_ratio)
        return hook
    
    # 注册 hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # 前向传播收集统计
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            model(inputs)
            break  # 只分析一个 batch
    
    # 移除 hooks
    for hook in hooks:
        hook.remove()
    
    # 分析结果
    for name, ratios in activation_stats.items():
        avg_ratio = sum(ratios) / len(ratios)
        print(f"{name}: 平均激活比例 = {avg_ratio:.2%}")
        if avg_ratio < 0.1:
            print(f"  警告: 可能存在 dead neuron 问题")
    
    return activation_stats
```

### 3.4.3 SwiGLU 的数值特性

SwiGLU 是 Swish + Gating 的组合：

$$
\text{SwiGLU}(x, W, V) = \text{Swish}(xW) \otimes xV
$$

```python
class SwiGLU(nn.Module):
    """
    SwiGLU 激活（PaLM、LLaMA-2 等使用）
    
    结构: (W1(x) * Swish(W3(x))) @ W2
    
    其中:
    - W1, W3: 投影到 hidden_dim
    - W2: 投影回 dim
    - Swish: SiLU 激活
    
    数值特性:
    - 门控机制提供额外的非线性
    - 需要 careful 初始化
    - 在 FP16 中可能遇到溢出
    """
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x):
        # SwiGLU: (W1(x) * Swish(W3(x))) @ W2
        # 在 FP32 中计算门控以避免数值问题
        original_dtype = x.dtype
        x_fp32 = x.float()
        
        gate = nn.functional.silu(self.w3(x_fp32))
        hidden = self.w1(x_fp32) * gate
        output = self.w2(hidden)
        
        return output.to(original_dtype)

class SmoothSwiGLU(nn.Module):
    """
    DeepSeek-V3 提出的平滑 SwiGLU 变体
    
    针对 FP8 训练优化，通过调整激活分布来改善数值稳定性
    
    机构: DeepSeek (幻方量化)
    信源: [DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v1#S5)
    """
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        
        # 平滑因子（可学习或固定）
        self.smooth_factor = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x):
        original_dtype = x.dtype
        x_fp32 = x.float()
        
        # 平滑的门控计算
        gate_input = self.w3(x_fp32) * torch.sigmoid(self.smooth_factor)
        gate = nn.functional.silu(gate_input)
        
        hidden = self.w1(x_fp32) * gate
        output = self.w2(hidden)
        
        return output.to(original_dtype)
```

DeepSeek-V3 提出平滑变体以改善 FP8 训练稳定性：

**机构**: DeepSeek (幻方量化)  
**信源**: [DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v1#S5)

## 3.5 注意力机制的数值稳定性

### 3.5.1 Softmax 的数值稳定性

#### 3.5.1.1 减去最大值的数学保证

标准 Softmax：

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$

**数值问题**：

当 $x_i$ 很大时，$e^{x_i}$ 可能上溢；当 $x_i$ 为很大的负数时，$e^{x_i}$ 可能下溢为0。

**数值稳定的 Softmax**：

$$
\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_{j} e^{x_j - m}}
$$

其中 $m = \max_j x_j$。

**数学保证**：

1. **上溢保护**：$x_i - m \leq 0$，因此 $e^{x_i - m} \leq 1$
2. **下溢保护**：至少有一个元素满足 $x_i - m = 0$，因此分母 $\geq 1$
3. **数值不变性**：分子分母同乘 $e^{-m}$，结果不变

**误差分析**：

设 $\tilde{x}_i = x_i - m$，则：

$$
\text{softmax}(x_i) = \frac{e^{\tilde{x}_i}}{\sum_j e^{\tilde{x}_j}}
$$

由于 $\tilde{x}_i \leq 0$ 且 $\max_i \tilde{x}_i = 0$，指数项都在 $[0, 1]$ 范围内。

```python
def stable_softmax(x, dim=-1):
    """
    数值稳定的 Softmax 实现
    
    算法:
    1. 计算 x_max = max(x, dim)
    2. 计算 x_shifted = x - x_max
    3. 计算 exp_x = exp(x_shifted)
    4. 返回 exp_x / sum(exp_x)
    
    数学保证:
    - x_shifted <= 0，避免 exp 上溢
    - max(x_shifted) = 0，确保至少一个 exp 值为 1
    - 分母 >= 1，避免除零
    """
    # 减去最大值（数值稳定的关键步骤）
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    
    # 计算指数
    exp_x = torch.exp(x_shifted)
    
    # 归一化
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def stable_log_softmax(x, dim=-1):
    """
    数值稳定的 Log-Softmax
    
    使用 log-sum-exp 技巧：
    log(softmax(x_i)) = x_i - max(x) - log(sum(exp(x - max(x))))
    """
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    
    # log(sum(exp(x_shifted)))
    log_sum_exp = torch.log(torch.exp(x_shifted).sum(dim=dim, keepdim=True))
    
    return x_shifted - log_sum_exp
```

### 3.5.2 Flash Attention 的数值精度

Flash Attention 在计算时保持 FP32 累加：

```python
# Flash Attention 自动处理数值精度
from flash_attn import flash_attn_func

# 内部实现:
# 1. 在线计算 softmax，避免存储大的 attention 矩阵
# 2. 使用 FP32 累加 attention 计算
# 3. 分块处理减少内存访问

output = flash_attn_func(q, k, v, causal=True)
```

**Flash Attention 的数值特性**：

1. **在线 Softmax**：不需要存储完整的 $N \times N$ attention 矩阵
2. **FP32 累加**：矩阵乘法使用 FP32 累加器
3. **分块计算**：减少数值误差累积

### 3.5.3 长序列注意力的数值问题

#### 3.5.3.1 注意力权重的分布

对于长序列，注意力分数可能变得非常尖锐（sharp），导致：

1. **梯度消失**：大部分注意力权重接近0，梯度无法传播
2. **数值下溢**：softmax 中的指数项差异过大

**注意力分数分析**：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

设 $S = \frac{QK^T}{\sqrt{d_k}}$，则：

- $S_{ij}$ 的方差约为1（假设 $Q, K$ 已归一化）
- 对于长序列，$S_{ij}$ 的分布可能变得非常分散

**温度参数调整**：

可以通过调整温度参数来控制注意力分布的尖锐程度：

$$
\text{Attention} = \text{softmax}\left(\frac{QK^T}{\tau \sqrt{d_k}}\right)V
$$

增大 $\tau$ 会使注意力分布更平滑。

```python
class StableAttention(nn.Module):
    """
    数值稳定的注意力实现
    
    优化策略:
    1. 使用 FP32 计算 attention scores
    2. 可选的温度参数调整
    3. 支持 attention 缩放
    4. 处理长序列的数值稳定性
    """
    def __init__(self, dim, num_heads, temperature=1.0, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.temperature = temperature
        self.dropout = dropout
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
    
    def forward(self, x, mask=None, is_causal=False):
        """
        前向传播
        
        参数:
            x: 输入张量 [batch, seq_len, dim]
            mask: 可选的 attention mask
            is_causal: 是否使用因果 mask
        """
        batch, seq_len, dim = x.shape
        
        # 投影并分头
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 使用 FP32 计算 attention 以提高数值稳定性
        q_fp32 = q.float()
        k_fp32 = k.float()
        
        # 计算 attention scores
        scores = torch.matmul(q_fp32, k_fp32.transpose(-2, -1))
        
        # 应用缩放和温度
        scores = scores * self.scale / self.temperature
        
        # 应用 mask（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 因果 mask
        if is_causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.to(scores.device), float('-inf'))
        
        # 数值稳定的 softmax
        # 内部自动减去最大值
        attn = torch.softmax(scores, dim=-1)
        
        # dropout
        if self.dropout > 0 and self.training:
            attn = nn.functional.dropout(attn, p=self.dropout)
        
        # 使用 FP32 计算输出
        v_fp32 = v.float()
        output = torch.matmul(attn, v_fp32)
        
        # 转换回原始数据类型并合并头
        output = output.to(q.dtype)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        
        return self.out_proj(output)

class MemoryEfficientAttention(nn.Module):
    """
    内存高效的注意力实现（模拟 Flash Attention）
    
    使用分块计算减少内存使用，同时保持数值稳定性
    """
    def __init__(self, dim, num_heads, block_size=1024):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.block_size = block_size
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        """分块注意力计算"""
        batch, seq_len, dim = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 对于短序列，使用标准 attention
        if seq_len <= self.block_size:
            return self._standard_attention(q, k, v)
        
        # 对于长序列，使用分块计算
        return self._block_attention(q, k, v)
    
    def _standard_attention(self, q, k, v):
        """标准 attention 计算"""
        batch, seq_len, dim = q.shape
        
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2).float()
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2).float()
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2).float()
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out_proj(output)
    
    def _block_attention(self, q, k, v):
        """分块 attention 计算（简化版）"""
        # 实际实现需要更复杂的分块逻辑
        # 这里仅作为示例
        return self._standard_attention(q, k, v)
```

## 3.6 本章小结

| 组件 | 稳定性要点 | 推荐配置 |
|------|-----------|----------|
| 初始化 | 根据激活选择，低精度减小方差 | He init for ReLU, 低精度限制 std |
| BatchNorm | eps 防止除零，小 batch 慎用 | eps=1e-3 for FP16 |
| LayerNorm | 在 FP32 中计算统计量 | custom implementation |
| RMSNorm | 无均值计算，更稳定 | LLaMA/GPT 首选 |
| 残差连接 | Pre-LN 更稳定 | Pre-LN for deep models |
| 激活函数 | GELU/Swish 在 FP32 计算 | wrapper pattern |
| 注意力 | Flash Attention 或 FP32 计算 | use flash_attn if available |

---

**上一章**: [第2章 数据层稳定性](./02-data-stability.md) | **下一章**: [第4章 数值精度与低精度训练](./04-numerical-precision.md)
