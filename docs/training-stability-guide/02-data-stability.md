# 第2章 数据层稳定性

数据是训练的起点，数据层的数值问题会在后续传播放大。本章覆盖数据预处理、增强、管道的稳定性。

## 2.1 数据预处理数值稳定性

### 2.1.1 标准化/归一化的数值误差分析

#### 除零的数学条件

标准标准化（Z-score normalization）定义为：

$$
x' = \frac{x - \mu}{\sigma}
$$

其中 $\mu$ 为样本均值，$\sigma$ 为样本标准差。除零问题发生在 $\sigma \approx 0$ 时。

**数学分析**：

样本方差的计算公式为：

$$
\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2
$$

当所有样本值相等（$x_1 = x_2 = \cdots = x_N = c$）时：

$$
\mu = c, \quad \sigma^2 = 0
$$

此时标准化出现除零错误。更一般地，考虑浮点精度限制，当：

$$
\sigma < \epsilon_{\text{machine}} \cdot |\mu|
$$

其中 $\epsilon_{\text{machine}} \approx 10^{-7}$（FP32）或 $10^{-3}$（FP16），标准化将产生数值不稳定。

**数值稳定的标准化算法**：

采用Welford在线算法计算均值和方差，具有更好的数值稳定性：

```python
import torch
import numpy as np

def welford_normalize(data, eps=1e-8):
    """
    使用Welford算法进行数值稳定的标准化。
    
    Welford算法通过增量更新避免了大数求和导致的精度损失：
    μ_n = μ_{n-1} + (x_n - μ_{n-1}) / n
    M_n = M_{n-1} + (x_n - μ_{n-1})(x_n - μ_n)
    σ_n² = M_n / n
    
    Args:
        data: 输入张量，形状为 (N, ...) 或任意形状
        eps: 防止除零的小常数，默认1e-8
        
    Returns:
        标准化后的张量
    """
    n = data.numel()
    if n == 0:
        return data
    
    # Welford在线算法
    mean = 0.0
    M2 = 0.0
    
    flat_data = data.flatten()
    for i, x in enumerate(flat_data):
        delta = x - mean
        mean = mean + delta / (i + 1)
        delta2 = x - mean
        M2 = M2 + delta * delta2
    
    # 样本方差（无偏估计使用 n-1）
    variance = M2 / n if n > 1 else 0.0
    std = torch.sqrt(torch.tensor(variance)) if isinstance(variance, (int, float)) else torch.sqrt(variance)
    
    # 数值安全的除法
    std = torch.clamp(std, min=eps)
    
    return (data - mean) / std


def safe_normalize(data, eps=1e-8):
    """
    安全的标准化，防止除零。
    
    数学原理：
    当 σ → 0 时，x' = (x - μ) / (σ + eps) → (x - μ) / eps
    这避免了Inf/NaN，但结果数值会很大，需要后续处理。
    
    Args:
        data: 输入张量
        eps: 数值稳定常数，默认1e-8
        
    Returns:
        标准化后的张量
    """
    mean = data.mean()
    std = data.std()
    
    # 添加 epsilon 防止除零
    # 当 std=0 时，结果变为 (data - mean) / eps
    # 如果 data == mean，则结果为 0
    return (data - mean) / (std + eps)
```

#### 极端值的数值影响

极端值（outliers）对标准化有显著影响。设数据集包含异常值 $x_{\text{out}}$：

$$
\mu_{\text{biased}} = \frac{(N-1)\mu_{\text{clean}} + x_{\text{out}}}{N}
$$

当 $|x_{\text{out}} - \mu_{\text{clean}}| \gg \sigma_{\text{clean}}$ 时：

$$
\sigma_{\text{biased}}^2 \approx \sigma_{\text{clean}}^2 + \frac{(x_{\text{out}} - \mu_{\text{clean}})^2}{N}
$$

这导致正常数据点的标准化值被压缩：

$$
x'_{\text{clean}} = \frac{x_{\text{clean}} - \mu_{\text{biased}}}{\sigma_{\text{biased}}} \ll x'_{\text{ideal}}
$$

**鲁棒归一化实现**：

```python
def robust_normalize(data, percentiles=(0.01, 0.99), eps=1e-8):
    """
    基于百分位数的鲁棒归一化。
    
    数学原理：
    使用分位数代替均值/方差，对异常值具有鲁棒性。
    设 p_low 和 p_high 为指定百分位数，则：
    x' = (x - p_low) / (p_high - p_low + eps)
    
    这种方法的崩溃点（breakdown point）为 min(p, 1-p)，
    即最多可以容忍 p*N 个异常值。
    
    Args:
        data: 输入张量或数组
        percentiles: 使用的百分位数，默认(0.01, 0.99)
        eps: 防止除零的小常数
        
    Returns:
        归一化到[0, 1]范围的数据
    """
    # 计算百分位数（对异常值鲁棒）
    low, high = np.percentile(data.cpu().numpy() if torch.is_tensor(data) else data, 
                               [p * 100 for p in percentiles])
    
    # 数值安全的归一化
    scale = high - low + eps
    normalized = (data - low) / scale
    
    # 钳制到有效范围
    return torch.clamp(normalized, 0.0, 1.0) if torch.is_tensor(data) else np.clip(normalized, 0.0, 1.0)
```

### 2.1.2 Z-score方法的统计理论

#### 正态假设下的功效

Z-score标准化基于正态分布假设。对于 $X \sim \mathcal{N}(\mu, \sigma^2)$，标准化后：

$$
Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)
$$

**统计功效分析**：

在假设检验框架下，Z-score方法检测异常值的统计功效为：

$$
\text{Power}(\alpha) = P(|Z| > z_{\alpha/2} | H_1)
$$

其中 $z_{\alpha/2}$ 是标准正态分布的临界值，$H_1$ 是备择假设（存在异常值）。

对于均值偏移模型（mean-shift model），设异常值来自 $X_{\text{out}} \sim \mathcal{N}(\mu + \delta, \sigma^2)$：

$$
\text{Power} = \Phi\left(-z_{\alpha/2} - \frac{\delta}{\sigma}\right) + 1 - \Phi\left(z_{\alpha/2} - \frac{\delta}{\sigma}\right)
$$

其中 $\Phi$ 是标准正态CDF。

```python
def zscore_statistical_properties(data, threshold=3.0):
    """
    分析Z-score方法的统计特性。
    
    理论背景：
    在标准正态假设下，|Z| > 3 的概率约为 0.0027。
    因此 threshold=3 对应约 99.73% 的置信水平。
    
    Args:
        data: 输入数据
        threshold: Z-score阈值
        
    Returns:
        包含统计指标的字典
    """
    from scipy import stats
    
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    # 计算Z-scores
    z_scores = np.abs((data - mean) / std)
    
    # 理论预期异常值数量（在正态假设下）
    alpha = 2 * (1 - stats.norm.cdf(threshold))
    expected_outliers = n * alpha
    
    # 实际检测到的异常值
    detected_outliers = np.sum(z_scores > threshold)
    
    # 计算统计功效（假设存在delta偏移的异常值）
    delta = 2 * std  # 假设异常值偏移2个标准差
    effect_size = delta / std
    power = (1 - stats.norm.cdf(threshold - effect_size)) + \
            stats.norm.cdf(-threshold - effect_size)
    
    return {
        'mean': mean,
        'std': std,
        'expected_outliers': expected_outliers,
        'detected_outliers': detected_outliers,
        'detection_rate': detected_outliers / n,
        'power_at_2sigma': power,
        'alpha_level': alpha
    }
```

#### 异常值对均值/方差的影响

设数据集 $D = \{x_1, \ldots, x_n\}$ 包含 $k$ 个异常值。污染率 $\epsilon = k/n$。

**对均值的影响**：

$$
\mu_{\text{contaminated}} = (1-\epsilon)\mu_{\text{clean}} + \epsilon \mu_{\text{out}}
$$

偏差为：

$$
\text{Bias}(\mu) = \epsilon(\mu_{\text{out}} - \mu_{\text{clean}})
$$

**对方差的影响**：

 contaminated方差可分解为：

$$
\sigma^2_{\text{contaminated}} = (1-\epsilon)\sigma^2_{\text{clean}} + \epsilon\sigma^2_{\text{out}} + \epsilon(1-\epsilon)(\mu_{\text{clean}} - \mu_{\text{out}})^2
$$

最后一项是组间方差，当 $|\mu_{\text{clean}} - \mu_{\text{out}}|$ 很大时占主导。

```python
def analyze_outlier_impact(data, outlier_indices):
    """
    分析异常值对均值和方差的影响。
    
    Args:
        data: 完整数据集
        outlier_indices: 异常值索引列表
        
    Returns:
        影响分析报告
    """
    n = len(data)
    k = len(outlier_indices)
    epsilon = k / n
    
    # 完整统计量
    mean_full = np.mean(data)
    var_full = np.var(data, ddof=1)
    
    # 清洁数据统计量
    clean_mask = np.ones(n, dtype=bool)
    clean_mask[outlier_indices] = False
    data_clean = data[clean_mask]
    
    mean_clean = np.mean(data_clean)
    var_clean = np.var(data_clean, ddof=1)
    
    # 异常值统计量
    data_out = data[outlier_indices]
    mean_out = np.mean(data_out) if len(data_out) > 0 else 0
    var_out = np.var(data_out, ddof=1) if len(data_out) > 1 else 0
    
    # 理论预测
    mean_predicted = (1 - epsilon) * mean_clean + epsilon * mean_out
    var_predicted = (1 - epsilon) * var_clean + epsilon * var_out + \
                    epsilon * (1 - epsilon) * (mean_clean - mean_out) ** 2
    
    return {
        'contamination_rate': epsilon,
        'mean': {
            'full': mean_full,
            'clean': mean_clean,
            'predicted': mean_predicted,
            'bias': mean_full - mean_clean,
            'relative_bias': (mean_full - mean_clean) / mean_clean if mean_clean != 0 else float('inf')
        },
        'variance': {
            'full': var_full,
            'clean': var_clean,
            'predicted': var_predicted,
            'inflation': var_full / var_clean if var_clean > 0 else float('inf')
        }
    }
```

### 2.1.3 IQR方法的鲁棒性分析

#### 分位数的计算复杂度

IQR（Interquartile Range，四分位距）定义为：

$$
\text{IQR} = Q_3 - Q_1
$$

其中 $Q_1$ 是第25百分位数，$Q_3$ 是第75百分位数。

**分位数计算算法**：

1. **排序方法**：时间复杂度 $O(n \log n)$，空间复杂度 $O(n)$ 或 $O(1)$（原地排序）
2. **快速选择（QuickSelect）**：平均 $O(n)$，最坏 $O(n^2)$
3. **中位数的中位数**：确定性 $O(n)$，但常数较大

对于大规模数据，通常使用近似分位数算法（如T-Digest、Q-Digest）。

```python
def efficient_quantile(data, q, method='default'):
    """
    高效计算分位数。
    
    复杂度分析：
    - 排序法：O(n log n)，最稳定
    - QuickSelect：平均O(n)，最坏O(n²)
    - 近似算法：O(n) 或 O(n log n)，适用于流式数据
    
    Args:
        data: 输入数据
        q: 分位数（0-1之间）
        method: 计算方法
        
    Returns:
        分位数值
    """
    if method == 'quickselect':
        # 使用numpy的partition实现近似O(n)
        k = int(q * (len(data) - 1))
        partitioned = np.partition(data, k)
        return partitioned[k]
    elif method == 'streaming':
        # P²算法（用于流式数据的近似分位数）
        return p2_quantile(data, q)
    else:
        # 默认使用numpy的优化实现
        return np.percentile(data, q * 100)


def p2_quantile(data, q):
    """
    P²算法计算近似分位数。
    
    适用于流式场景，只需维护少量标记点。
    空间复杂度O(1)，时间复杂度O(n)。
    
    参考：Jain, R. & Chlamtac, I. (1985). 
    "The P² algorithm for dynamic calculation of quantiles and 
    histograms without storing observations."
    """
    n = len(data)
    if n == 0:
        return 0
    
    # 初始化5个标记点
    markers = [data[0]]
    for x in data[1:]:
        if len(markers) < 5:
            markers.append(x)
            markers.sort()
        else:
            # 更新标记点位置
            markers = update_markers(markers, x, q, n)
    
    # 返回目标分位数估计
    return markers[2]  # 简化实现，实际应插值


def update_markers(markers, x, q, n):
    """P²算法标记点更新（简化版）"""
    # 找到x应该插入的位置
    markers.append(x)
    markers.sort()
    if len(markers) > 5:
        markers.pop(0) if x > markers[2] else markers.pop()
    return markers
```

#### 对异常值的容忍度

IQR方法的崩溃点（breakdown point）为25%，即可以容忍最多25%的数据被污染。

**鲁棒性分析**：

设数据被污染率为 $\epsilon$，则：

- 当 $\epsilon < 0.25$ 时，$Q_1$ 和 $Q_3$ 保持稳定
- 当 $\epsilon \geq 0.25$ 时，分位数可能被异常值"拉偏"

异常值检测规则（Tukey's fences）：

$$
\text{Lower} = Q_1 - k \cdot \text{IQR}, \quad \text{Upper} = Q_3 + k \cdot \text{IQR}
$$

通常 $k=1.5$（温和异常值）或 $k=3$（极端异常值）。

```python
def detect_outliers_iqr(data, k=1.5):
    """
    IQR方法检测异常值。
    
    数学原理：
    使用Tukey's fences识别异常值：
    - 温和异常值：在 [Q1 - 1.5*IQR, Q3 + 1.5*IQR] 之外
    - 极端异常值：在 [Q1 - 3*IQR, Q3 + 3*IQR] 之外
    
    鲁棒性：
    - 崩溃点：25%（可容忍最多25%的异常值）
    - 对极端异常值不敏感（不像Z-score）
    
    Args:
        data: 输入张量
        k: IQR乘数，默认1.5
        
    Returns:
        异常值掩码
    """
    # 计算四分位数
    q1, q3 = torch.quantile(data, torch.tensor([0.25, 0.75]))
    iqr = q3 - q1
    
    # 计算边界
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    return (data < lower_bound) | (data > upper_bound)


def detect_outliers_zscore(data, threshold=3.0):
    """
    Z-score方法检测异常值。
    
    数学原理：
    Z = (x - μ) / σ
    当 |Z| > threshold 时判定为异常值。
    
    在标准正态假设下：
    - threshold=2 对应约95%置信度
    - threshold=3 对应约99.7%置信度
    
    注意：对异常值敏感，异常值会膨胀σ导致检测能力下降。
    
    Args:
        data: 输入张量
        threshold: Z-score阈值，默认3.0
        
    Returns:
        异常值掩码
    """
    mean = data.mean()
    std = data.std()
    
    # 防止除零
    std = torch.clamp(std, min=1e-8)
    
    z_scores = torch.abs((data - mean) / std)
    return z_scores > threshold
```

**处理策略**：

```python
def handle_outliers(data, method='clip', lower=None, upper=None):
    """
    处理异常值。
    
    方法说明：
    - clip: 钳制到边界，保留数据量但可能引入偏差
    - remove: 删除异常值，减少数据量但保持分布
    - replace: 用中位数替换，保留数据量且更鲁棒
    - winsorize:  Winsorization，用百分位数边界替换
    
    Args:
        data: 输入数据
        method: 处理方法
        lower, upper: 边界值
        
    Returns:
        处理后的数据
    """
    if method == 'clip':
        # 钳制到边界
        return torch.clamp(data, lower, upper)
    
    elif method == 'remove':
        # 删除异常值
        mask = (data >= lower) & (data <= upper)
        return data[mask]
    
    elif method == 'replace':
        # 用中位数替换
        median = data.median()
        data = data.clone()
        data[(data < lower) | (data > upper)] = median
        return data
    
    elif method == 'winsorize':
        # Winsorization：用边界值替换
        data = data.clone()
        data[data < lower] = lower
        data[data > upper] = upper
        return data
    
    else:
        raise ValueError(f"Unknown method: {method}")
```

### 2.1.4 数据类型转换陷阱

```python
from PIL import Image
import numpy as np
import torch

def demonstrate_dtype_traps():
    """
    演示数据类型转换中的数值陷阱。
    
    数值问题：
    1. uint8溢出：uint8范围[0,255]，运算可能溢出
    2. FP16下溢：小于~6e-5的值变为0
    3. 整数除法截断：Python 2风格除法
    """
    # 危险：默认转换可能丢失精度
    image = Image.open("image.jpg")  # 8-bit RGB
    image_array = np.array(image)  # uint8, 范围[0, 255]
    image_tensor = torch.tensor(image_array)  # 仍为uint8
    
    # 危险操作1：uint8运算溢出
    # image_tensor + 10  # 如果像素值>245，会溢出回绕
    
    # 危险操作2：直接除 255 在 FP16 中
    # image_fp16 = image_tensor.half() / 255.0  # 风险：中间结果可能下溢
    
    # 正确做法：先转 float 再处理
    image_tensor = image_tensor.float()  # 转为FP32
    
    # 安全做法：在 FP32 中计算，再转 FP16
    image_safe = image_tensor.float() / 255.0  # 归一化到[0,1]
    image_fp16_safe = image_safe.half()  # 此时数值范围安全
    
    return image_safe, image_fp16_safe


def safe_dtype_conversion(tensor, target_dtype=torch.float32):
    """
    安全的数据类型转换。
    
    策略：
    1. 整数转浮点：先扩宽到FP32，避免精度损失
    2. 浮点降精度：检查数值范围，避免溢出/下溢
    3. 批量转换：使用torch的向量化操作
    
    Args:
        tensor: 输入张量
        target_dtype: 目标数据类型
        
    Returns:
        转换后的张量
    """
    current_dtype = tensor.dtype
    
    # 整数转浮点
    if current_dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        if target_dtype in [torch.float16, torch.bfloat16]:
            # 先转FP32，再转目标类型
            return tensor.float().to(target_dtype)
        else:
            return tensor.to(target_dtype)
    
    # 浮点转整数（需要量化）
    elif target_dtype in [torch.uint8, torch.int8]:
        # 钳制到目标类型范围
        info = torch.iinfo(target_dtype)
        tensor = torch.clamp(tensor, info.min, info.max)
        return tensor.to(target_dtype)
    
    # 浮点间转换
    else:
        if target_dtype == torch.float16:
            # 检查下溢风险
            min_val = tensor.abs().min()
            if min_val > 0 and min_val < 6e-5:  # FP16最小正规格数
                print(f"Warning: Values may underflow in FP16 (min={min_val:.2e})")
        return tensor.to(target_dtype)
```

## 2.2 数据增强的数值边界

### 2.2.1 图像变换的数值溢出分析

#### 插值算法的误差

图像几何变换（旋转、缩放、仿射）涉及像素重采样。设变换函数为 $T: \mathbb{R}^2 \rightarrow \mathbb{R}^2$，则输出像素值为：

$$
I'(x, y) = I(T^{-1}(x, y))
$$

由于 $T^{-1}(x, y)$ 通常不在整数网格上，需要使用插值。

**最近邻插值**：

$$
I'(x, y) = I(\lfloor x' + 0.5 \rfloor, \lfloor y' + 0.5 \rfloor)
$$

无新值产生，但可能引入锯齿。

**双线性插值**：

设 $(x', y')$ 的四个邻近像素为 $Q_{11}, Q_{12}, Q_{21}, Q_{22}$：

$$
I'(x, y) = \sum_{i,j \in \{1,2\}} w_{ij} \cdot Q_{ij}
$$

其中权重 $w_{ij}$ 满足 $\sum w_{ij} = 1$。由于凸组合性质：

$$
\min(Q) \leq I'(x, y) \leq \max(Q)
$$

双线性插值不会产生超出原始范围的值。

**双三次插值**：

使用16个邻近像素的加权：

$$
I'(x, y) = \sum_{i=-1}^{2} \sum_{j=-1}^{2} w_{ij} \cdot Q_{ij}
$$

权重可能为负（如Cubic kernel），导致：

$$
I'(x, y) \notin [\min(Q), \max(Q)]
$$

这是数值溢出的主要来源。

```python
import torch
import torch.nn.functional as F

def analyze_interpolation_error(image, angle, scale):
    """
    分析不同插值方法的数值误差。
    
    误差来源：
    1. 最近邻：混叠误差，高频信息丢失
    2. 双线性：平滑误差，边缘模糊
    3. 双三次：振铃效应，可能产生负值/超范围值
    
    Args:
        image: 输入图像，形状(C, H, W)
        angle: 旋转角度（度）
        scale: 缩放因子
        
    Returns:
        各插值方法的结果和误差分析
    """
    import torchvision.transforms as T
    
    results = {}
    original_range = (image.min().item(), image.max().item())
    
    for mode in ['nearest', 'bilinear', 'bicubic']:
        # 应用旋转变换
        transform = T.Compose([
            T.RandomRotation((angle, angle)),
        ])
        
        # 手动实现以控制插值模式
        theta = torch.tensor([
            [torch.cos(torch.tensor(angle * 3.14159 / 180)), 
             -torch.sin(torch.tensor(angle * 3.14159 / 180)), 0],
            [torch.sin(torch.tensor(angle * 3.14159 / 180)), 
             torch.cos(torch.tensor(angle * 3.14159 / 180)), 0]
        ], dtype=torch.float).unsqueeze(0)
        
        grid = F.affine_grid(theta, image.unsqueeze(0).size(), align_corners=False)
        transformed = F.grid_sample(
            image.unsqueeze(0), grid, 
            mode=mode, 
            padding_mode='zeros',
            align_corners=False
        ).squeeze(0)
        
        # 分析数值范围
        new_range = (transformed.min().item(), transformed.max().item())
        
        results[mode] = {
            'output': transformed,
            'original_range': original_range,
            'new_range': new_range,
            'range_violation': new_range[0] < original_range[0] or new_range[1] > original_range[1],
            'min_violation': transformed.min().item() - original_range[0],
            'max_violation': transformed.max().item() - original_range[1]
        }
    
    return results
```

#### 边界处理的数值问题

边界处理策略影响数值稳定性：

1. **零填充（Zeros）**：边界外视为0，可能导致边缘值跳变
2. **常数填充（Constant）**：边界外为常数c，可能产生不连续
3. **复制填充（Replicate）**：使用边界像素值，保持连续性
4. **反射填充（Reflect）**：镜像反射，保持导数连续
5. **循环填充（Wrap）**：周期性边界，适合纹理数据

```python
def safe_augmentation_pipeline():
    """
    数值安全的数据增强流程。
    
    设计原则：
    1. 插值后钳制到有效范围
    2. 使用反射填充避免边缘效应
    3. 分阶段转换精度
    """
    import torchvision.transforms as T
    
    return T.Compose([
        # 几何变换（可能产生负值或 >255）
        T.RandomRotation(15, interpolation=T.InterpolationMode.BILINEAR),
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        
        # 数值安全：钳制到有效范围
        T.Lambda(lambda x: torch.clamp(x, 0, 255)),
        
        # 颜色变换（在钳制后进行）
        T.ColorJitter(brightness=0.2, contrast=0.2),
        
        # 再次钳制
        T.Lambda(lambda x: torch.clamp(x, 0, 255)),
        
        # 转为张量并归一化
        T.ToTensor(),  # 自动除以 255，转为 [0, 1]
    ])


def analyze_padding_effects(image, kernel_size=3):
    """
    分析不同边界填充策略的数值影响。
    
    填充策略对比：
    - zeros: 边缘值被拉向0，引入偏差
    - replicate: 保持边缘值，但可能扩展边界模式
    - reflect: 保持导数连续，适合自然图像
    - circular: 周期性假设，适合纹理
    
    Args:
        image: 输入图像
        kernel_size: 卷积核大小
        
    Returns:
        各填充策略的结果
    """
    results = {}
    
    for padding_mode in ['zeros', 'replicate', 'reflect', 'circular']:
        # 使用grid_sample测试不同填充
        # 创建一个稍微超出边界的grid
        theta = torch.eye(2, 3).unsqueeze(0)
        grid = F.affine_grid(theta, image.unsqueeze(0).size(), align_corners=False)
        
        # 稍微扩展grid以触发边界处理
        grid_ext = grid * 1.1  # 扩展10%
        
        output = F.grid_sample(
            image.unsqueeze(0), 
            grid_ext,
            mode='bilinear',
            padding_mode=padding_mode,
            align_corners=False
        ).squeeze(0)
        
        results[padding_mode] = {
            'output': output,
            'edge_mean': output[:, :, 0].mean().item(),  # 左边缘
            'edge_std': output[:, :, 0].std().item()
        }
    
    return results
```

### 2.2.2 Mixup/CutMix的数值特性

#### lambda采样的分布

**Mixup** 通过凸组合生成虚拟样本：

$$
\tilde{x} = \lambda x_i + (1-\lambda) x_j
$$

$$
\tilde{y} = \lambda y_i + (1-\lambda) y_j
$$

其中 $\lambda \sim \text{Beta}(\alpha, \alpha)$，$\alpha > 0$。

**Beta分布性质**：

Beta分布的概率密度函数：

$$
f(\lambda; \alpha, \beta) = \frac{\lambda^{\alpha-1}(1-\lambda)^{\beta-1}}{B(\alpha, \beta)}
$$

当 $\alpha = \beta$ 时，分布关于0.5对称。特别地：

- $\alpha = 1$：均匀分布 $\text{Uniform}(0, 1)$
- $\alpha < 1$：U型分布，倾向于0和1
- $\alpha > 1$：单峰分布，集中在0.5附近
- $\alpha \rightarrow \infty$：退化到 $\lambda = 0.5$

**数值考虑**：

当 $\alpha \ll 1$ 时，$\lambda$ 接近0或1的概率很高：

$$
P(\lambda < \epsilon) \approx \frac{\epsilon^{\alpha}}{\alpha \cdot B(\alpha, \alpha)}
$$

在FP16中，当 $\lambda < 6 \times 10^{-5}$ 时，$\lambda x$ 可能下溢为0。

```python
def mixup_data(x, y, alpha=0.4, eps=1e-6):
    """
    Mixup数据增强（数值安全版本）。
    
    数学原理：
    λ ~ Beta(α, α)
    x̃ = λ·x_i + (1-λ)·x_j
    ỹ = λ·y_i + (1-λ)·y_j
    
    数值考虑：
    1. 极小的α会导致λ接近0或1，混合效果减弱
    2. FP16中λx可能下溢
    3. 需要确保λ在(0,1)开区间内
    
    Args:
        x: 输入批次，形状(B, ...)
        y: 标签，形状(B, ...)
        alpha: Beta分布参数
        eps: 数值稳定常数
        
    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        # 从Beta分布采样
        lam = np.random.beta(alpha, alpha)
        
        # 数值安全：钳制到安全范围
        # 避免极端值导致FP16下溢
        lam = max(eps, min(lam, 1 - eps))
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    
    # 随机打乱索引
    index = torch.randperm(batch_size, device=x.device)
    
    # 混合输入
    # 使用FP32进行混合计算，再转回原始精度
    x_fp32 = x.float()
    mixed_x_fp32 = lam * x_fp32 + (1 - lam) * x_fp32[index]
    mixed_x = mixed_x_fp32.to(x.dtype)
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def analyze_mixup_distribution(alpha, n_samples=10000):
    """
    分析Mixup中lambda的分布特性。
    
    Args:
        alpha: Beta分布参数
        n_samples: 采样数量
        
    Returns:
        分布统计信息
    """
    samples = np.random.beta(alpha, alpha, n_samples)
    
    # 计算统计量
    mean = samples.mean()
    var = samples.var()
    
    # 理论值
    theoretical_mean = 0.5
    theoretical_var = 1 / (4 * (2 * alpha + 1))
    
    # FP16下溢风险
    fp16_min = 6e-5
    underflow_risk = np.mean(samples < fp16_min)
    
    # 接近边界的风险
    near_boundary = np.mean((samples < 0.01) | (samples > 0.99))
    
    return {
        'alpha': alpha,
        'empirical_mean': mean,
        'empirical_var': var,
        'theoretical_mean': theoretical_mean,
        'theoretical_var': theoretical_var,
        'fp16_underflow_risk': underflow_risk,
        'near_boundary_ratio': near_boundary
    }
```

#### 混合比例的数值精度

**CutMix** 通过空间混合实现：

$$
\tilde{x} = \mathbf{M} \odot x_i + (1 - \mathbf{M}) \odot x_j
$$

其中 $\mathbf{M}$ 是二进制掩码，混合比例由裁剪区域决定：

$$
\lambda = 1 - \frac{\text{area}(\mathbf{M})}{H \times W}
$$

**数值精度问题**：

1. 掩码生成涉及整数运算，需要小心舍入误差
2. 边界框坐标需要钳制到有效范围
3. 实际 $\lambda$ 与采样 $\lambda$ 可能不一致

```python
def cutmix_data(x, y, alpha=1.0, eps=1e-6):
    """
    CutMix数据增强（数值安全版本）。
    
    数学原理：
    1. 从Beta(α, α)采样λ
    2. 计算裁剪区域面积 = (1-λ) * H * W
    3. 随机选择中心点，生成裁剪框
    4. 混合：x̃ = M ⊙ x_i + (1-M) ⊙ x_j
    5. 调整λ以匹配实际裁剪比例
    
    数值考虑：
    1. 整数除法舍入误差
    2. 边界框越界检查
    3. 实际λ与采样λ的差异
    
    Args:
        x: 输入批次，形状(B, C, H, W)
        y: 标签，形状(B,)
        alpha: Beta分布参数
        eps: 数值稳定常数
        
    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(eps, min(lam, 1 - eps))
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    C, H, W = x.size(1), x.size(2), x.size(3)
    
    # 随机打乱
    index = torch.randperm(batch_size, device=x.device)
    
    # 计算裁剪尺寸
    # 目标面积比例 = 1 - λ
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # 随机中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # 计算边界框（数值安全）
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # 创建混合图像
    x_mixed = x.clone()
    x_mixed[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # 调整lambda以匹配实际裁剪比例
    # 这是CutMix的关键：使用实际面积比例
    actual_area = (bbx2 - bbx1) * (bby2 - bby1)
    total_area = H * W
    lam_adjusted = 1 - actual_area / total_area
    
    return x_mixed, y, y[index], lam_adjusted
```

### 2.2.3 增强管道的误差传播

数据增强管道中，每个操作的数值误差会累积传播。设第 $i$ 个操作的误差为 $\epsilon_i$，则总误差：

$$
\epsilon_{\text{total}} = \sum_{i=1}^{n} \epsilon_i \prod_{j=i+1}^{n} \left|\frac{\partial f_j}{\partial x}\right|
$$

对于线性操作，误差累积为：

$$
\text{Var}(\epsilon_{\text{total}}) = \sum_{i=1}^{n} \text{Var}(\epsilon_i)
$$

```python
class NumericallySafeAugmentation:
    """
    数值安全的数据增强管道。
    
    设计原则：
    1. 每个操作后钳制到有效范围
    2. 使用FP32进行中间计算
    3. 跟踪数值范围变化
    4. 提供误差估计
    """
    
    def __init__(self, operations, clamp_range=(0, 255)):
        """
        Args:
            operations: 增强操作列表
            clamp_range: 数值钳制范围
        """
        self.operations = operations
        self.clamp_range = clamp_range
        self.range_history = []
    
    def __call__(self, x):
        """
        应用增强管道。
        
        Args:
            x: 输入张量
            
        Returns:
            增强后的张量
        """
        current = x.float()  # 转为FP32
        self.range_history = [(current.min().item(), current.max().item())]
        
        for i, op in enumerate(self.operations):
            # 应用操作
            current = op(current)
            
            # 数值安全：钳制
            current = torch.clamp(current, self.clamp_range[0], self.clamp_range[1])
            
            # 记录范围
            self.range_history.append((current.min().item(), current.max().item()))
        
        return current
    
    def get_error_estimate(self):
        """
        估计数值误差传播。
        
        Returns:
            误差估计报告
        """
        if len(self.range_history) < 2:
            return None
        
        # 计算每步的范围变化
        range_changes = []
        for i in range(1, len(self.range_history)):
            prev_range = self.range_history[i-1]
            curr_range = self.range_history[i]
            
            # 范围扩展比例
            expansion = (curr_range[1] - curr_range[0]) / max(prev_range[1] - prev_range[0], 1e-8)
            range_changes.append(expansion)
        
        return {
            'range_history': self.range_history,
            'range_changes': range_changes,
            'total_expansion': np.prod(range_changes) if range_changes else 1.0
        }


def analyze_augmentation_error_propagation(pipeline, test_input, n_trials=100):
    """
    分析增强管道的误差传播。
    
    Args:
        pipeline: 增强管道
        test_input: 测试输入
        n_trials: 试验次数
        
    Returns:
        误差分析报告
    """
    outputs = []
    
    for _ in range(n_trials):
        output = pipeline(test_input)
        outputs.append(output)
    
    outputs = torch.stack(outputs)
    
    # 计算统计量
    mean_output = outputs.mean(dim=0)
    std_output = outputs.std(dim=0)
    
    # 估计数值噪声
    # 假设主要噪声来源是浮点舍入
    machine_epsilon = 1e-7 if outputs.dtype == torch.float32 else 1e-3
    estimated_noise = machine_epsilon * outputs.abs().mean()
    
    return {
        'mean': mean_output,
        'std': std_output,
        'cv': (std_output / mean_output.abs()).mean().item(),  # 变异系数
        'estimated_machine_noise': estimated_noise.item(),
        'signal_to_noise': (outputs.abs().mean() / estimated_noise).item()
    }
```

## 2.3 数据管道中的精度保持

### 2.3.1 数据类型转换的精度损失

#### uint8到float的转换误差

uint8表示范围 $[0, 255]$ 的整数，转换为float时：

$$
x_{\text{float}} = \frac{x_{\text{uint8}}}{255.0}
$$

**量化误差分析**：

uint8的量化步长为1，归一化后的步长为：

$$
\Delta = \frac{1}{255} \approx 0.00392
$$

转换到FP32时，由于FP32的精度远高于此（约7位有效数字），量化误差可忽略。

但转换到FP16时，需要关注：
- FP16的精度约为3位有效数字
- 在 $[0, 1]$ 范围内，FP16的步长约为 $2^{-10} \approx 0.001$
- 某些uint8值在FP16中无法精确表示

```python
def analyze_uint8_to_float_conversion():
    """
    分析uint8到浮点的转换误差。
    
    误差来源：
    1. 量化误差：uint8只有256个离散值
    2. 表示误差：FP16无法精确表示某些值
    3. 舍入误差：除法运算的舍入
    """
    # 所有可能的uint8值
    uint8_values = torch.arange(256, dtype=torch.uint8)
    
    # 转换为FP32
    fp32_values = uint8_values.float() / 255.0
    
    # 转换为FP16
    fp16_values = fp32_values.half()
    
    # 转换回FP32比较
    fp16_back_to_fp32 = fp16_values.float()
    
    # 计算误差
    conversion_error = (fp32_values - fp16_back_to_fp32).abs()
    
    return {
        'max_error': conversion_error.max().item(),
        'mean_error': conversion_error.mean().item(),
        'values_with_error': (conversion_error > 0).sum().item(),
        'error_distribution': conversion_error[conversion_error > 0]
    }


def safe_uint8_to_float(uint8_tensor, target_dtype=torch.float32):
    """
    安全的uint8到浮点转换。
    
    策略：
    1. 先扩展到FP32
    2. 归一化
    3. 再转到目标精度
    
    Args:
        uint8_tensor: uint8张量
        target_dtype: 目标浮点类型
        
    Returns:
        转换后的浮点张量
    """
    # 步骤1：转为FP32（避免溢出）
    fp32 = uint8_tensor.float()
    
    # 步骤2：归一化到[0, 1]
    normalized = fp32 / 255.0
    
    # 步骤3：转到目标精度
    if target_dtype == torch.float16:
        # 检查数值范围
        if normalized.max() > 65504 or normalized.min() < -65504:
            print("Warning: Values out of FP16 range")
        return normalized.half()
    elif target_dtype == torch.bfloat16:
        return normalized.bfloat16()
    else:
        return normalized
```

#### FP16转换的下溢风险

FP16的数值范围：
- 最小正规格数：$2^{-14} \approx 6.1 \times 10^{-5}$
- 最小次正规格数：$2^{-24} \approx 6.0 \times 10^{-8}$
- 最大有限值：$2^{16} \times (2 - 2^{-10}) \approx 65504$

下溢发生时，小于最小次正规格数的值变为0：

$$
x_{\text{FP16}} = \begin{cases}
0 & \text{if } |x| < 2^{-24} \\
\text{subnormal} & \text{if } 2^{-24} \leq |x| < 2^{-14} \\
\text{normal} & \text{if } 2^{-14} \leq |x| \leq 65504 \\
\text{inf} & \text{otherwise}
\end{cases}
$$

```python
def analyze_fp16_underflow_risk(tensor):
    """
    分析FP16转换的下溢风险。
    
    Args:
        tensor: 输入张量
        
    Returns:
        风险分析报告
    """
    # FP16阈值
    FP16_MIN_NORMAL = 6.1035e-5  # 2^-14
    FP16_MIN_SUBNORMAL = 5.96e-8  # 2^-24
    
    abs_tensor = tensor.abs()
    
    # 统计各类数值
    total_elements = tensor.numel()
    
    underflow_to_zero = (abs_tensor < FP16_MIN_SUBNORMAL).sum().item()
    subnormal_range = ((abs_tensor >= FP16_MIN_SUBNORMAL) & 
                       (abs_tensor < FP16_MIN_NORMAL)).sum().item()
    normal_range = ((abs_tensor >= FP16_MIN_NORMAL) & 
                    (abs_tensor <= 65504)).sum().item()
    overflow = (abs_tensor > 65504).sum().item()
    
    return {
        'total_elements': total_elements,
        'underflow_to_zero': underflow_to_zero,
        'underflow_ratio': underflow_to_zero / total_elements,
        'subnormal_range': subnormal_range,
        'subnormal_ratio': subnormal_range / total_elements,
        'normal_range': normal_range,
        'normal_ratio': normal_range / total_elements,
        'overflow': overflow,
        'risk_level': 'high' if underflow_to_zero / total_elements > 0.01 else 
                      'medium' if underflow_to_zero / total_elements > 0.001 else 'low'
    }


def safe_fp16_conversion(tensor, scale_factor=None):
    """
    安全的FP16转换。
    
    策略：
    1. 检测下溢风险
    2. 如有必要，先缩放再转换
    3. 记录缩放因子以便还原
    
    Args:
        tensor: 输入张量
        scale_factor: 预定义的缩放因子
        
    Returns:
        转换后的张量和元信息
    """
    risk = analyze_fp16_underflow_risk(tensor)
    
    if risk['risk_level'] == 'low' and scale_factor is None:
        # 安全，直接转换
        return tensor.half(), {'scale_factor': 1.0, 'original_risk': risk}
    
    # 需要缩放
    if scale_factor is None:
        # 自动计算缩放因子
        max_val = tensor.abs().max()
        target_max = 10000.0  # 留有余量
        scale_factor = target_max / max_val if max_val > 0 else 1.0
    
    # 缩放、转换、记录
    scaled = tensor * scale_factor
    converted = scaled.half()
    
    return converted, {
        'scale_factor': scale_factor,
        'original_risk': risk,
        'scaled_max': scaled.abs().max().item()
    }
```

### 2.3.2 DataLoader的内存对齐

#### batch内数据布局的数值影响

DataLoader的`collate_fn`决定batch数据的内存布局。不合适的布局可能导致：

1. **非对齐访问**：性能下降，某些硬件上可能出错
2. **内存碎片**：GPU内存利用率降低
3. **类型不一致**：混合精度训练失败

**最佳实践**：

```python
from torch.utils.data import DataLoader
import torch

def fp32_collate_fn(batch):
    """
    确保batch在FP32中的collate函数。
    
    设计考虑：
    1. 默认collate_fn保持原始类型
    2. uint8图像需要显式转FP32
    3. 标签保持long类型
    
    Args:
        batch: 样本列表，每个样本是(data, label)元组
        
    Returns:
        batched_data, batched_labels
    """
    data, labels = torch.utils.data.default_collate(batch)
    
    # 确保数据在FP32
    if data.dtype == torch.uint8:
        data = data.float() / 255.0
    elif data.dtype != torch.float32:
        data = data.float()
    
    return data, labels


def memory_efficient_collate_fn(batch):
    """
    内存高效的collate函数。
    
    优化策略：
    1. 预分配张量，避免重复分配
    2. 使用pin_memory加速CPU到GPU传输
    3. 对齐内存布局（NCHW格式）
    """
    # 分离数据和标签
    data_list = [item[0] for item in batch]
    label_list = [item[1] for item in batch]
    
    batch_size = len(batch)
    
    # 获取形状
    C, H, W = data_list[0].shape
    
    # 预分配连续内存
    data_batch = torch.zeros(batch_size, C, H, W, dtype=torch.float32)
    labels_batch = torch.zeros(batch_size, dtype=torch.long)
    
    # 填充数据
    for i, (data, label) in enumerate(zip(data_list, label_list)):
        data_batch[i] = data.float() / 255.0 if data.dtype == torch.uint8 else data
        labels_batch[i] = label
    
    return data_batch, labels_batch


# 使用示例
dataloader = DataLoader(
    dataset, 
    batch_size=32,
    collate_fn=fp32_collate_fn,
    pin_memory=True,  # 加速CPU到GPU传输
    num_workers=4
)
```

#### 多进程的数据一致性

多进程数据加载时，每个worker有独立的随机状态。需要确保：

1. **可复现性**：设置随机种子
2. **独立性**：避免worker间相关
3. **同步**：某些增强需要全局状态

```python
def worker_init_fn(worker_id):
    """
    初始化worker的随机状态。
    
    策略：
    1. 基于worker_id设置不同种子
    2. 确保不同epoch的随机性
    3. 保持可复现性
    
    Args:
        worker_id: worker标识符
    """
    # 获取基础种子
    base_seed = torch.initial_seed() % (2**32)
    
    # 每个worker使用不同种子
    np.random.seed(base_seed + worker_id)
    
    # 设置Python random模块种子
    import random
    random.seed(base_seed + worker_id)


# 使用示例
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    worker_init_fn=worker_init_fn,
    generator=torch.Generator().manual_seed(42)  # 可复现
)


class ReproducibleDataLoader:
    """
    可复现的数据加载器包装。
    
    功能：
    1. 固定随机种子
    2. 记录数据顺序
    3. 支持断点续训
    """
    
    def __init__(self, dataset, batch_size, num_workers=0, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.epoch = 0
        
    def __iter__(self):
        # 每个epoch使用不同但确定的种子
        epoch_seed = self.seed + self.epoch
        
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=lambda worker_id: np.random.seed(epoch_seed + worker_id),
            generator=torch.Generator().manual_seed(epoch_seed),
            shuffle=True
        )
        
        self.epoch += 1
        return iter(dataloader)
    
    def set_epoch(self, epoch):
        """设置当前epoch，用于恢复训练"""
        self.epoch = epoch
```

### 2.3.3 预加载数据的精度选择

```python
class SafeDataset(torch.utils.data.Dataset):
    """
    数值安全的数据集类。
    
    设计原则：
    1. 以低精度存储（节省内存）
    2. 高精度处理（数值稳定）
    3. 延迟转换（按需加载）
    
    支持：
    - 内存映射（大文件）
    - 延迟精度转换
    - 缓存机制
    """
    
    def __init__(self, data_path, dtype=torch.float32, use_mmap=True):
        """
        Args:
            data_path: 数据文件路径
            dtype: 处理时的目标精度
            use_mmap: 是否使用内存映射
        """
        # 以内存映射模式加载，节省RAM
        mmap_mode = 'r' if use_mmap else None
        self.data = np.load(data_path, mmap_mode=mmap_mode)
        self.dtype = dtype
        self.storage_dtype = self.data.dtype
        
    def __getitem__(self, idx):
        """
        获取样本。
        
        流程：
        1. 从存储读取（可能uint8或float16）
        2. 转为FP32进行处理
        3. 按需归一化
        """
        x = self.data[idx]
        
        # 转为张量并提升精度
        tensor = torch.from_numpy(x)
        
        # 数值安全转换
        if self.storage_dtype == np.uint8:
            # uint8 -> FP32 -> [0,1]
            tensor = tensor.float() / 255.0
        elif self.storage_dtype == np.float16:
            # FP16 -> FP32
            tensor = tensor.float()
        else:
            tensor = tensor.to(self.dtype)
        
        return tensor.to(self.dtype)
    
    def __len__(self):
        return len(self.data)


class StreamingDataset(torch.utils.data.Dataset):
    """
    流式数据集，适合超大数据集。
    
    特点：
    1. 按需从磁盘读取
    2. 预取和缓存
    3. 数值转换流水线
    """
    
    def __init__(self, file_list, transform=None, cache_size=1000):
        self.file_list = file_list
        self.transform = transform
        self.cache = {}
        self.cache_size = cache_size
        self.cache_order = []
        
    def __getitem__(self, idx):
        # 检查缓存
        if idx in self.cache:
            return self.cache[idx]
        
        # 从文件加载
        data = self._load_file(self.file_list[idx])
        
        # 应用变换（包括数值转换）
        if self.transform:
            data = self.transform(data)
        
        # 更新缓存
        self._update_cache(idx, data)
        
        return data
    
    def _load_file(self, path):
        """加载单个文件，确保数值安全"""
        data = np.load(path)
        
        # 统一转为FP32
        if data.dtype == np.uint8:
            return data.astype(np.float32) / 255.0
        return data.astype(np.float32)
    
    def _update_cache(self, idx, data):
        """更新LRU缓存"""
        if len(self.cache) >= self.cache_size:
            # 移除最旧的
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
        
        self.cache[idx] = data
        self.cache_order.append(idx)
    
    def __len__(self):
        return len(self.file_list)
```

## 2.4 特定数据类型的数值问题

### 2.4.1 图像数据的数值特性

#### [0,255]到[0,1]的映射精度

图像数据通常以uint8存储，范围 $[0, 255]$。归一化到 $[0, 1]$：

$$
x_{\text{normalized}} = \frac{x}{255}
$$

**精度分析**：

uint8有256个离散值，归一化后的离散值为：

$$
\left\{0, \frac{1}{255}, \frac{2}{255}, \ldots, 1\right\}
$$

相邻值的间隔：

$$
\Delta = \frac{1}{255} \approx 0.00392
$$

在FP32中，这些值可以精确表示（FP32有约7位十进制精度）。

在FP16中，需要验证表示精度：

```python
def analyze_image_normalization_precision():
    """
    分析图像归一化的精度。
    
    分析内容：
    1. uint8到FP32的精确表示
    2. uint8到FP16的表示误差
    3. 反归一化的精度损失
    """
    # 所有uint8值
    uint8_vals = torch.arange(256, dtype=torch.uint8)
    
    # 归一化到[0,1]
    fp32_normalized = uint8_vals.float() / 255.0
    
    # 转为FP16
    fp16_normalized = fp32_normalized.half()
    fp16_back = fp16_normalized.float()
    
    # 反归一化
    fp32_denorm = (fp32_normalized * 255).round().to(torch.uint8)
    fp16_denorm = (fp16_back * 255).round().to(torch.uint8)
    
    # 计算反归一化误差
    reconstruction_error = (fp32_denorm != fp16_denorm).sum().item()
    
    # 最大误差
    max_diff = (fp32_normalized - fp16_back).abs().max().item()
    
    return {
        'reconstruction_error_count': reconstruction_error,
        'reconstruction_error_rate': reconstruction_error / 256,
        'max_representation_error': max_diff,
        'max_pixel_error': (max_diff * 255)
    }


class ImagePreprocessor:
    """
    图像预处理数值安全类。
    
    提供：
    1. 安全的类型转换
    2. 归一化/反归一化
    3. 标准化（ImageNet统计）
    """
    
    # ImageNet统计量
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
    
    @staticmethod
    def to_float(image):
        """
        uint8 [0,255] -> float32 [0, 1]
        
        数值安全：
        1. 先转float避免溢出
        2. 再除以255
        """
        if image.dtype == torch.uint8:
            return image.float() / 255.0
        return image.float()
    
    @staticmethod
    def to_uint8(image):
        """
        float [0,1] -> uint8 [0, 255]
        
        数值安全：
        1. 钳制到[0,1]避免溢出
        2. 乘以255
        3. 四舍五入
        4. 转为uint8
        """
        image = torch.clamp(image, 0.0, 1.0)
        return (image * 255).round().to(torch.uint8)
    
    @staticmethod
    def normalize(image, mean=None, std=None):
        """
        标准化，防止除零。
        
        x' = (x - mean) / (std + eps)
        """
        if mean is None:
            mean = ImagePreprocessor.IMAGENET_MEAN
        if std is None:
            std = ImagePreprocessor.IMAGENET_STD
            
        # 确保维度匹配
        if mean.dim() == 1:
            mean = mean.view(-1, 1, 1)
        if std.dim() == 1:
            std = std.view(-1, 1, 1)
        
        # 数值安全：防止除零
        eps = 1e-8
        std_safe = torch.where(std < eps, torch.ones_like(std), std)
        
        return (image - mean) / std_safe
    
    @staticmethod
    def denormalize(image, mean=None, std=None):
        """反标准化"""
        if mean is None:
            mean = ImagePreprocessor.IMAGENET_MEAN
        if std is None:
            std = ImagePreprocessor.IMAGENET_STD
            
        if mean.dim() == 1:
            mean = mean.view(-1, 1, 1)
        if std.dim() == 1:
            std = std.view(-1, 1, 1)
        
        return image * std + mean
```

### 2.4.2 时序数据的数值稳定性

#### 归一化窗口的选择

时序数据的归一化需要考虑时间依赖性。常用方法：

1. **全局归一化**：使用整个序列的统计量
2. **滑动窗口归一化**：使用局部窗口统计量
3. **指数加权归一化**：给予近期数据更高权重

**数学形式**：

全局Z-score：

$$
x'_t = \frac{x_t - \mu_{\text{global}}}{\sigma_{\text{global}}}
$$

滑动窗口Z-score（窗口大小 $w$）：

$$
\mu_t = \frac{1}{w}\sum_{i=t-w+1}^{t} x_i, \quad x'_t = \frac{x_t - \mu_t}{\sigma_t}
$$

指数加权：

$$
\mu_t = \frac{\sum_{i=1}^{t} \alpha^{t-i} x_i}{\sum_{i=1}^{t} \alpha^{t-i}}, \quad \alpha \in (0, 1)
$$

```python
def normalize_timeseries(data, method='zscore', window=None, alpha=0.9):
    """
    时序数据标准化。
    
    方法对比：
    1. zscore（全局）：适合平稳序列
    2. minmax（全局）：保留原始范围信息
    3. windowed：适合非平稳序列，但引入延迟
    4. exponential：适合在线学习，权重可调
    
    Args:
        data: 时序数据，形状(..., T)
        method: 标准化方法
        window: 滑动窗口大小
        alpha: 指数加权系数
        
    Returns:
        标准化后的数据
    """
    eps = 1e-8
    
    if method == 'zscore':
        # 全局Z-score
        mean = data.mean(dim=-1, keepdim=True)
        std = data.std(dim=-1, keepdim=True)
        std = torch.clamp(std, min=eps)
        return (data - mean) / std
    
    elif method == 'minmax':
        # 全局Min-Max
        min_val = data.min(dim=-1, keepdim=True)[0]
        max_val = data.max(dim=-1, keepdim=True)[0]
        range_val = torch.clamp(max_val - min_val, min=eps)
        return (data - min_val) / range_val
    
    elif method == 'windowed':
        # 滑动窗口Z-score
        if window is None:
            window = min(100, data.shape[-1] // 10)
        
        # 使用卷积计算滑动均值
        kernel = torch.ones(window) / window
        kernel = kernel.to(data.device).view(1, 1, -1)
        
        # 扩展维度以使用conv1d
        original_shape = data.shape
        data_flat = data.reshape(-1, 1, original_shape[-1])
        
        # 计算滑动均值（使用反射填充）
        mean_flat = F.conv1d(
            F.pad(data_flat, (window-1, 0), mode='reflect'),
            kernel,
            padding=0
        )
        
        # 计算滑动方差
        mean_sq_flat = F.conv1d(
            F.pad(data_flat**2, (window-1, 0), mode='reflect'),
            kernel,
            padding=0
        )
        var_flat = mean_sq_flat - mean_flat**2
        std_flat = torch.sqrt(var_flat + eps)
        
        # 标准化
        normalized_flat = (data_flat - mean_flat) / std_flat
        
        return normalized_flat.reshape(original_shape)
    
    elif method == 'exponential':
        # 指数加权移动平均
        weights = torch.tensor([alpha**(data.shape[-1]-1-i) for i in range(data.shape[-1])])
        weights = weights / weights.sum()
        weights = weights.to(data.device)
        
        # 加权均值
        for _ in range(data.dim() - 1):
            weights = weights.unsqueeze(0)
        
        mean = (data * weights).sum(dim=-1, keepdim=True)
        
        # 加权方差（无偏估计）
        var = ((data - mean)**2 * weights).sum(dim=-1, keepdim=True)
        std = torch.sqrt(var + eps)
        
        return (data - mean) / std
    
    else:
        raise ValueError(f"Unknown method: {method}")


def robust_timeseries_normalize(data, quantile_low=0.05, quantile_high=0.95):
    """
    鲁棒的时序归一化。
    
    使用分位数代替min/max，对异常值更鲁棒。
    
    Args:
        data: 时序数据
        quantile_low: 下分位数
        quantile_high: 上分位数
        
    Returns:
        归一化后的数据
    """
    eps = 1e-8
    
    # 计算分位数
    q_low = torch.quantile(data, quantile_low, dim=-1, keepdim=True)
    q_high = torch.quantile(data, quantile_high, dim=-1, keepdim=True)
    
    # 鲁棒范围
    robust_range = q_high - q_low
    robust_range = torch.clamp(robust_range, min=eps)
    
    # 归一化
    normalized = (data - q_low) / robust_range
    
    # 钳制到合理范围
    return torch.clamp(normalized, -1.0, 2.0)  # 允许一些超出[0,1]的值
```

### 2.4.3 文本/嵌入数据

#### 索引查找的数值问题

文本处理中的主要数值问题：

1. **索引越界**：访问超出词表范围的嵌入
2. **Padding处理**：填充值的数值影响
3. **嵌入初始化**：初始值的数值范围

```python
import torch.nn as nn

def safe_embedding_lookup(embeddings, indices, padding_idx=None, max_norm=None):
    """
    安全的嵌入查找。
    
    数值安全考虑：
    1. 索引越界检查
    2. Padding处理（梯度屏蔽）
    3. 嵌入归一化
    
    Args:
        embeddings: 嵌入层或嵌入张量
        indices: 索引张量
        padding_idx: 填充索引
        max_norm: 最大范数（用于归一化）
        
    Returns:
        嵌入向量
    """
    if isinstance(embeddings, nn.Embedding):
        num_embeddings = embeddings.num_embeddings
        embedding_dim = embeddings.embedding_dim
    else:
        num_embeddings = embeddings.size(0)
        embedding_dim = embeddings.size(1)
    
    # 索引越界检查
    if indices.max() >= num_embeddings:
        raise ValueError(f"Index {indices.max()} out of range [0, {num_embeddings})")
    if indices.min() < 0:
        raise ValueError(f"Negative index {indices.min()} not allowed")
    
    # 执行查找
    if isinstance(embeddings, nn.Embedding):
        output = embeddings(indices)
    else:
        output = embeddings[indices]
    
    # 处理padding（数值安全：确保padding位置为0）
    if padding_idx is not None:
        mask = (indices == padding_idx).unsqueeze(-1).expand_as(output)
        output = output.masked_fill(mask, 0.0)
    
    # 可选：最大范数约束
    if max_norm is not None:
        norms = output.norm(dim=-1, keepdim=True)
        scale = torch.clamp(max_norm / norms, max=1.0)
        output = output * scale
    
    return output


class SafeEmbedding(nn.Module):
    """
    数值安全的嵌入层。
    
    特性：
    1. 合理的初始化范围
    2. 索引验证
    3. 梯度屏蔽
    """
    
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, 
                 max_norm=None, scale=0.02):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        
        # 数值安全的初始化
        # 使用较小的标准差，避免初始值过大
        self.reset_parameters(scale)
    
    def reset_parameters(self, scale=0.02):
        """重置参数，使用小范围均匀分布"""
        nn.init.uniform_(self.weight, -scale, scale)
        
        # 如果指定了padding_idx，将该行初始化为0
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)
    
    def forward(self, indices):
        """前向传播，包含安全检查"""
        # 调试模式下检查索引
        if torch.is_grad_enabled() and torch.any(indices >= self.num_embeddings):
            invalid = indices[indices >= self.num_embeddings]
            raise RuntimeError(f"Invalid indices: {invalid}")
        
        # 查找嵌入
        output = F.embedding(
            indices, self.weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm
        )
        
        return output


def analyze_embedding_statistics(embedding_layer, sample_indices=None):
    """
    分析嵌入层的数值统计。
    
    Args:
        embedding_layer: 嵌入层
        sample_indices: 采样索引（None表示全部）
        
    Returns:
        统计报告
    """
    weight = embedding_layer.weight.data
    
    if sample_indices is not None:
        weight = weight[sample_indices]
    
    # 基本统计
    stats = {
        'shape': list(weight.shape),
        'mean': weight.mean().item(),
        'std': weight.std().item(),
        'min': weight.min().item(),
        'max': weight.max().item(),
        'l2_norm_mean': weight.norm(dim=-1).mean().item(),
        'l2_norm_std': weight.norm(dim=-1).std().item(),
    }
    
    # 检查数值异常
    stats['has_nan'] = torch.isnan(weight).any().item()
    stats['has_inf'] = torch.isinf(weight).any().item()
    
    # 检查极端值（可能是异常）
    extreme_threshold = 10.0
    stats['extreme_values'] = (weight.abs() > extreme_threshold).sum().item()
    
    return stats
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

**关键要点**：

1. **标准化数值稳定性**：使用Welford算法进行在线均值方差计算，添加epsilon防止除零，考虑鲁棒统计量（IQR）替代经典统计量。

2. **数据增强数值边界**：理解插值算法的数值特性（双三次可能超范围），Mixup的Beta分布参数影响，CutMix的实际lambda调整。

3. **精度保持策略**：uint8到float的转换顺序，FP16下溢风险检测，DataLoader的内存对齐和类型一致性。

4. **特定数据类型**：图像数据的归一化精度分析，时序数据的窗口选择策略，文本嵌入的索引安全和初始化。

---

**上一章**: [第1章 引言](./01-introduction.md) | **下一章**: [第3章 模型架构稳定性](./03-architecture-stability.md)
