# 第7章 分布式训练稳定性

分布式训练引入了通信和同步相关的数值稳定性问题。本章聚焦 All-reduce、ZeRO、流水线并行等场景下的数值保障。

## 7.1 梯度同步的数值误差累积

### 7.1.1 All-reduce 的数值误差

**问题描述：**

Ring All-reduce 的累加顺序不确定，导致浮点累加误差不可复现：

```
正确: (a + b) + c = 1.0000000 + 2.0000000 + 3.0000000 = 6.0000000
实际: a + (b + c) = 1.0000000 + (2.0000000 + 3.0000000) = 5.9999999
```

**高精度通信方案：**

```python
def accurate_all_reduce(tensor):
    """在 FP32 中进行 All-reduce 减少精度损失"""
    tensor_fp32 = tensor.float()
    dist.all_reduce(tensor_fp32)
    tensor.copy_(tensor_fp32)
```

**BFloat16 特殊处理：**

```python
def bf16_stable_all_reduce(tensor):
    """BF16 梯度同步，使用补偿累加"""
    # BF16 范围大但精度低，需要特殊处理
    if tensor.dtype == torch.bfloat16:
        # 转换为 FP32 通信
        tensor_fp32 = tensor.float()
        dist.all_reduce(tensor_fp32)
        # 随机舍入回 BF16
        tensor.copy_(stochastic_round_to_bf16(tensor_fp32))
    else:
        dist.all_reduce(tensor)
```

### 7.1.2 高精度中间累加

**误差反馈 All-reduce：**

```python
class AccurateGradientSync:
    """带误差补偿的梯度同步
    
    核心思想：将本轮压缩/量化误差累积到下一轮补偿
    """
    def __init__(self, model, world_size):
        self.model = model
        self.world_size = world_size
        self.error_buffers = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.error_buffers[name] = torch.zeros_like(param)
    
    def sync_gradients(self):
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            
            # 添加累积误差
            param.grad.add_(self.error_buffers[name])
            
            # FP32 通信
            grad_fp32 = param.grad.float()
            dist.all_reduce(grad_fp32)
            grad_fp32.div_(self.world_size)
            
            # 计算新的量化误差
            param.grad.copy_(grad_fp32)
            self.error_buffers[name] = grad_fp32 - param.grad.float()
```

**分层累加策略：**

```python
def hierarchical_all_reduce(tensor, group_sizes=[8, 4]):
    """分层 All-reduce 减少跨节点误差
    
    先在节点内累加（低延迟），再跨节点聚合
    """
    # 节点内 reduce
    intra_node_tensor = tensor.clone()
    dist.all_reduce(intra_node_tensor, group=intra_node_group)
    
    # 跨节点 reduce
    dist.all_reduce(intra_node_tensor, group=cross_node_group)
    
    return intra_node_tensor
```

## 7.2 ZeRO 优化器的精度保持

### 7.2.1 ZeRO 阶段 1/2/3

**各阶段数值特性对比：**

| Stage | 分片内容 | 内存节省 | 数值风险 | 适用场景 |
|-------|---------|---------|---------|---------|
| Stage 1 | 优化器状态 | 4× | 低 | 大多数场景 |
| Stage 2 | 优化器状态 + 梯度 | 8× | 中 | 大模型训练 |
| Stage 3 | 参数 + 梯度 + 优化器状态 | 与数据并行度线性相关 | 高 | 超大模型 |

**实现要点：**

```python
# DeepSpeed ZeRO 配置要点
deepspeed_config = {
    "zero_optimization": {
        "stage": 2,
        "offload_param": {"device": "cpu", "pin_memory": True},
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "round_robin_gradients": True,  # 改善内存碎片
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,  # 动态损失缩放
        "initial_scale_power": 16
    }
}
```

**机构**: [Rajbhandari et al. 2021] — Microsoft Research  
**信源**: [ZeRO-Infinity](https://arxiv.org/abs/2104.07857)

### 7.2.2 优化器状态的通信精度

**CPU/NVMe Offloading 数值保障：**

```python
class StableOffloading:
    """确保 offloading 过程中的数值稳定性"""
    
    def __init__(self, device='cpu', pin_memory=True):
        self.device = device
        self.pin_memory = pin_memory
    
    def offload_optimizer_state(self, state_dict):
        """Offload 时保持 FP32"""
        offloaded = {}
        for key, tensor in state_dict.items():
            # 始终使用 FP32 存储
            if tensor.dtype in [torch.float16, torch.bfloat16]:
                tensor = tensor.float()
            offloaded[key] = tensor.to(
                self.device, 
                pin_memory=self.pin_memory if self.device == 'cpu' else False
            )
        return offloaded
    
    def load_optimizer_state(self, offloaded_dict):
        """加载时恢复原始精度"""
        loaded = {}
        for key, tensor in offloaded_dict.items():
            loaded[key] = tensor.cuda(non_blocking=True)
        return loaded
```

**NVMe 卸载的特殊考虑：**

```python
def nvme_offload_config():
    """NVMe 卸载的数值稳定性配置"""
    return {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "nvme",
                "nvme_path": "/nvme",
                "buffer_count": 4,
                "pin_memory": True  # 确保传输精度
            },
            "offload_param": {
                "device": "nvme",
                "nvme_path": "/nvme"
            }
        },
        # 关键：保持主权重 FP32
        "fp32_allreduce": True
    }
```

## 7.3 流水线并行稳定性

### 7.3.1 激活检查点的精度

**问题描述：**

流水线并行的激活检查点（activation checkpointing）在重计算时可能产生数值差异：

```python
from torch.utils.checkpoint import checkpoint

def pipeline_stage_forward(self, x):
    """带精度保障的激活检查点"""
    if self.training:
        # use_reentrant=False 更稳定
        return checkpoint(
            self.layers, x, 
            use_reentrant=False,
            preserve_rng_state=True  # 保持 dropout 等状态
        )
    else:
        return self.layers(x)
```

**微批次间的一致性：**

```python
class StablePipelineStage(nn.Module):
    """确保流水线各微批次数值一致性"""
    
    def __init__(self, layers, num_microbatches):
        super().__init__()
        self.layers = layers
        self.num_microbatches = num_microbatches
        self.activation_buffers = {}
    
    def forward(self, x, microbatch_id):
        # 保存 RNG 状态确保重计算一致性
        rng_state = torch.get_rng_state()
        
        if self.training and microbatch_id in self.activation_buffers:
            # 重计算路径
            x = self.activation_buffers[microbatch_id]
            with torch.random.fork_rng():
                torch.set_rng_state(rng_state)
                output = self.layers(x)
        else:
            # 前向路径
            output = self.layers(x)
            if self.training:
                self.activation_buffers[microbatch_id] = x.detach()
        
        return output
```

### 7.3.2 气泡填充与数值稳定性

```python
def stable_pipeline_schedule(microbatches, num_stages):
    """稳定的流水线调度策略
    
    减少流水线气泡对数值的影响
    """
    schedule = []
    
    # GPipe 风格：前向全完成再反向
    # 但可能导致激活内存峰值
    
    # PipeDream-Flush 风格：交替进行，更稳定
    for step in range(len(microbatches) + num_stages - 1):
        stage_id = step % num_stages
        microbatch_id = step // num_stages
        
        if microbatch_id < len(microbatches):
            schedule.append(('F', stage_id, microbatch_id))
    
    # 反向阶段
    for step in range(len(microbatches) + num_stages - 1):
        stage_id = num_stages - 1 - (step % num_stages)
        microbatch_id = len(microbatches) - 1 - (step // num_stages)
        
        if 0 <= microbatch_id < len(microbatches):
            schedule.append(('B', stage_id, microbatch_id))
    
    return schedule
```

## 7.4 张量并行的数值考虑

### 7.4.1 All-gather 与 Reduce-scatter

```python
class TensorParallelLinear(nn.Module):
    """张量并行线性层的数值稳定实现"""
    
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.world_size = world_size
        # 每个 rank 只保存部分权重
        self.weight = nn.Parameter(
            torch.empty(out_features // world_size, in_features)
        )
    
    def forward(self, x):
        # 本地计算
        local_output = F.linear(x, self.weight)
        
        # All-gather 收集结果
        # 使用 FP32 累加减少误差
        if self.training:
            local_output_fp32 = local_output.float()
            gathered = self.all_gather_fp32(local_output_fp32)
            output = gathered.to(local_output.dtype)
        else:
            output = self.all_gather(local_output)
        
        return output
    
    def all_gather_fp32(self, tensor):
        """FP32 All-gather 保障精度"""
        world_size = self.world_size
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=-1)
```

## 7.5 本章小结

| 主题 | 关键建议 |
|------|----------|
| All-reduce | 在 FP32 中进行通信，或使用误差补偿 |
| ZeRO | Stage 2 最常用，Stage 3 注意参数更新精度 |
| 流水线并行 | 使用激活检查点，注意 RNG 状态一致性 |
| 张量并行 | All-gather/Reduce-scatter 使用 FP32 |

**分布式训练数值检查清单：**

- [ ] All-reduce 在 FP32 中进行
- [ ] ZeRO-3 参数更新后检查 NaN
- [ ] 流水线并行激活检查点使用 use_reentrant=False
- [ ] 多节点训练启用 hierarchical all-reduce
- [ ] 混合精度训练保持主权重 FP32

---

**上一章**: [第6章 优化器稳定性](./06-optimizer-stability.md) | **下一章**: [第8章 调试与诊断方法论](./08-debugging-methodology.md)
