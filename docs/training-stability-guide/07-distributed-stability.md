# 第7章 分布式训练稳定性

分布式训练引入了通信和同步相关的数值稳定性问题。

## 7.1 梯度同步的数值误差累积

### 7.1.1 All-reduce 的数值误差

```python
def accurate_all_reduce(tensor):
    tensor_fp32 = tensor.float()
    dist.all_reduce(tensor_fp32)
    tensor.copy_(tensor_fp32)
```

### 7.1.2 高精度中间累加

```python
class AccurateGradientSync:
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
            param.grad.add_(self.error_buffers[name])
            grad_fp32 = param.grad.float()
            dist.all_reduce(grad_fp32)
            grad_fp32.div_(self.world_size)
            param.grad.copy_(grad_fp32)
            self.error_buffers[name] = grad_fp32 - param.grad.float()
```

## 7.2 ZeRO 优化器的精度保持

### 7.2.1 ZeRO 阶段 1/2/3

- **Stage 1**: 优化器状态分片
- **Stage 2**: 梯度分片
- **Stage 3**: 参数分片

**机构**: [Rajbhandari et al. 2021] — Microsoft Research  
**信源**: [ZeRO-Infinity](https://arxiv.org/abs/2104.07857)

### 7.2.2 优化器状态的通信精度

```python
deepspeed_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_param": {"device": "cpu", "pin_memory": True},
        "offload_optimizer": {"device": "cpu", "pin_memory": True}
    },
    "fp16": {"enabled": True, "loss_scale": 0, "initial_scale_power": 16}
}
```

## 7.3 流水线并行稳定性

### 7.3.1 激活检查点的精度

```python
from torch.utils.checkpoint import checkpoint

def pipeline_stage_forward(self, x):
    if self.training:
        return checkpoint(self.layers, x, use_reentrant=False)
    else:
        return self.layers(x)
```

## 7.4 本章小结

| 主题 | 关键建议 |
|------|----------|
| All-reduce | 在 FP32 中进行通信，或使用误差补偿 |
| ZeRO | 优化器状态分片，注意 CPU/NVMe offloading 精度 |
| 流水线并行 | 使用激活检查点，注意微批次间一致性 |

---

**上一章**: [第6章 优化器稳定性](./06-optimizer-stability.md) | **下一章**: [第8章 调试与诊断方法论](./08-debugging-methodology.md)
