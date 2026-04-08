# 第9章 实践检查清单

可执行、可验证的检查清单，确保训练稳定性。

## 9.1 训练前检查清单

### 数据准备

- [ ] 数据已完成标准化/归一化，且添加了 epsilon 防止除零
- [ ] 异常值已检测并处理（Z-score > 3 或 IQR 范围外）
- [ ] 数据类型转换安全（先转 FP32 再处理）
- [ ] DataLoader 使用自定义 collate_fn 确保 batch 在 FP32 中
- [ ] 数据增强不会导致数值溢出（已添加 clamp）

### 模型准备

- [ ] 权重初始化根据激活函数选择（ReLU 用 Kaiming，Tanh 用 Xavier）
- [ ] 低精度训练时限制初始化标准差（max_std=0.01 for FP16）
- [ ] 归一化层 epsilon 已调整（FP16 建议 eps=1e-3）
- [ ] 使用 Pre-LN 结构（Transformer 类模型）
- [ ] 激活函数数值安全（GELU 在 FP32 计算）

### 训练配置

- [ ] 混合精度配置正确（AMP + GradScaler for FP16）
- [ ] 损失缩放初始值合理（通常 2^16）
- [ ] 梯度裁剪阈值设置（max_norm=1.0）
- [ ] 学习率 warmup 步数设置（通常占总步数的 1-10%）
- [ ] 数值异常监控钩子已安装

## 9.2 低精度训练专项清单

### BF16 训练

- [ ] 硬件支持 BF16（A100/H100/TPU 等）
- [ ] PyTorch 版本 >= 1.10
- [ ] autocast(dtype=torch.bfloat16) 配置正确

### FP8 训练

- [ ] 硬件支持 FP8（H100/H200/B200）
- [ ] 安装了 Transformer Engine
- [ ] 缩放因子策略配置（Delayed vs Current Scaling）
- [ ] 敏感层精度回退（Embedding、Norm、Attention、输出层保持 BF16）

### 混合精度检查

- [ ] Master weights 保持 FP32
- [ ] 损失计算在 FP32
- [ ] 梯度缩放正确应用
- [ ] 已启用 GradScaler.update()

## 9.3 分布式训练清单

### 数据并行

- [ ] DistributedSampler 正确使用
- [ ] 梯度同步在 FP32
- [ ] All-reduce 后梯度除以 world_size

### ZeRO 配置

- [ ] ZeRO stage 选择正确（2/3）
- [ ] 优化器状态分片正常
- [ ] CPU/NVMe offloading 精度保持 FP32

### 流水线并行

- [ ] 微批次数量（chunks）设置合理
- [ ] 激活检查点精度保持

## 9.4 问题诊断流程图

### 训练崩溃诊断

```
训练出现 NaN/Inf
├── 检查梯度范数
│   ├── 梯度范数 > 100 → 梯度爆炸 → 增大梯度裁剪
│   └── 梯度范数 < 1e-7 → 梯度消失 → 检查初始化
├── 检查激活值
│   ├── 激活值包含 Inf → 前向传播溢出 → 使用 Pre-LN
│   └── 激活值全为 0 → Dead ReLU → 使用 LeakyReLU
├── 检查损失值
│   ├── 损失突然增大 → 学习率过大 → 使用 warmup
│   └── 损失为 NaN → 数值下溢/上溢 → 使用 BF16
└── 检查优化器状态
    ├── 二阶矩爆炸 → Adam epsilon 过小 → 增大 epsilon
    └── 动量为 NaN → 梯度包含 NaN → 检查数据
```

## 9.5 生产环境部署清单

### 训练前

- [ ] 在小型数据集上验证数值稳定性
- [ ] 单卡训练稳定后再启动分布式
- [ ] 检查点加载/保存测试通过
- [ ] 监控告警配置完成

### 训练中

- [ ] 训练进度实时监控
- [ ] 资源使用监控（GPU 显存、CPU 内存）
- [ ] 异常自动恢复机制
- [ ] 定期备份检查点到远程存储

### 训练后

- [ ] 最终模型验证通过
- [ ] 训练日志归档
- [ ] 超参数配置文档化

---

**上一章**: [第8章 调试与诊断方法论](./08-debugging-methodology.md) | **下一章**: [附录](./appendix.md)
