# 附录

## A. 精度格式对比表

### A.1 浮点格式参数

| 格式 | 指数位 | 尾数位 | 总位数 | 偏置值 | 最小正值 | 最大正值 | 机器 epsilon |
|------|--------|--------|--------|--------|----------|----------|--------------|
| FP64 | 11 | 52 | 64 | 1023 | ~5e-324 | ~1.8e308 | ~2.2e-16 |
| FP32 | 8 | 23 | 32 | 127 | ~1.2e-38 | ~3.4e38 | ~1.2e-7 |
| FP16 | 5 | 10 | 16 | 15 | ~6.1e-5 | ~6.55e4 | ~9.8e-4 |
| BF16 | 8 | 7 | 16 | 127 | ~1.2e-38 | ~3.4e38 | ~7.8e-3 |
| E4M3 | 4 | 3 | 8 | 7 | ~1.0e-2 | 448 | ~0.125 |
| E5M2 | 5 | 2 | 8 | 15 | ~1.5e-5 | 57,344 | ~0.25 |

## B. 硬件支持矩阵

### B.1 NVIDIA GPU

| GPU | 架构 | FP16 TensorCore | BF16 TensorCore | FP8 TensorCore | MXFP8 |
|-----|------|-----------------|-----------------|----------------|-------|
| V100 | Volta | ✅ | ❌ | ❌ | ❌ |
| T4 | Turing | ✅ | ❌ | ❌ | ❌ |
| RTX 3090 | Ampere | ✅ | ✅ | ❌ | ❌ |
| A100 | Ampere | ✅ | ✅ | ❌ | ❌ |
| A10 | Ampere | ✅ | ✅ | ❌ | ❌ |
| H100 | Hopper | ✅ | ✅ | ✅ | ❌ |
| H200 | Hopper | ✅ | ✅ | ✅ | ❌ |
| B200 | Blackwell | ✅ | ✅ | ✅ | ✅ |

## C. 关键论文索引

### 初始化与归一化

- **Xavier Initialization** — University of Toronto (AISTATS 2010)
- **Kaiming Initialization** — Microsoft Research Asia (ICCV 2015)
- **Batch Normalization** — Google (ICML 2015)
- **RMSNorm** — University of Edinburgh (NeurIPS 2019)

### 混合精度训练

- **Mixed Precision Training** — NVIDIA, Baidu, UC Berkeley (ICLR 2018)
- **BFloat16** — Google Brain
- **FP8 Formats** — NVIDIA, Arm, Intel, Qualcomm

### 低精度训练算法

- **Stochastic Rounding** — EPFL, IBM Research (AISTATS 2025)
- **SAM Optimizer** — FAIR/Meta AI (ICLR 2021)
- **Unit Scaling** — Graphcore, Cambridge

### 分布式训练

- **Deep Gradient Compression** — Seoul National University, NVIDIA (ICLR 2018)
- **L-GRECO** — KAIST (MLSys 2024)
- **ZeRO** — Microsoft Research (SC 2020)

### 工业实践

- **DeepSeek-V3** — DeepSeek (2024)
- **NVIDIA Transformer Engine** — NVIDIA

## D. 常用工具和资源

### 开源库

| 库名 | 用途 | 链接 |
|------|------|------|
| PyTorch AMP | 自动混合精度 | torch.cuda.amp |
| Transformer Engine | FP8 训练 | github.com/NVIDIA/TransformerEngine |
| DeepSpeed | 分布式训练 | github.com/microsoft/DeepSpeed |
| bitsandbytes | 8-bit 优化器 | github.com/TimDettmers/bitsandbytes |

### 监控工具

| 工具 | 用途 | 链接 |
|------|------|------|
| Weights & Biases | 实验跟踪 | wandb.ai |
| TensorBoard | 可视化 | tensorflow.org/tensorboard |
| PyTorch Profiler | 性能分析 | pytorch.org/docs/stable/profiler |

## E. 术语表

| 术语 | 定义 |
|------|------|
| 数值稳定性 | 算法在有限精度计算中产生准确结果的能力 |
| 混合精度训练 | 使用 FP16/BF16 计算，FP32 存储的训练方法 |
| 梯度缩放 | 在反向传播前放大损失，防止梯度下溢 |
| 微缩放 | 块级共享指数的细粒度量化技术 |
| 随机舍入 | 按概率舍入，保持无偏性的舍入方法 |
| 误差反馈 | 压缩误差累积并在后续迭代补偿的机制 |
| 梯度裁剪 | 限制梯度范数，防止梯度爆炸的技术 |
| 谱归一化 | 控制权重矩阵谱范数的正则化方法 |
| 平坦极小值 | 损失函数曲面上较平坦的局部最小点 |
