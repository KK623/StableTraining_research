# 模型训练稳定性技术指南

![Status](https://img.shields.io/badge/status-complete-success)
![Chapters](https://img.shields.io/badge/chapters-9-blue)
![Code Examples](https://img.shields.io/badge/code%20examples-12-orange)

## 文档定位

面向深度学习工程师和研究员的系统性技术指南，聚焦**数值稳定性**与**低精度训练**。

**所有技术点均附可靠信源及机构归属**。

## 章节导航

| 章节 | 主题 | 文件 |
|------|------|------|
| 第1章 | 引言 | [01-introduction.md](./01-introduction.md) |
| 第2章 | 数据层稳定性 | [02-data-stability.md](./02-data-stability.md) |
| 第3章 | 模型架构稳定性 | [03-architecture-stability.md](./03-architecture-stability.md) |
| 第4章 | 数值精度与低精度训练（核心） | [04-numerical-precision.md](./04-numerical-precision.md) |
| 第5章 | 算法设计层面的稳定性保障（核心） | [05-algorithm-design.md](./05-algorithm-design.md) |
| 第6章 | 优化器稳定性 | [06-optimizer-stability.md](./06-optimizer-stability.md) |
| 第7章 | 分布式训练稳定性 | [07-distributed-stability.md](./07-distributed-stability.md) |
| 第8章 | 调试与诊断方法论 | [08-debugging-methodology.md](./08-debugging-methodology.md) |
| 第9章 | 实践检查清单 | [09-checklists.md](./09-checklists.md) |
| 附录 | 格式对比表、硬件矩阵、论文索引 | [appendix.md](./appendix.md) |

## 快速开始

- 遇到训练崩溃？查看 [第8章](./08-debugging-methodology.md) 诊断流程
- 计划使用 FP8/BF16？先读 [第4章](./04-numerical-precision.md)
- 需要落地 checklist？直接跳到 [第9章](./09-checklists.md)

## 配套代码

`code-examples/` 目录包含各章节的可运行代码示例。

| 代码文件 | 说明 |
|----------|------|
| [data_preprocessing.py](./code-examples/data_preprocessing.py) | 数据预处理数值稳定性 |
| [initialization.py](./code-examples/initialization.py) | 初始化与架构稳定性 |
| [mixed_precision.py](./code-examples/mixed_precision.py) | 混合精度训练 |
| [fp8_simulation.py](./code-examples/fp8_simulation.py) | FP8 数值模拟 |
| [stochastic_rounding.py](./code-examples/stochastic_rounding.py) | 随机舍入实现 |
| [sam_optimizer.py](./code-examples/sam_optimizer.py) | SAM 优化器 |
| [gradient_clipping.py](./code-examples/gradient_clipping.py) | 梯度裁剪策略 |
| [eight_bit_optimizer.py](./code-examples/eight_bit_optimizer.py) | 8-bit 优化器 |
| [distributed_setup.py](./code-examples/distributed_setup.py) | 分布式训练设置 |
| [debugging_hooks.py](./code-examples/debugging_hooks.py) | 调试钩子 |
| [training_monitor.py](./code-examples/training_monitor.py) | 训练监控器 |

## 技术覆盖

### 数值精度
- FP32/FP16/BF16/FP8 数值特性
- 混合精度训练 (AMP + GradScaler)
- MXFP 微缩放技术
- 量化感知训练 (QAT)

### 算法层面
- 随机舍入 (Stochastic Rounding)
- Sharpness-Aware Minimization (SAM)
- 单元缩放 (Unit Scaling / u-μP)
- 谱归一化 (Spectral Normalization)
- 梯度压缩 (L-GRECO, Top-K)

### 工程实践
- 初始化策略 (Xavier, Kaiming, Fixup)
- 归一化层 (BatchNorm, LayerNorm, RMSNorm)
- 梯度裁剪与学习率调度
- 8-bit 优化器
- ZeRO 分布式训练

## 信源标准

本文档所有技术点均标注：
- **提出机构**: 大学/公司/研究实验室
- **原始论文**: 顶会论文或 arXiv 预印本
- **官方文档**: 框架/硬件厂商文档

信源可靠性分级：
1. **一级**: 顶会论文 (NeurIPS/ICML/ICLR/MLSys/OSDI)、官方文档
2. **二级**: 高引 arXiv、官方技术博客
3. **三级**: 开源实现、社区经验

## 贡献与反馈

本文档持续更新中。如发现错误或有改进建议，欢迎提交 Issue 或 PR。

---

*最后更新: 2026-04-08*
