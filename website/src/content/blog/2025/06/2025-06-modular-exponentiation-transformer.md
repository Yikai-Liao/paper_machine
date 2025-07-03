---
title: "Learning Modular Exponentiation with Transformers"
pubDatetime: 2025-06-30T10:00:44+00:00
slug: "2025-06-modular-exponentiation-transformer"
type: "arxiv"
id: "2506.23679"
score: 0.7228022831376535
author: "grok-3-latest"
authors: ["David Demitri Africa", "Sara M. Kapoor", "Theo Simon Sorg"]
tags: ["Transformer", "Modular Arithmetic", "Sampling Strategy", "Mechanistic Interpretability", "Grokking"]
institution: ["University of Cambridge"]
description: "本文通过精心设计的采样策略训练 Transformer 模型高精度完成模块化指数运算，并利用可解释性分析揭示其学习动态和内部计算结构，为神经网络在数理任务中的应用提供了新视角。"
---

> **Summary:** 本文通过精心设计的采样策略训练 Transformer 模型高精度完成模块化指数运算，并利用可解释性分析揭示其学习动态和内部计算结构，为神经网络在数理任务中的应用提供了新视角。 

> **Keywords:** Transformer, Modular Arithmetic, Sampling Strategy, Mechanistic Interpretability, Grokking

**Authors:** David Demitri Africa, Sara M. Kapoor, Theo Simon Sorg

**Institution(s):** University of Cambridge


## Problem Background

模块化指数运算（Modular Exponentiation）是数论和密码学（如 RSA 和 Diffie-Hellman 密钥交换）的核心操作，但其非线性与周期性特性对机器学习模型的泛化能力构成挑战。
现有研究虽已探索 Transformer 模型在算术任务（如最大公约数 GCD）中的表现，但模块化指数运算的复杂交互尚未被系统性研究。
本文旨在探究 Transformer 模型是否能有效学习该任务，以及其内部如何表示和处理数理信息，进而为神经网络在数理任务中的可解释性和应用提供新见解。

## Method

* **数据采样策略**：设计了多种采样方法以捕捉模块化算术的统计特性，解决输入输出分布不平衡问题：
  * **均匀操作数（Uniform Operands）**：随机均匀采样操作数 a, b, c，计算结果 d，导致结果偏向小值。
  * **均匀结果（Uniform Outcomes）**：通过拒绝采样确保结果 d 均匀分布，减少输出偏差。
  * **倒数操作数（Reciprocal Operands）**：对 a, b 采用对数均匀分布，平衡输入范围，同时对 c 和 d 施加约束以提升覆盖率。
* **模型架构与训练**：使用 4 层编码器-解码器 Transformer 模型，嵌入维度 256，配备 8 个注意力头，批大小 256，采用 Adam 优化器（学习率 10^-4），每轮生成 300,000 个样本。
* **整数表示**：将整数转化为特定字符串模板（如基数 1000），以适应 Transformer 的离散 token 处理。
* **机制可解释性分析**：
  * **主成分分析（PCA）**：分析嵌入空间中数理属性（如奇偶性、素性、欧拉函数）的表示演变，探索模型学习动态。
  * **激活补丁（Activation Patching）**：通过替换激活值识别模型内部关键计算子图（Circuits），评估各组件对任务的因果影响。
这些方法结合训练与分析，旨在揭示 Transformer 如何学习模块化算术及其内部机制。

## Experiment

* **性能表现**：在倒数操作数采样策略下，模型测试准确率达到 80.39%，显著优于均匀操作数策略的 13.17%，验证了数据分布设计对学习效果的关键影响。
* **学习动态**：观察到 Grokking 现象，模型在训练中展现出从记忆到泛化的突然转变，例如模数 23 的倍数在 1725-1750 轮间准确率从 20% 跃升至 100%，其他模数（如 31、39）也有类似同步提升。
* **嵌入空间分析**：PCA 结果显示 Grokking 后嵌入空间趋于集中，但数理属性（如奇偶性、素性）的聚类不明显，表明内部结构可能更复杂或需更精细工具分析。
* **激活补丁结果**：发现最后一层注意力头组成的子图在普通指数运算任务上与完整模型准确率一致，表明高层注意力机制集中编码了任务特定转换。
* **实验设置评价**：实验设计全面，涵盖多种采样策略、基数选择（1000、1013 等）和学习动态分析，验证了不同分布和架构参数的影响；但嵌入空间的可解释性结果有限，可能是分析深度或工具的限制。

## Further Thoughts

论文揭示了 Transformer 模型通过数据分布设计和内部子图功能分离学习复杂算术任务的能力，启发我思考：是否可以通过动态调整采样策略（例如根据模型学习进度自适应分布）进一步加速泛化？此外，最后一层注意力头对普通指数运算的关键作用提示，是否可以在预训练阶段针对特定算术任务设计架构或损失函数，以增强功能模块化分离，进而提升模型在数理任务中的效率和可解释性？