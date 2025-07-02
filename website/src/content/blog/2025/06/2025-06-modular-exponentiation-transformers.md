---
title: "Learning Modular Exponentiation with Transformers"
pubDatetime: 2025-06-30T10:00:44+00:00
slug: "2025-06-modular-exponentiation-transformers"
type: "arxiv"
id: "2506.23679"
score: 0.7228022831376535
author: "grok-3-latest"
authors: ["David Demitri Africa", "Sara M. Kapoor", "Theo Simon Sorg"]
tags: ["Transformer", "Modular Arithmetic", "Mechanistic Interpretability", "Sampling Strategy", "Numerical Reasoning", "Grokking"]
institution: ["University of Cambridge"]
description: "本文通过精心设计的采样策略和机制可解释性方法，首次系统研究了 Transformer 模型在模块化指数运算任务上的高精度学习能力，揭示了其内部算术表示和学习动态。"
---

> **Summary:** 本文通过精心设计的采样策略和机制可解释性方法，首次系统研究了 Transformer 模型在模块化指数运算任务上的高精度学习能力，揭示了其内部算术表示和学习动态。 

> **Keywords:** Transformer, Modular Arithmetic, Mechanistic Interpretability, Sampling Strategy, Numerical Reasoning, Grokking

**Authors:** David Demitri Africa, Sara M. Kapoor, Theo Simon Sorg

**Institution(s):** University of Cambridge


## Problem Background

模块化指数运算（Modular Exponentiation）是数论和密码学（如 RSA 和 Diffie-Hellman 算法）的核心操作，但机器学习模型在处理这种非线性、周期性数学任务时面临泛化和内部表示的挑战。
本文从机制可解释性（Mechanistic Interpretability）角度出发，研究 Transformer 模型如何学习和表示模块化指数运算的数学结构，填补了这一领域的空白，为理论研究（数值推理能力）和实际应用（安全计算）提供了新视角。

## Method

* **数据采样策略**：针对模块化指数运算涉及三个操作数（a, b, c）及结果 d，设计了多种采样方法以捕捉算术统计特性：
  * **均匀操作数（Uniform Operands）**：随机均匀采样 a, b, c 并计算 d，但导致结果 d 偏向小值，造成类别不平衡。
  * **均匀结果（Uniform Outcomes）**：通过拒绝采样确保 d 均匀分布，缓解输出偏态。
  * **倒数操作数（Reciprocal Operands）**：对 a, b 采用对数均匀分布（Log-Uniform），有效平衡数据分布，提升模型性能。
* **模型架构与训练**：使用 4 层编码器-解码器 Transformer，嵌入维度 256，8 个注意力头，Adam 优化器（学习率 10^-4），输入整数通过字符串模板（如 V3 + a + b + c + d）表示，适应离散 token 处理。
* **机制可解释性分析**：
  * **主成分分析（PCA）**：研究嵌入空间中数值 token 的数理模式（如奇偶性、素性、欧拉函数值等），观察学习前后结构变化。
  * **激活补丁（Activation Patching）**：通过替换模型组件激活值，识别对特定任务（如常规指数运算）关键的子图（Circuits），揭示内部计算机制。
* **创新点**：结合数据分布设计与可解释性工具，从学习动态和内部结构两方面剖析 Transformer 的算术能力。

## Experiment

* **性能表现**：倒数操作数采样模型表现最佳，测试集准确率超 80%，验证集超 90%；均匀操作数采样因类别不平衡仅达 13.17%，均匀结果采样有所改善但非必要。
* **学习动态**：观察到 Grokking 现象，如在第 1725-1750 个 epoch 间，模型对某些模数（如 23 倍数）准确率从 20% 骤升至 100%，表明逐步内化数学结构。
* **嵌入空间分析**：Grokking 前后嵌入空间结构变化明显，后期嵌入趋于集中，但按数理属性聚类不显著，内部数学表示尚不完全清晰。
* **激活补丁结果**：最后一层注意力头组成的子图即可完成常规指数运算，准确率与完整模型相当，表明模型将算术操作分解为特定电路。
* **评估**：实验设置全面，涵盖多种采样策略、基数选择及可解释性分析，但嵌入空间数理模式不明显可能限制机制理解深度。

## Further Thoughts

Grokking 现象揭示了模型通过发现数学规律（如倍数关系）实现泛化，启发我们是否可以通过设计特定数据分布加速数学结构学习；
激活补丁显示模型将复杂任务分解为子任务的能力，提示是否可设计针对数学任务优化的高效架构；
采样策略对性能的影响突出数据分布设计的重要性，鼓励探索更多基于数理特性的采样方法以提升其他数学任务表现。