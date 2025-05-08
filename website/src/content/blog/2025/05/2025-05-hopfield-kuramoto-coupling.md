---
title: "Binding threshold units with artificial oscillatory neurons"
pubDatetime: 2025-05-06T15:54:52+00:00
slug: "2025-05-hopfield-kuramoto-coupling"
type: "arxiv"
id: "2505.03648"
score: 0.4438167008155235
author: "grok-3-latest"
authors: ["Vladimir Fanaskov", "Ivan Oseledets"]
tags: ["Neural Network", "Oscillatory Neuron", "Threshold Unit", "Synchronization", "Associative Memory"]
institution: ["AIRI", "Skoltech"]
description: "本文提出了一种理论严谨的 Hopfield-Kuramoto 关联记忆模型，通过低秩耦合机制将阈值单元和振荡神经元结合，显著提升了神经网络对任务变化的适应性，并通过 MNIST 实验验证了其计算潜力。"
---

> **Summary:** 本文提出了一种理论严谨的 Hopfield-Kuramoto 关联记忆模型，通过低秩耦合机制将阈值单元和振荡神经元结合，显著提升了神经网络对任务变化的适应性，并通过 MNIST 实验验证了其计算潜力。 

> **Keywords:** Neural Network, Oscillatory Neuron, Threshold Unit, Synchronization, Associative Memory

**Authors:** Vladimir Fanaskov, Ivan Oseledets

**Institution(s):** AIRI, Skoltech


## Problem Background

传统神经网络中的阈值单元（Threshold Units）擅长模拟神经元活动的平均强度，但忽略了生物神经元中关键的时间相位信息，而振荡神经元（Oscillatory Neurons）通过频率调制和同步机制捕捉生物大脑的动态交互特性，在对象发现和推理任务中表现出潜力。
论文旨在解决如何将这两种神经元在理论和实践上耦合，形成统一模型，以结合两者的优势并探索新的计算能力，同时模拟生物神经编码的多样性。

## Method

*   **理论基础：** 论文基于动态系统理论，构建了一个 Hopfield-Kuramoto 关联记忆模型，通过 Lyapunov 函数确保系统的稳定性，其中阈值单元遵循 Hopfield 模型（模拟平均神经活动），振荡神经元遵循广义 Kuramoto 模型（模拟频率调制和同步）。
*   **耦合机制：** 设计了特定的交互项，将振荡神经元和阈值单元连接起来，具体通过低秩修正（Low-Rank Correction）调整阈值单元的权重矩阵，这种修正可以动态依赖于振荡神经元的状态，类似于深度学习中的 LoRA 方法或 Hebbian 学习机制。
*   **实现细节：** 振荡神经元能够实现多路复用（Multiplexing），即根据输入刺激动态改变阈值单元的记忆内容或计算路径；同时，阈值单元的活动也可以反向影响振荡神经元的同步行为。
*   **优势：** 这种方法不改变原始神经元模型的基本动态，仅通过交互项引入耦合，既保留了各自的计算特性，又增加了系统的适应性和表达能力。

## Experiment

*   **有效性：** 在 MNIST 数据集上，预训练的 Hopfield 子网络在原始任务上测试准确率为 97.2%，但在修改任务（如标签交换或合并）上准确率下降至 76.3%-91.7%；通过引入振荡神经元并微调耦合模型，准确率显著提升，非关联任务恢复至最高 97.2%，关联任务提升至最高 93.6%。
*   **参数影响：** 实验对比了振荡自由度 D 和邻居数量 k 的影响，显示 D=6 通常优于 D=4，较大的 k 提升了振荡子网络的表达能力，尤其在非关联任务中效果明显。
*   **合理性：** 实验设置全面，涵盖了多种任务变体（Swap, Conflation, Associative Swap 等），并通过冻结 Hopfield 参数、仅训练耦合参数的方式验证了耦合机制的独立贡献，设计合理且数据支持结论。

## Further Thoughts

振荡神经元作为动态权重调整机制的潜力令人印象深刻，类似于 LoRA 或‘fast weights’，这可能为大型神经网络的轻量级微调或上下文依赖的多任务学习提供新思路；此外，振荡与阈值单元分别模拟神经编码的强度和频率特性，启发我们进一步探索生物启发的混合计算范式，或许可以在强化学习或元学习中引入类似机制，实现更灵活的计算路径切换。