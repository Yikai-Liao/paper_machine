---
title: "Bottlenecked Transformers: Periodic KV Cache Abstraction for Generalised Reasoning"
pubDatetime: 2025-05-22T17:33:49+00:00
slug: "2025-05-bottlenecked-transformer-kv"
type: "arxiv"
id: "2505.16950"
score: 0.7954459800659404
author: "grok-3-latest"
authors: ["Adnan Oomerjee", "Zafeirios Fountas", "Zhongwei Yu", "Haitham Bou-Ammar", "Jun Wang"]
tags: ["LLM", "Information Bottleneck", "Transformer Architecture", "KV Cache", "Reasoning", "Generalization"]
institution: ["UCL Centre for AI", "Huawei Noah’s Ark", "Hong Kong University of Science and Technology"]
description: "本文通过信息瓶颈理论揭示Transformer在泛化推理中的局限，提出Bottlenecked Transformer架构，利用周期性KV缓存重写实现输入压缩与预测信息保留的平衡，显著提升推理任务性能和泛化能力。"
---

> **Summary:** 本文通过信息瓶颈理论揭示Transformer在泛化推理中的局限，提出Bottlenecked Transformer架构，利用周期性KV缓存重写实现输入压缩与预测信息保留的平衡，显著提升推理任务性能和泛化能力。 

> **Keywords:** LLM, Information Bottleneck, Transformer Architecture, KV Cache, Reasoning, Generalization

**Authors:** Adnan Oomerjee, Zafeirios Fountas, Zhongwei Yu, Haitham Bou-Ammar, Jun Wang

**Institution(s):** UCL Centre for AI, Huawei Noah’s Ark, Hong Kong University of Science and Technology


## Problem Background

大型语言模型（LLMs）基于Transformer架构在信息检索和模式识别任务中表现出色，但在超出训练分布的泛化推理（extrapolation）任务上表现不佳，往往依赖于训练数据中的模式插值（interpolation）而非真正的抽象推理。
论文从信息瓶颈理论（Information Bottleneck Theory, IB）的视角分析，认为Transformer的KV缓存作为终端信息瓶颈，在自回归训练目标下倾向于保留过多输入信息而未能有效压缩和抽象出对未来预测有用的特征，限制了模型在分布外（OOD）任务上的表现。
核心问题在于如何在Transformer中引入信息压缩机制，平衡输入信息的保留与预测信息的提取，从而提升泛化推理能力。

## Method

*   **理论基础：** 基于信息瓶颈理论（IB），作者证明了解码器型Transformer的KV缓存作为终端信息瓶颈（terminal bottleneck），在自回归训练目标下会最大化输入信息（I(X;Z)）和输出预测信息（I(Z;Y)），导致其更偏向于记忆输入前缀而非抽象出通用特征，限制了泛化能力。
*   **核心创新 - Cache Processor模块：** 提出了一种新型架构'Bottlenecked Transformer'，通过引入一个附加的Transformer模块——Cache Processor，周期性地（每隔固定数量的token，记为B）对整个KV缓存进行全局重写（rewrite）。
    *   **重写机制：** Cache Processor通过学习一个更新增量（∆-update），调整KV缓存内容，使其不再负责重建输入前缀，而是专注于编码对未来token预测有用的特征。
    *   **架构细节：** Cache Processor本身是一个多层Transformer，处理每一层的KV缓存切片，并通过一组可学习的token在层间传递信息；重写时保留KV缓存的完整长度和维度，确保信息检索能力不受损；不使用因果掩码（causal masking），允许全局计算以提取预测特征。
*   **训练与生成流程：** 模型采用端到端训练，使用标准交叉熵损失，不引入显式互信息正则化，而是依赖随机梯度下降（SGD）的噪声隐式压缩输入信息；生成时，Cache Processor每隔B个token重写一次缓存，丢弃旧缓存并基于新缓存继续生成。
*   **设计目标：** 通过周期性重处理，引入类似RNN的固定大小表示特性，同时保留Transformer的长上下文处理优势，实现输入压缩与预测能力之间的平衡。

## Experiment

*   **任务与设置：** 在三个合成多步推理任务（整数乘法、多项式求值、数独）上进行评估，任务设计为Markov决策过程（MDP），每次状态转换独立处理，确保模型依赖于每一步推理而非缓存上下文；数据集分为训练、验证和测试集，包含分布内（in-distribution）和分布外（OOD）难度，评估全面。
*   **性能提升：** Bottlenecked Transformer在所有任务上显著优于Vanilla Transformer，即使参数量远小于后者（例如，4M+2.8M参数的Bottlenecked模型在整数乘法任务上准确率达98.23%，而16M参数的Vanilla模型仅为80.17%）；尤其在分布内任务上，性能接近完美。
*   **泛化能力：** 在分布外任务上，Bottlenecked Transformer表现出更强的泛化能力，例如在整数乘法任务中，OOD准确率随Cache Processor容量增加而提升（最高达19.22%），而Vanilla Transformer几乎无改进甚至下降。
*   **基线对比：** 相比缓存剪枝方法（如H2O），Bottlenecked Transformer避免了剪枝导致的预测信息丢失问题，性能更优，尤其在需要完整上下文的任务（如数独）中，剪枝方法表现极差。
*   **额外分析：** 通过注意力行熵（attention row entropy）分析，验证了Cache Processor降低了输入信息复杂度（I(X;Z)），将缓存容量重新分配给预测特征，提升了泛化能力。
*   **结论：** 实验结果支持了作者的理论假设，方法提升显著，实验设计合理且全面。

## Further Thoughts

信息瓶颈理论（IB）为分析Transformer泛化能力提供了新视角，未来是否可探索其他深度学习架构中的信息瓶颈，设计普适压缩机制？
周期性KV缓存重写类似于神经科学中的记忆巩固，是否可应用于其他序列模型或多模态任务？
论文提出自适应Cache Processor的潜力，动态决定重写时机和方式，是否可结合注意力机制或稀疏性正则化实现更精细压缩？
周期性重写与大脑记忆巩固过程的类比是否能推动AI模型设计更接近人类认知机制，例如引入‘离线’处理阶段增强学习效果？