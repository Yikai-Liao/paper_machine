---
title: "Comet: Accelerating Private Inference for Large Language Model by Predicting Activation Sparsity"
pubDatetime: 2025-05-12T05:29:30+00:00
slug: "2025-05-comet-sparsity-inference"
type: "arxiv"
id: "2505.07239"
score: 0.6175049989435286
author: "grok-3-latest"
authors: ["Guang Yan", "Yuhui Zhang", "Zimu Guo", "Lutan Zhao", "Xiaojun Chen", "Chen Wang", "Wenhao Wang", "Dan Meng", "Rui Hou"]
tags: ["LLM", "Activation Sparsity", "Private Inference", "MPC", "Sparse Computation"]
institution: ["State Key Laboratory of Cyberspace Security Defense, Institute of Information Engineering, CAS", "University of Chinese Academy of Sciences", "Tsinghua University"]
description: "本文提出 *Comet* 系统，通过预测激活稀疏性并设计高效私有推理协议，在保护隐私的同时显著加速大型语言模型的推理，实现了 1.87× 到 2.63× 的性能提升。"
---

> **Summary:** 本文提出 *Comet* 系统，通过预测激活稀疏性并设计高效私有推理协议，在保护隐私的同时显著加速大型语言模型的推理，实现了 1.87× 到 2.63× 的性能提升。 

> **Keywords:** LLM, Activation Sparsity, Private Inference, MPC, Sparse Computation

**Authors:** Guang Yan, Yuhui Zhang, Zimu Guo, Lutan Zhao, Xiaojun Chen, Chen Wang, Wenhao Wang, Dan Meng, Rui Hou

**Institution(s):** State Key Laboratory of Cyberspace Security Defense, Institute of Information Engineering, CAS, University of Chinese Academy of Sciences, Tsinghua University


## Problem Background

随着大型语言模型（LLM）广泛部署在云端提供推理服务，用户输入的敏感数据面临隐私泄露风险。
安全多方计算（MPC）是一种有效的隐私保护方法，但其在 LLM 推理中的应用受到高通信开销的限制，尤其是在处理大规模模型时，通信时间占比超过 85%。
论文注意到 LLM 的激活稀疏性（Activation Sparsity），即非线性激活函数（如 ReLU）后大量神经元输出为零，若能预测并跳过这些零值相关的计算和通信，可显著提升私有推理效率。

## Method

*   **核心思想：** 提出 *Comet* 系统，利用激活稀疏性预测减少 LLM 私有推理中的计算和通信开销，同时通过 MPC 保护隐私。
*   **激活稀疏性预测器：** 设计轻量级神经网络（由两层低秩全连接层和阈值层组成）预测激活函数输出（如 MHA 和 FFN 模块）的稀疏性分布。预测器预训练于公开数据集，并通过遗忘洗牌（Oblivious Shuffle）技术保护稀疏性分布的隐私，避免暴露非零元素位置，同时实现高效索引。
*   **私有推理协议：** 针对线性层，提出两种稀疏矩阵乘法协议：
    *   **SOMM（Sparse Output Matrix Multiplication）：** 用于前置线性层，根据输出稀疏性分布，通过分组非零输出元素（利用空间局部性），将多次点乘合并为块矩阵乘法，避免重复掩码和通信。
    *   **SIMM（Sparse Input Matrix Multiplication）：** 用于后置线性层，基于输入稀疏性分布，采用列-行乘法策略，将同一列的非零元素聚合为子向量，减少重复通信。
  针对非线性层，直接跳过对应零值的计算。所有操作在 MPC 框架下安全执行。
*   **KV 缓存管理：** 针对稀疏性预测导致的 KV 缓存不连续问题，提出两种策略：
    *   **合并缓存缺失请求：** 将多个缓存缺失请求合并为单一批次处理，减少重复通信。
    *   **预取非激活头：** 基于成本-收益分析，预计算部分非激活注意力头的 KV 值，提升未来缓存命中率，减少后续通信开销。

## Experiment

*   **有效性：** 在多个 LLM 模型（OPT-1.3B 到 6.7B，Llama2-7B）上，*Comet* 相比六种现有私有推理系统（如 Puma, MPCFormer）实现了 1.87× 到 2.63× 的端到端加速，以及 1.94× 到 2.64× 的通信量减少，尤其在大模型和高稀疏性场景下效果更显著。
*   **全面性：** 实验设置考虑了不同模型规模、输出长度（1 到 16 个 token）和网络带宽（100Mbps 到 5Gbps），验证了方法在通信敏感场景（如低带宽）下的优越性，加速比可达 3.25×。此外，针对不同层（线性层和非线性层）以及不同模块（MHA 和 FFN）的性能分解表明，FFN 层因更高稀疏性（约 90%）获得更大加速（高达 8.08×）。
*   **精度影响：** 准确性测试显示 *Comet* 仅带来约 1.5% 的精度损失，预测器召回率达 93%，通过阈值调整可进一步平衡速度和精度，表明方法在性能提升和模型质量之间取得较好折衷。
*   **开销分析：** 预测器开销占总推理时间的不到 15%，遗忘洗牌技术显著降低索引开销（提升 4.1× 到 4.8×），而缓存管理和稀疏矩阵乘法协议进一步优化了通信效率。

## Further Thoughts

激活稀疏性作为 LLM 的固有特性，不仅可用于 MPC 场景的私有推理加速，还可能启发其他隐私计算框架（如联邦学习）中的稀疏性优化策略；遗忘洗牌技术在保护隐私的同时实现高效索引，对隐私保护的数据处理任务（如安全数据库查询）具有借鉴意义；KV 缓存管理中的成本-收益预取机制，可推广至其他缓存优化问题，动态平衡计算与通信开销。