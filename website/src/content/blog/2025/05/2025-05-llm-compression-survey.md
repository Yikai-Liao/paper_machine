---
title: "Optimizing LLMs for Resource-Constrained Environments: A Survey of Model Compression Techniques"
pubDatetime: 2025-05-05T01:27:47+00:00
slug: "2025-05-llm-compression-survey"
type: "arxiv"
id: "2505.02309"
score: 0.7052733890847532
author: "grok-3-latest"
authors: ["Sanjay Surendranath Girija", "Shashank Kapoor", "Lakshit Arora", "Dipen Pradhan", "Aman Raj", "Ankit Shetgaonkar"]
tags: ["LLM", "Model Compression", "Knowledge Distillation", "Quantization", "Pruning"]
institution: ["Google"]
description: "本文全面综述了大型语言模型（LLMs）压缩技术，包括知识蒸馏、量化和剪枝，分析了其在资源受限环境中的应用效果和未来方向，为高效部署提供了重要参考。"
---

> **Summary:** 本文全面综述了大型语言模型（LLMs）压缩技术，包括知识蒸馏、量化和剪枝，分析了其在资源受限环境中的应用效果和未来方向，为高效部署提供了重要参考。 

> **Keywords:** LLM, Model Compression, Knowledge Distillation, Quantization, Pruning

**Authors:** Sanjay Surendranath Girija, Shashank Kapoor, Lakshit Arora, Dipen Pradhan, Aman Raj, Ankit Shetgaonkar

**Institution(s):** Google


## Problem Background

大型语言模型（LLMs）因其巨大的参数量和资源需求（如高计算成本、内存占用、能耗和延迟），难以直接部署在资源受限的移动和边缘设备上。
论文旨在通过模型压缩技术解决这一问题，减少模型大小和推理成本，同时尽量保持性能，以实现 LLMs 在资源受限环境中的高效应用。

## Method

*   **知识蒸馏（Knowledge Distillation, KD）**：通过将大模型（教师模型）的知识转移到小模型（学生模型），使学生模型以较少的参数模仿教师模型的输出或中间表示。具体形式包括：
    *   **软目标蒸馏**：学生模型学习教师模型的软化概率分布（通过高温度参数），以获取更丰富的标签信息。
    *   **特征蒸馏**：利用教师模型中间层的特征（hints）指导学生模型对应层的学习。
    *   **关系蒸馏**：保持数据点或中间表示之间的关系（如距离或角度）一致。
    *   **自蒸馏**：模型从自身或早期版本学习，用于正则化和提升小模型性能。
    *   **多教师蒸馏**：从多个教师模型学习，结合不同任务专长的知识。
*   **模型量化（Model Quantization）**：通过降低参数和激活值的数值精度（如从 32 位浮点数到 8 位整数甚至更低），减少内存占用和加速推理。主要方法包括：
    *   **训练后量化（PTQ）**：在训练完成后直接转换模型精度，常使用校准数据集优化映射，适用于快速部署但可能损失精度。
    *   **量化感知训练（QAT）**：在训练过程中模拟量化操作，使模型参数对量化误差更鲁棒，通常精度更高但训练成本增加。
    *   **混合精度量化**：不同层或组件采用不同精度，平衡效率与性能。
    *   **二值化和三值化量化**：极端减少到 1 位或 2 位表示，显著降低模型大小。
*   **模型剪枝（Model Pruning）**：移除模型中冗余或不重要的部分（如权重、神经元、层）以减少规模，分为：
    *   **非结构化剪枝**：移除个别低重要性权重，生成稀疏矩阵，压缩率高但硬件加速受限。
    *   **结构化剪枝**：移除整个组件（如层、通道），对硬件更友好但压缩率可能较低。
*   **其他策略**：
    *   **专家混合（Mixture of Experts, MoE）**：通过路由机制仅激活部分专家网络，降低计算成本。
    *   **提前退出（Early-Exit Strategies）**：在模型早期层设置预测出口，跳过不必要的计算以提升效率。

## Experiment

*   **有效性**：论文综述了文献中的多个案例，显示模型压缩技术在减少内存和计算成本方面效果显著。例如，知识蒸馏（如 GKD、MiniLLM）可使学生模型参数减少数倍甚至数十倍，同时在某些任务上性能接近或超越大模型；量化技术（如 ZeroQuant、GPTQ）可实现 2x-5x 内存减少和 3x-5x 推理加速，精度损失很小；剪枝（如 Lottery Ticket Hypothesis）可减少高达 90% 参数而对精度影响有限。
*   **全面性**：实验覆盖了多种 LLM（如 OPT、LLaMA、BLOOM）和任务（如翻译、总结、推理），设置较为全面，提供了不同技术在资源受限环境中的适用性对比。
*   **局限性**：部分技术（如非结构化剪枝）在硬件加速上的效果受限，量化到极低精度时精度损失仍需进一步优化，整体权衡（性能 vs 效率）仍需根据具体应用场景调整。

## Further Thoughts

论文提到的未来方向中，神经架构搜索（NAS）与硬件感知设计结合，为 LLM 压缩提供了新思路，启发我思考是否可以开发动态压缩策略，根据设备资源实时调整量化精度或剪枝比例。此外，量化与蒸馏的联合应用，以及针对异常值的高精度处理（如 LLM.int8() 和 AWQ），提示了混合策略在异构环境中的潜力，未来或许可以探索跨模型、跨任务的压缩技术迁移学习，以提升通用性。