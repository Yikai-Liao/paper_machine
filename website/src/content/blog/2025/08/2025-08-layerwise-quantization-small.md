---
title: "Exploring Layer-wise Information Effectiveness for Post-Training Quantization in Small Language Models"
pubDatetime: 2025-08-05T11:17:04+00:00
slug: "2025-08-layerwise-quantization-small"
type: "arxiv"
id: "2508.03332"
score: 0.8242998405200755
author: "grok-3-latest"
authors: ["He Xiao", "Qingyao Yang", "Dirui Xie", "Wendong Xu", "Wenyong Zhou", "Haobo Liu", "Zhengwu Liu", "Ngai Wong"]
tags: ["LLM", "Small Models", "Quantization", "Layer-wise Analysis", "Compression"]
institution: ["The University of Hong Kong", "Huazhong University of Science and Technology"]
description: "本文提出LieQ框架，通过层级信息有效性评估自动分配比特宽度，在极低比特量化下显著减少小型语言模型精度损失，同时保持硬件友好性。"
---

> **Summary:** 本文提出LieQ框架，通过层级信息有效性评估自动分配比特宽度，在极低比特量化下显著减少小型语言模型精度损失，同时保持硬件友好性。 

> **Keywords:** LLM, Small Models, Quantization, Layer-wise Analysis, Compression

**Authors:** He Xiao, Qingyao Yang, Dirui Xie, Wendong Xu, Wenyong Zhou, Haobo Liu, Zhengwu Liu, Ngai Wong

**Institution(s):** The University of Hong Kong, Huazhong University of Science and Technology


## Problem Background

大型语言模型（LLMs）因参数量巨大，在边缘设备上部署面临内存和延迟挑战，即便是参数小于7B的小型语言模型（Small Language Models, SLMs）也常超出设备内存限制（4-12GB）。
后训练量化（Post-Training Quantization, PTQ）是压缩模型的关键技术，但现有方法在极低比特（如2比特）量化时会导致小型模型精度严重下降，因为这些模型缺乏冗余来抵消量化噪声。
论文针对三个挑战：如何在保持精度的同时实现结构化量化？如何量化层级敏感性指导压缩？如何在极低比特下平衡硬件效率和精度鲁棒性？

## Method

*   **核心思想:** 提出LieQ框架，通过层级信息有效性（Layer-wise Information Effectiveness）评估Transformer模型中每一层的贡献，自动分配比特宽度，在极低比特量化下保护关键层信息，同时保持硬件友好性。
*   **具体实现:** 
    *   **诊断指标:** 引入三个互补指标评估层级重要性：
        - **Perplexity Drop（困惑度下降）**：移除某层后测量模型预测性能变化，反映该层对整体预测的贡献。
        - **Representational Compactness（表示紧凑性）**：基于注意力机制投影表示的谱特性，比较训练后与随机初始化的表示，量化训练带来的结构化信息增益。
        - **Top-k Energy Gain（前k能量增益）**：测量训练后表示中前k个主成分的能量占比变化，反映低秩结构强度。
    *   **分数计算:** 将三个指标归一化并加权组合为统一的层级信息有效性分数（Layer-wise Information Effectiveness Score），用于识别最关键的层。
    *   **比特分配:** 采用‘层内统一、层间混合’策略，对高分层分配较高比特（如4比特），对低分层分配较低比特（如2比特），避免不规则内存布局。
    *   **集成性:** 可与现有量化方法（如GPTQ, AWQ）结合，进一步优化压缩效果。
*   **关键优势:** 无需额外微调或重训练，计算开销低（仅需少量前向传播），且支持硬件加速。

## Experiment

*   **有效性:** LieQ在极低比特量化下显著优于现有方法，在Qwen3-4B模型上以2.05比特量化恢复了FP16基准性能的95.9%，比GPTQ和AWQ分别高出19.7%和18.1%；在LLaMA3.2-3B上以2.07比特量化保持了98.2%的基准精度。
*   **压缩比:** 实现了高达4倍的内存压缩，显著降低边缘设备部署门槛。
*   **硬件友好性:** 层内统一比特分配避免了不规则内存布局，在RTX 4090 GPU上推理延迟低于其他混合精度方法，保持了GPU tensor-core的高吞吐量。
*   **实验设置:** 实验覆盖多种小型模型（Qwen3, LLaMA3.x）和任务（语言生成、推理），数据集包括WikiText-2, C4及7个零样本推理任务，指标全面（困惑度、准确率）；消融研究验证了不同4比特层数量的影响，设置合理。
*   **局限性:** 实验主要在高端GPU上进行，未充分验证在实际边缘设备（如移动SoC）上的性能。

## Further Thoughts

LieQ通过层级诊断指标揭示了Transformer模型中各层的信息贡献差异，这种思路可扩展至其他压缩技术（如剪枝）或模型架构设计优化；例如，是否可以在预训练阶段基于这些指标设计更紧凑的模型，减少冗余层？此外，层级信息有效性分数是否能动态调整以适应不同任务需求（如推理 vs. 生成）？另一个方向是结合硬件特性（如加速器支持的比特配置）进一步优化比特分配策略，提升实际部署效率。