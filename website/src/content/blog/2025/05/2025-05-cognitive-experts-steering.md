---
title: "Two Experts Are All You Need for Steering Thinking: Reinforcing Cognitive Effort in MoE Reasoning Models Without Additional Training"
pubDatetime: 2025-05-20T17:59:16+00:00
slug: "2025-05-cognitive-experts-steering"
type: "arxiv"
id: "2505.14681"
score: 0.7117490358137106
author: "grok-3-latest"
authors: ["Mengru Wang", "Xingyu Chen", "Yue Wang", "Zhiwei He", "Jiahao Xu", "Tian Liang", "Qiuzhi Liu", "Yunzhi Yao", "Wenxuan Wang", "Ruotian Ma", "Haitao Mi", "Ningyu Zhang", "Zhaopeng Tu", "Xiaolong Li", "Dong Yu"]
tags: ["LLM", "MoE Architecture", "Reasoning", "Inference Steering", "Cognitive Efficiency"]
institution: ["Tencent", "Zhejiang University"]
description: "本文提出了一种轻量化的推理时干预方法 Reinforcing Cognitive Experts (RICE)，通过识别并增强 MoE 模型中的认知专家显著提升推理准确性和效率，无需额外训练。"
---

> **Summary:** 本文提出了一种轻量化的推理时干预方法 Reinforcing Cognitive Experts (RICE)，通过识别并增强 MoE 模型中的认知专家显著提升推理准确性和效率，无需额外训练。 

> **Keywords:** LLM, MoE Architecture, Reasoning, Inference Steering, Cognitive Efficiency

**Authors:** Mengru Wang, Xingyu Chen, Yue Wang, Zhiwei He, Jiahao Xu, Tian Liang, Qiuzhi Liu, Yunzhi Yao, Wenxuan Wang, Ruotian Ma, Haitao Mi, Ningyu Zhang, Zhaopeng Tu, Xiaolong Li, Dong Yu

**Institution(s):** Tencent, Zhejiang University


## Problem Background

大型推理模型（Large Reasoning Models, LRMs）在推理任务中表现出强大能力，但基于 Mixture-of-Experts (MoE) 架构的模型常因认知效率问题（如过思考和欠思考）导致性能不佳或资源浪费。
本文旨在解决如何在不进行额外训练的情况下，通过推理时干预提升模型的推理深度和效率。

## Method

*   **核心思想:** 提出一种名为 Reinforcing Cognitive Experts (RICE) 的推理时干预方法，通过识别并增强 MoE 模型中与推理高度相关的‘认知专家’（Cognitive Experts），提升推理能力。
*   **具体步骤:**
    *   **专家识别:** 使用归一化点互信息（normalized Pointwise Mutual Information, nPMI）度量专家与推理标记（如 `<think>` 和 `</think>`）的相关性，计算每个专家的 nPMI 分数，识别出与推理过程高度关联的专家，定义为‘认知专家’。
    *   **权重调整:** 在推理时，对选定的认知专家（通常为排名前两位的专家）应用强化倍数（β），增加其在 MoE 路由中的权重，从而增强推理深度，而不改变模型原始参数。
    *   **参数优化:** 通过实验确定专家数量和强化倍数的合适组合（如选择两个专家，β=64），避免过度干预导致性能下降。
*   **特点:** 该方法轻量化，仅需一次前向传播即可识别专家，无需额外训练或标注数据，且干预过程具有可解释性，直接针对推理行为进行调控。

## Experiment

*   **有效性:** 在 DeepSeek-R1 模型上，RICE 方法将 AIME24 数据集的准确率从 73.3% 提升至 83.3%；在 Qwen3-235B 模型上，AIME25 数据集准确率从 66.7% 提升至 73.3%，表明方法显著提升了推理性能。
*   **效率提升:** 增强认知专家后，模型推理效率提高，例如 DeepSeek-R1 在 AIME24 上生成 token 数从 9,219 减少至 8,317，‘思考’次数从 12.0 减少至 10.2，表明更高效的推理过程。
*   **泛化能力:** 认知专家在跨领域（数学、物理、化学、生物）和未见任务（AIME25）上表现出较好的迁移能力，DeepSeek-R1 平均准确率从 73.4% 提升至 75.6%。
*   **对比优势:** 相较于提示工程和解码约束等方法，RICE 在 AIME 数据集上的平均准确率提升至 78.7%，超过最佳基线 2.0%。
*   **实验设置合理性:** 实验覆盖多个模型（DeepSeek-R1, Qwen3-235B）和领域（数学、科学推理），通过网格搜索优化参数（如专家数量和强化倍数），设置全面且合理；但高强化倍数下性能下降提示了干预强度的潜在风险。

## Further Thoughts

论文中‘认知专家’的概念和其跨领域迁移能力启发了我，MoE 架构中的专家分工类似于人脑功能分区，是否可以通过结合语义分析或动态路由机制进一步精细化专家识别？此外，是否可以设计自适应方法，根据任务复杂性动态调整专家数量和权重？另一个方向是将推理时干预与训练时优化结合，形成混合策略，既保留轻量性又提升长期性能。