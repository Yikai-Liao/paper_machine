---
title: "Unveiling the Compositional Ability Gap in Vision-Language Reasoning Model"
pubDatetime: 2025-05-26T01:42:38+00:00
slug: "2025-05-vlm-compositional-gap"
type: "arxiv"
id: "2505.19406"
score: 0.7758248901574932
author: "grok-3-latest"
authors: ["Tianle Li", "Jihai Zhang", "Yongming Rao", "Yu Cheng"]
tags: ["LLM", "VLM", "Compositional Reasoning", "Cross-Modal", "Reinforcement Learning", "Supervised Fine-Tuning", "Generalization", "Visual Grounding"]
institution: ["The Chinese University of Hong Kong", "Tencent Hunyuan Research"]
description: "本文通过ComPABench基准揭示了视觉-语言模型（VLMs）在跨模态和跨任务组合推理上的显著差距，并提出RL-Ground方法，通过'先描述后推理'和中间步骤奖励显著提升了组合泛化能力。"
---

> **Summary:** 本文通过ComPABench基准揭示了视觉-语言模型（VLMs）在跨模态和跨任务组合推理上的显著差距，并提出RL-Ground方法，通过'先描述后推理'和中间步骤奖励显著提升了组合泛化能力。 

> **Keywords:** LLM, VLM, Compositional Reasoning, Cross-Modal, Reinforcement Learning, Supervised Fine-Tuning, Generalization, Visual Grounding

**Authors:** Tianle Li, Jihai Zhang, Yongming Rao, Yu Cheng

**Institution(s):** The Chinese University of Hong Kong, Tencent Hunyuan Research


## Problem Background

大型视觉-语言模型（VLMs）在单个任务上表现出色，但其是否能通过类似大型语言模型（LLMs）的后训练策略（如强化学习RL）继承强大的组合推理能力仍未被充分探索。
论文聚焦于VLMs在跨模态（Cross-Modal）和跨任务（Cross-Task）场景下的组合泛化能力差距，解决的关键问题是：纯文本训练的推理能力是否能迁移到多模态任务？独立学习的视觉推理技能是否能整合用于复合任务？组合能力是否能泛化到分布外（OOD）场景？

## Method

*   **核心思想:** 通过设计诊断任务和对比不同后训练策略，系统性评估VLMs的组合推理能力，并提出改进方法以弥合能力差距。
*   **任务设计:** 构建ComPABench基准，包含跨模态、跨任务和OOD泛化任务，涉及几何推理（Geometric Reasoning）和空间推理（Spatial Reasoning），以测试模型在多模态和组合场景下的表现。
*   **训练策略:** 对比三种后训练方法：
    *   监督微调（Supervised Fine-Tuning, SFT）：基于负对数似然（NLL）目标，使用成对数据对模型进行任务对齐。
    *   强化学习（Reinforcement Learning, RL）：采用组相对策略优化（Group Relative Policy Optimization, GRPO），通过最终答案奖励优化模型推理能力。
    *   SFT初始化RL（SFT-init RL）：从SFT模型初始化RL训练，以加速收敛并提高稳定性。
*   **改进方法RL-Ground:** 提出一种增强策略，包含两个关键组件：
    *   '先描述后推理'（Caption-Before-Thinking）：强制模型在推理前通过<caption>模块描述视觉内容，促进视觉到文本的转换和对齐。
    *   中间步骤奖励（Progress Reward）：在推理的中间步骤（如形状面积计算或距离估计）提供细粒度监督，而非仅依赖最终答案奖励，增强组合推理能力。
*   **实现细节:** 使用Qwen2.5-VL-3B和7B模型，训练数据量为4K样本/任务，评估数据量为500样本/任务，确保实验的可控性和可重复性。

## Experiment

*   **跨模态泛化（RQ1）:** 纯文本训练的模型在多模态任务上表现极差（如SFT模型准确率从99.8%降至4.2%），表明文本推理能力无法直接迁移到视觉输入；RL略优于SFT（如7B模型多模态准确率从20.8%提升至28.0%），但差距仍显著；从纯文本初始化多模态RL可进一步提升性能（如3B模型准确率从49.6%提升至64.4%）。
*   **跨任务组合（RQ2）:** SFT在组合任务上表现极差（如纯文本准确率仅0.6%），表明其无法整合独立技能；RL显著优于SFT（如纯文本准确率提升至93%）；RL-Ground在多模态组合任务上表现最佳（如7B模型准确率达52.8%），验证了视觉-文本对齐和中间奖励的有效性。
*   **OOD泛化（RQ3）:** SFT在OOD任务上表现不稳定（如最大面积任务准确率仅1.4%）；RL表现出较强泛化能力（如7B模型组合任务准确率达40.4%）；RL-Ground在所有OOD任务中表现最佳（如7B模型组合任务准确率达52.8%）。
*   **实验设置评价:** 实验设计全面，涵盖不同模型规模（3B和7B）、训练策略和任务类型，数据量充足（4K训练样本，500评估样本）；但任务基于合成数据，可能无法完全反映真实世界多模态场景的复杂性。

## Further Thoughts

RL-Ground的'先描述后推理'策略启发了我，这种将视觉输入转化为文本描述的方法类似于人类分步处理复杂任务的思维方式，或许可以通过自适应提示（Adaptive Prompting）进一步优化，根据任务难度动态调整描述的详细程度；此外，组合能力差距的揭示也让我思考，是否可以在预训练阶段引入更多跨模态组合数据，以减少后训练阶段的泛化负担，从而提升VLMs在真实场景中的表现。