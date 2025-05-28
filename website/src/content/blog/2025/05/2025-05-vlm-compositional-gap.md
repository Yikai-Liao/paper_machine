---
title: "Unveiling the Compositional Ability Gap in Vision-Language Reasoning Model"
pubDatetime: 2025-05-26T01:42:38+00:00
slug: "2025-05-vlm-compositional-gap"
type: "arxiv"
id: "2505.19406"
score: 0.7758248901574932
author: "grok-3-latest"
authors: ["Tianle Li", "Jihai Zhang", "Yongming Rao", "Yu Cheng"]
tags: ["Vision-Language Model", "Compositional Reasoning", "Reinforcement Learning", "Supervised Fine-Tuning", "Cross-Modal Generalization"]
institution: ["The Chinese University of Hong Kong", "Tencent Hunyuan Research"]
description: "本文通过 ComPABench 基准揭示了视觉-语言模型（VLMs）在组合推理上的差距，并提出 RL-Ground 方法，通过 '先描述后推理' 和中间步骤奖励显著提升了跨模态、跨任务和分布外推理能力。"
---

> **Summary:** 本文通过 ComPABench 基准揭示了视觉-语言模型（VLMs）在组合推理上的差距，并提出 RL-Ground 方法，通过 '先描述后推理' 和中间步骤奖励显著提升了跨模态、跨任务和分布外推理能力。 

> **Keywords:** Vision-Language Model, Compositional Reasoning, Reinforcement Learning, Supervised Fine-Tuning, Cross-Modal Generalization

**Authors:** Tianle Li, Jihai Zhang, Yongming Rao, Yu Cheng

**Institution(s):** The Chinese University of Hong Kong, Tencent Hunyuan Research


## Problem Background

大型视觉-语言模型（VLMs）在多模态推理任务上的组合能力（Compositional Reasoning）尚未被充分探索，尤其是在跨模态（纯文本到视觉输入）、跨任务（整合不同推理技能）和分布外（OOD）场景下。
尽管大型语言模型（LLMs）通过强化学习（RL）等后训练策略展现了强大的推理能力，但 VLMs 是否能继承类似能力仍是一个开放问题。
论文旨在揭示当前训练策略（如监督微调 SFT 和 RL）的局限性，并解决 VLMs 在组合泛化上的关键差距。

## Method

*   **诊断基准设计 (ComPABench):** 构建了一个系统性基准，包含跨模态、跨任务和分布外（OOD）任务，用于评估 VLMs 的组合推理能力。具体任务包括几何推理（计算形状面积）、空间推理（确定网格位置）及其组合变体，覆盖纯文本和多模态输入两种形式。
*   **后训练策略对比:** 对比了三种训练方法：
    *   监督微调（SFT）：基于负对数似然（NLL）目标，使用成对数据（输入-输出）对模型进行微调，强调语法正确性和语义对齐。
    *   强化学习（RL with GRPO）：采用组相对策略优化（Group Relative Policy Optimization），通过最终答案正确性和格式合规性等奖励信号优化模型，同时引入 KL 正则化项以保持与参考策略的一致性。
    *   SFT 初始化 RL（SFT-init RL）：从 SFT 训练的检查点初始化 RL 训练，旨在加速收敛并减少早期不稳定性。
*   **创新方法 RL-Ground:** 提出了一种改进的 RL 训练策略，包含两个核心组件：
    *   '先描述后推理'（Caption-Before-Thinking）：强制模型在推理前通过 `<caption>` 模块将视觉内容转化为自然语言描述，促进视觉-文本对齐。
    *   中间步骤奖励（Progress Reward）：在推理的中间步骤（如形状面积计算或距离估计）提供细粒度监督，而不仅仅依赖最终答案的奖励，增强组合推理能力。
*   **实现细节:** 实验基于 Qwen2.5-VL-3B 和 7B 模型，训练数据量为每任务 4K 样本，测试数据量为 500 样本，确保评估的全面性和可靠性。

## Experiment

*   **跨模态泛化效果 (RQ1):** 纯文本训练的模型在多模态任务上表现大幅下降，例如 SFT 在网格位置任务上从 99.8% 降至 4.2%，表明纯文本推理能力难以迁移到视觉输入。RL 略有改进（如 7B 模型在多模态形状面积任务上从 20.8% 提升至 28.0%），但仍远低于纯文本表现。初始化多模态 RL 时，使用纯文本训练的模型可显著提升性能（如 3B 模型网格位置任务从 49.6% 提升至 64.4%）。
*   **跨任务组合效果 (RQ2):** SFT 在独立任务上表现优秀，但在组合任务上几乎失败（多模态组合任务准确率仅 2.2%-7.2%），而 RL 表现较好（7B 模型达 31.2%）。RL-Ground 进一步提升至 52.8%（7B），显示出结构化提示和中间奖励的显著效果。
*   **分布外泛化效果 (RQ3):** SFT 在 OOD 任务上表现不稳定（例如最大面积任务准确率仅 1.4%-1.8%），而 RL 表现出较强泛化能力（7B 模型在 OOD 组合任务上达 40.4%）。RL-Ground 在 OOD 组合任务上达到 52.8%（7B），与分布内表现一致，鲁棒性突出。
*   **实验设置合理性:** 实验基于 Qwen2.5-VL-3B 和 7B 模型，覆盖多种任务类型和模型规模，数据量充足（4K 训练样本，500 测试样本），任务设计精细（包括纯文本和多模态变体），支持结论的可信度。

## Further Thoughts

RL-Ground 的 '先描述后推理' 策略通过将视觉输入转化为文本描述，构建了一种模态转换的中间步骤，这启发我们可以在多模态模型中引入更多中间表示层来分解复杂任务。此外，Progress Reward 表明细粒度监督对组合推理至关重要，未来是否可以探索自适应奖励机制，根据任务难度动态调整中间奖励权重？或者通过无监督方式生成中间描述，减少对人工标注的依赖？