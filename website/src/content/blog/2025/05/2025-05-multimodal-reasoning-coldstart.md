---
title: "Advancing Multimodal Reasoning via Reinforcement Learning with Cold Start"
pubDatetime: 2025-05-28T13:21:38+00:00
slug: "2025-05-multimodal-reasoning-coldstart"
type: "arxiv"
id: "2505.22334"
score: 0.6955345863541152
author: "grok-3-latest"
authors: ["Lai Wei", "Yuting Li", "Kaipeng Zheng", "Chen Wang", "Yue Wang", "Linghe Kong", "Lichao Sun", "Weiran Huang"]
tags: ["LLM", "Multimodal Reasoning", "Reinforcement Learning", "Supervised Fine-Tuning", "Chain of Thought"]
institution: ["Shanghai Jiao Tong University", "Shanghai Innovation Institute", "Zhongguancun Academy", "Lehigh University"]
description: "本文提出一种两阶段训练框架（SFT+RL），通过冷启动监督微调和强化学习显著提升多模态大语言模型在数学推理任务中的性能，实现了3B和7B规模的开源模型中最先进结果。"
---

> **Summary:** 本文提出一种两阶段训练框架（SFT+RL），通过冷启动监督微调和强化学习显著提升多模态大语言模型在数学推理任务中的性能，实现了3B和7B规模的开源模型中最先进结果。 

> **Keywords:** LLM, Multimodal Reasoning, Reinforcement Learning, Supervised Fine-Tuning, Chain of Thought

**Authors:** Lai Wei, Yuting Li, Kaipeng Zheng, Chen Wang, Yue Wang, Linghe Kong, Lichao Sun, Weiran Huang

**Institution(s):** Shanghai Jiao Tong University, Shanghai Innovation Institute, Zhongguancun Academy, Lehigh University


## Problem Background

多模态大语言模型（MLLMs）在复杂推理任务（如数学推理）中的潜力尚未充分挖掘，尤其是在结合视觉和文本信息时。
现有研究表明强化学习（RL）能提升大型语言模型（LLMs）的推理能力，但其在多模态领域的应用仍不成熟，且所谓的‘顿悟时刻’（Aha Moment）模式在RL前已存在于MLLMs中，却不一定与推理能力提升相关。
因此，关键问题是如何设计有效的训练策略，确保MLLMs在多模态推理任务中真正提升性能。

## Method

*   **核心思想:** 提出一种两阶段训练框架，通过监督微调（SFT）作为冷启动（Cold Start）为模型注入结构化推理模式，随后利用强化学习（RL）进一步激活和优化推理能力。
*   **第一阶段 - 冷启动（SFT）:**
    *   设计多种链式思维（Chain-of-Thought, CoT）数据集用于监督微调，包括：
        - **Distilled-CoT**：通过从更大模型（如Qwen2.5-VL-7B和32B）蒸馏数据，利用拒绝采样（Rejection Sampling）生成高质量推理轨迹。
        - **Reflection-CoT**：引入反思模式，结合正确和错误推理路径，模拟‘重新评估’过程，增强自我纠正能力。
        - **Caption-CoT**：先描述图像内容再推理，强调视觉信息提取。
        - **Self-Critic-CoT**：通过自我评论和迭代改进，生成结构化推理步骤。
    *   目标是为后续RL阶段提供坚实的推理基础。
*   **第二阶段 - 强化学习（RL）:**
    *   采用GRPO（Group Reward Policy Optimization）算法，通过对一组响应进行奖励评估优化模型策略。
    *   GRPO直接利用组归一化奖励估计优势，无需额外价值模型，并引入KL散度约束避免过度偏离参考模型。
    *   目标是进一步提升SFT阶段建立的推理能力，特别是在多模态数学推理任务中。

## Experiment

*   **有效性:** 提出的SFT+RL组合方法在3B和7B规模的Qwen2.5-VL模型上显著提升性能，例如7B模型在MathVista上从66.3%提升至73.4%，在We-Math上从62.9%提升至70.4%，平均提升6.19个百分点；3B模型平均提升10.84个百分点，甚至在部分任务上与7B模型性能相当。
*   **优越性:** 与基线模型（如MM-Eureka, VLAA-Thinker）相比，本文方法在同等规模下实现最先进性能，7B模型超越部分更大规模模型（如GPT-4o, Skywork R1V-38B）。
*   **实验设置合理性:** 实验设计全面，涵盖多种CoT数据策略的消融研究，验证了冷启动阶段数据质量和策略选择对最终性能的影响；数据集（如MathVision, MathVista）覆盖多模态数学推理多个方面，具有代表性。
*   **局限性:** 实验仅限于3B和7B规模，未验证更大模型适用性；RL算法仅采用GRPO，未与其他方法对比。

## Further Thoughts

冷启动策略（SFT）对RL性能的显著影响启发我们，是否可以在其他领域（如纯文本推理或代码生成）设计类似‘预热’阶段，利用高质量数据奠定基础？
此外，‘顿悟时刻’与推理能力提升无关的发现，提示我们是否应通过更精细指标（如推理步骤逻辑一致性）评估模型真实能力？
最后，推理结构比答案正确性更关键的结论，是否意味着在数据稀缺场景下可通过合成结构化推理轨迹提升性能？