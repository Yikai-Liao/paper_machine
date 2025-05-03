---
title: "Phi-4-reasoning Technical Report"
pubDatetime: 2025-04-30T05:05:09+00:00
slug: "2025-04-phi4-reasoning-enhancement"
type: "arxiv"
id: "2504.21318"
score: 0.7278407750426621
author: "grok-3-latest"
authors: ["Marah Abdin", "Sahaj Agarwal", "Ahmed Awadallah", "Vidhisha Balachandran", "Harkirat Behl", "Lingjiao Chen", "Gustavo de Rosa", "Suriya Gunasekar", "Mojan Javaheripi", "Neel Joshi", "Piero Kauffmann", "Yash Lara", "Caio César Teodoro Mendes", "Arindam Mitra", "Besmira Nushi", "Dimitris Papailiopoulos", "Olli Saarikivi", "Shital Shah", "Vaishnavi Shrivastava", "Vibhav Vineet", "Yue Wu", "Safoora Yousefi", "Guoqing Zheng"]
tags: ["LLM", "Reasoning", "Supervised Fine-Tuning", "Reinforcement Learning", "Inference Scaling"]
institution: ["Microsoft"]
description: "本文通过监督微调和强化学习，基于 14B 参数的 Phi-4 模型开发出 Phi-4-reasoning 和 Phi-4-reasoning-plus，显著提升复杂推理任务性能并展现出与更大规模模型的竞争力。"
---

> **Summary:** 本文通过监督微调和强化学习，基于 14B 参数的 Phi-4 模型开发出 Phi-4-reasoning 和 Phi-4-reasoning-plus，显著提升复杂推理任务性能并展现出与更大规模模型的竞争力。 

> **Keywords:** LLM, Reasoning, Supervised Fine-Tuning, Reinforcement Learning, Inference Scaling

**Authors:** Marah Abdin, Sahaj Agarwal, Ahmed Awadallah, Vidhisha Balachandran, Harkirat Behl, Lingjiao Chen, Gustavo de Rosa, Suriya Gunasekar, Mojan Javaheripi, Neel Joshi, Piero Kauffmann, Yash Lara, Caio César Teodoro Mendes, Arindam Mitra, Besmira Nushi, Dimitris Papailiopoulos, Olli Saarikivi, Shital Shah, Vaishnavi Shrivastava, Vibhav Vineet, Yue Wu, Safoora Yousefi, Guoqing Zheng

**Institution(s):** Microsoft


## Problem Background

大型语言模型（LLMs）在复杂推理任务（如数学、科学、编码）中需要更多的推理时计算（inference-time compute）来提升性能，而小型模型由于资源限制往往表现不足。
本文旨在通过监督微调（SFT）和强化学习（RL），基于 14B 参数的 Phi-4 模型，开发出在推理任务上具有竞争力的 Phi-4-reasoning 和 Phi-4-reasoning-plus，解决小型模型在推理能力上的关键瓶颈，同时探索推理技能对通用任务的泛化影响。

## Method

*   **核心思想:** 通过数据驱动的监督微调（SFT）和强化学习（RL），增强小型语言模型（14B 参数）的推理能力，使其在复杂任务上接近甚至超越更大规模模型，同时保持通用能力和安全性。
*   **监督微调（SFT）细节:** 
    *   基于 Phi-4 模型，使用超过 140 万个精心筛选的提示-响应对（共 83 亿 token）进行训练，数据覆盖 STEM、编码和安全领域。
    *   提示筛选为‘可教’样本（即处于模型能力边界的问题），通过 LLM 评估和启发式难度估计，确保学习效果最大化。
    *   引入 <think> 和 </think> 标记结构化推理过程，将上下文长度从 16K 扩展到 32K 以支持长推理链。
    *   数据混合（data mixture）优化，通过探索和扩展阶段调整不同领域数据的权重，确保训练效率。
*   **强化学习（RL）细节:** 
    *   在 Phi-4-reasoning 基础上，使用 Group Relative Policy Optimization (GRPO) 算法进一步优化数学推理能力。
    *   训练数据为约 6K 个数学问题，奖励函数设计为‘长度感知准确性’（length-aware accuracy），鼓励正确答案时简洁，错误答案时深入思考，同时惩罚重复和格式错误。
    *   使用 32 Nvidia H100 GPU，批大小 64，学习率 5e-8，训练 90 步，显著提升数学任务性能。
*   **数据策略:** 
    *   种子数据从公开网站、现有数据集和合成问题中筛选，响应由 o3-mini 模型生成，确保高质量推理轨迹。
    *   对训练数据进行严格去污染处理，避免与评估基准重叠。
*   **关键创新:** 不改变模型架构，仅通过训练策略和数据质量提升推理能力，同时通过系统消息和格式化输出增强推理一致性。

## Experiment

*   **有效性:** Phi-4-reasoning 和 Phi-4-reasoning-plus 在多个推理基准（如 AIME、Omni-MATH、GPQA、LiveCodeBench）上显著优于基础模型 Phi-4，数学任务准确率提升超过 50 个百分点，编码任务提升超过 25 个百分点；相比更大规模模型（如 DeepSeek-R1-Distill-Llama-70B 和 o1-mini），Phi-4-reasoning-plus 在多个任务上表现出竞争力，甚至接近 DeepSeek-R1（671B 参数）的水平。
*   **全面性:** 实验覆盖数学、科学、编码、算法问题解决、规划和空间推理等领域，基准选择多样化；特别关注 AIME 2025 的无污染性（数据在训练后发布），并通过 50 次独立运行减少采样方差，增强统计稳健性。
*   **权衡与局限:** Phi-4-reasoning-plus 在数学任务上准确率更高，但生成 token 长度平均比 Phi-4-reasoning 长 1.5 倍，推理成本增加；模型在生物、化学和离散数学等领域的提升较小，表明仍有改进空间。
*   **泛化性:** 推理训练带来通用基准（如 IFEval、ArenaHard）上的非平凡提升，表明推理技能具有跨领域泛化能力。

## Further Thoughts

论文通过筛选‘可教’提示和高质量合成数据，证明小型模型可以通过数据策略达到大模型性能，这启发我思考如何在多语言或低资源任务中应用类似筛选策略；此外，推理时计算的动态分配（如 best-of-N 策略）显著提升性能，是否可以通过自适应策略根据任务难度动态调整计算资源？最后，RL 仅针对数学训练却对其他推理任务有提升，是否可以通过多任务 RL 或元学习进一步增强跨领域泛化能力？