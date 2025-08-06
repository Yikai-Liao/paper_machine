---
title: "MolReasoner: Toward Effective and Interpretable Reasoning for Molecular LLMs"
pubDatetime: 2025-08-04T05:10:11+00:00
slug: "2025-08-molecular-reasoning-llm"
type: "arxiv"
id: "2508.02066"
score: 0.7569148774336469
author: "grok-3-latest"
authors: ["Guojiang Zhao", "Sihang Li", "Zixiang Lu", "Zheng Cheng", "Haitao Lin", "Lirong Wu", "Hanchen Xia", "Hengxing Cai", "Wentao Guo", "Hongshuai Wang", "Mingjun Xu", "Siyu Zhu", "Guolin Ke", "Linfeng Zhang", "Zhifeng Gao"]
tags: ["LLM", "Molecular Reasoning", "Reinforcement Learning", "Chain of Thought", "Supervised Fine-Tuning"]
institution: ["DP Technology, Beijing, China", "AI for Science Institute, Beijing, China", "Shanghai Jiao Tong University, Shanghai, China", "Fudan University, Shanghai, China"]
description: "本文提出 MolReasoner，一个通过监督微调和强化学习的两阶段框架，显著提升大型语言模型在分子任务中的推理能力、可解释性和生成质量。"
---

> **Summary:** 本文提出 MolReasoner，一个通过监督微调和强化学习的两阶段框架，显著提升大型语言模型在分子任务中的推理能力、可解释性和生成质量。 

> **Keywords:** LLM, Molecular Reasoning, Reinforcement Learning, Chain of Thought, Supervised Fine-Tuning

**Authors:** Guojiang Zhao, Sihang Li, Zixiang Lu, Zheng Cheng, Haitao Lin, Lirong Wu, Hanchen Xia, Hengxing Cai, Wentao Guo, Hongshuai Wang, Mingjun Xu, Siyu Zhu, Guolin Ke, Linfeng Zhang, Zhifeng Gao

**Institution(s):** DP Technology, Beijing, China, AI for Science Institute, Beijing, China, Shanghai Jiao Tong University, Shanghai, China, Fudan University, Shanghai, China


## Problem Background

大型语言模型（LLMs）在分子科学领域的推理能力尚未充分挖掘，现有方法多依赖通用提示，缺乏领域特定的分子语义，导致生成结果化学上不合理；同时，单纯微调方法缺乏中间推理指导，倾向于记忆而非推理，泛化能力和可解释性较差。
论文试图解决的核心问题是：如何让 LLMs 从记忆转向真正的分子推理，提升在分子任务中的准确性、可解释性和泛化能力。

## Method

*   **核心思想:** 提出 MolReasoner，一个两阶段训练框架，通过监督微调和强化学习，从记忆转向分子推理，提升模型的化学理解和推理深度。
*   **Mol-SFT（Molecular Supervised Fine-Tuning）阶段:** 
    *   利用 GPT-4o 生成约 42,000 个高质量的合成 Chain-of-Thought（CoT）数据（包括分子描述和生成任务），通过知识引导提示和化学准确性验证，确保数据质量。
    *   以标准自回归语言建模目标进行微调，使模型学习结构化推理格式、领域术语和初步推理能力，为后续强化学习奠定基础。
*   **Mol-RL（Molecular Reinforcement Learning）阶段:** 
    *   采用 Group Relative Policy Optimization（GRPO）算法，通过生成多个候选响应并基于奖励函数优化推理路径和生成结果。
    *   设计多级奖励函数：
        -   分子描述任务：结合格式准确性（Format Accuracy）和语言相似性（基于 BLEU、ROUGE、METEOR 等指标）。
        -   分子生成任务：结合格式准确性、结构相似性（包括指纹相似性、SELFIES 相似性、片段相似性和官能团匹配），确保化学结构的准确性和语义一致性。
    *   通过多级奖励反馈，从全局分子语义到局部结构细节对齐化学知识，提升生成结果的化学合理性。
*   **关键创新:** 两阶段方法解决了‘冷启动’问题，先通过 Mol-SFT 初始化推理能力，再通过 Mol-RL 深度优化，确保模型从‘有效’生成转向‘高质量’生成，同时增强可解释性。

## Experiment

*   **有效性:** 实验在 ChEBI-20 数据集上进行，MolReasoner 在分子描述和文本生成分子任务中显著优于基线模型（如 GPT-4o、Qwen 系列、LLaMA 系列及 Mol-Instructions），例如分子描述任务 BLEU-2 得分达 0.4383（基线最高 0.1670），分子生成任务 BLEU 得分达 0.7841（基线最高 0.3049）。
*   **合理性:** 实验设置全面，涵盖多种模型规模和方法类型，评价指标多样（包括 BLEU、ROUGE、METEOR、分子相似性等），并验证生成分子的化学有效性；消融研究进一步确认了 Mol-SFT 和 Mol-RL 各阶段及奖励函数的贡献。
*   **局限性:** 依赖 GPT-4o 合成数据可能引入偏差，奖励函数未考虑合成可行性或三维结构，计算成本较高（Mol-RL 训练需 1200 GPU 小时），但整体实验设计严谨，数据支持方法有效性。

## Further Thoughts

MolReasoner 的两阶段训练框架（从监督微调到强化学习）为其他领域的复杂推理任务提供了借鉴，尤其是在需要结合领域知识和推理能力的场景中；多级奖励函数设计启发我们在强化学习中针对特定领域构建细粒度反馈机制；此外，利用通用模型生成领域特定推理数据来‘冷启动’模型的策略，可广泛应用于数据稀缺领域。发散性思考：是否可以结合多模态数据（如分子图、三维结构）进一步提升分子理解？奖励函数是否能引入实验验证数据以提高实用性？