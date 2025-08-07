---
title: "MolReasoner: Toward Effective and Interpretable Reasoning for Molecular LLMs"
pubDatetime: 2025-08-04T05:10:11+00:00
slug: "2025-08-molecular-reasoning-llm"
type: "arxiv"
id: "2508.02066"
score: 0.7569148774336469
author: "grok-3-latest"
authors: ["Guojiang Zhao", "Sihang Li", "Zixiang Lu", "Zheng Cheng", "Haitao Lin", "Lirong Wu", "Hanchen Xia", "Hengxing Cai", "Wentao Guo", "Hongshuai Wang", "Mingjun Xu", "Siyu Zhu", "Guolin Ke", "Linfeng Zhang", "Zhifeng Gao"]
tags: ["LLM", "Molecular Reasoning", "Chain of Thought", "Reinforcement Learning", "Structure Alignment"]
institution: ["DP Technology, Beijing, China", "AI for Science Institute, Beijing, China", "Shanghai Jiao Tong University, Shanghai, China", "Fudan University, Shanghai, China"]
description: "本文提出MolReasoner两阶段框架，通过合成CoT数据和强化学习，将大型语言模型从记忆导向转向分子推理，显著提升了分子任务中的准确性和可解释性。"
---

> **Summary:** 本文提出MolReasoner两阶段框架，通过合成CoT数据和强化学习，将大型语言模型从记忆导向转向分子推理，显著提升了分子任务中的准确性和可解释性。 

> **Keywords:** LLM, Molecular Reasoning, Chain of Thought, Reinforcement Learning, Structure Alignment

**Authors:** Guojiang Zhao, Sihang Li, Zixiang Lu, Zheng Cheng, Haitao Lin, Lirong Wu, Hanchen Xia, Hengxing Cai, Wentao Guo, Hongshuai Wang, Mingjun Xu, Siyu Zhu, Guolin Ke, Linfeng Zhang, Zhifeng Gao

**Institution(s):** DP Technology, Beijing, China, AI for Science Institute, Beijing, China, Shanghai Jiao Tong University, Shanghai, China, Fudan University, Shanghai, China


## Problem Background

大型语言模型（LLMs）在多个领域表现出色，但在分子推理任务中存在显著不足。
现有方法主要依赖通用提示（Prompt-based Methods），由于缺乏化学领域特定语义，导致模型难以准确捕捉分子结构信息，常生成化学上不合理的结构。
此外，单纯的微调方法（Fine-tuning without Explicit Reasoning）缺乏显式推理指导，倾向于记忆而非理解化学原理，泛化能力和可解释性较差。
论文旨在解决核心问题：如何超越记忆，让LLMs在分子任务中实现真正的推理能力，以支持药物发现和材料设计等应用。

## Method

*   **核心思想:** 提出MolReasoner，一个两阶段训练框架，通过从记忆转向推理，提升LLMs在分子任务中的准确性和可解释性。
*   **第一阶段 - Mol-SFT (Molecular Supervised Fine-Tuning):** 
    *   利用GPT-4o生成合成Chain-of-Thought（CoT）数据，涵盖分子描述和生成任务，通过知识引导的提示模板和化学结构信息（如分子量、功能团）确保推理轨迹的化学准确性。
    *   使用约42,000个高质量CoT样本进行监督微调，基于自回归语言建模目标训练模型，使其掌握初步推理格式和领域特定术语。
    *   这一阶段为模型提供浅层推理先验，为后续强化学习奠定基础。
*   **第二阶段 - Mol-RL (Molecular Reinforcement Learning):** 
    *   采用Group Relative Policy Optimization（GRPO）算法，通过生成多个候选响应并基于奖励函数评估，优化推理路径和生成结果。
    *   设计多层次奖励函数：对于分子描述任务，使用语言相似度奖励（基于BLEU、ROUGE等指标）；对于分子生成任务，使用结构相似度奖励（包括指纹相似度、SELFIES相似度、片段相似度和功能团匹配）。
    *   通过化学感知的反馈机制，确保生成结果在全局语义和局部结构上与输入一致，进一步提升推理深度。
*   **关键创新:** 通过CoT数据解决冷启动问题，结合强化学习和多维度奖励函数，实现从记忆到推理的范式转变，同时增强模型对分子结构的理解和输出可解释性。

## Experiment

*   **有效性:** 实验在ChEBI-20数据集上进行，MolReasoner在分子描述和文本生成分子任务中显著优于基线模型（如GPT-4o、Qwen系列、Mol-Instructions）。例如，在分子描述任务中，BLEU-2分数提升至0.4383（基线最高为0.1670），在分子生成任务中，BLEU分数达到0.7841（基线最高为0.3049），显示出方法在准确性和化学一致性上的明显提升。
*   **合理性:** 实验设置全面，涵盖语言生成指标（BLEU、ROUGE、METEOR）和化学结构指标（Tanimoto相似度、功能团匹配等），同时通过消融研究验证了Mol-SFT和Mol-RL各阶段及奖励函数的作用，证明了两阶段框架的协同效应。
*   **局限性与成本:** 论文指出依赖GPT-4o生成的CoT数据可能引入偏差，奖励函数未考虑合成可行性和3D结构，且Mol-RL阶段计算成本较高（约1200 GPU小时），对大规模应用构成挑战。

## Further Thoughts

论文中的合成CoT数据解决冷启动问题的思路令人启发，是否可以在其他缺乏专家标注数据的科学领域（如物理学、材料科学）中，通过通用模型生成初始推理数据来引导模型学习？
此外，多层次奖励函数的设计结合全局语义和局部结构信息，确保化学一致性，这种方法是否可以推广到其他跨模态任务（如图像-文本生成），通过多维度奖励优化复杂数据理解？
最后，从记忆到推理的范式转变强调可解释性，是否意味着未来AI模型需要在训练中更多融入领域知识和逻辑推理，而不仅仅依赖数据驱动模式？