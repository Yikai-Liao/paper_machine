---
title: "DeepCritic: Deliberate Critique with Large Language Models"
pubDatetime: 2025-05-01T17:03:17+00:00
slug: "2025-05-deepcritic-math-critique"
type: "arxiv"
id: "2505.00662"
score: 0.721612304706838
author: "grok-3-latest"
authors: ["Wenkai Yang", "Jingwen Chen", "Yankai Lin", "Ji-Rong Wen"]
tags: ["LLM", "Critique Model", "Mathematical Reasoning", "Supervised Fine-Tuning", "Reinforcement Learning"]
institution: ["Renmin University of China", "Beijing Jiaotong University"]
description: "本文提出 DeepCritic 框架，通过两阶段训练（监督微调与强化学习）显著提升大型语言模型在数学推理任务中的批判能力，为自动化监督和模型自我改进铺平道路。"
---

> **Summary:** 本文提出 DeepCritic 框架，通过两阶段训练（监督微调与强化学习）显著提升大型语言模型在数学推理任务中的批判能力，为自动化监督和模型自我改进铺平道路。 

> **Keywords:** LLM, Critique Model, Mathematical Reasoning, Supervised Fine-Tuning, Reinforcement Learning

**Authors:** Wenkai Yang, Jingwen Chen, Yankai Lin, Ji-Rong Wen

**Institution(s):** Renmin University of China, Beijing Jiaotong University


## Problem Background

随着大型语言模型（LLMs）能力的快速提升，提供准确、有效的反馈和可扩展的监督成为迫切需求。
当前 LLM 批判模型在复杂领域（如数学推理）中生成的反馈过于浅显，缺乏深入分析和批判性思维，导致判断准确性低，无法为生成模型提供有效改进指导，限制了自动化监督的潜力。

## Method

*   **核心思想:** 提出 DeepCritic 框架，通过两阶段训练提升 LLM 批判模型在数学推理任务中的批判深度和准确性，使其能够对每一步推理进行深思熟虑的分析和反馈。
*   **第一阶段 - 监督微调（Supervised Fine-Tuning, SFT）:**
    *   使用 Qwen2.5-72B-Instruct 模型生成 4.5K 条长篇批判数据作为种子数据集。
    *   数据生成过程包括：对每个推理步骤生成初步批判（Initial Critique），基于初步批判生成深入批判（In-Depth Critique）以从不同视角验证步骤或反思初步批判的缺陷，最后合并两者为深思熟虑的批判（Deliberate Critique）。
    *   在种子数据上进行监督微调，使模型初步具备多视角评估和元批判（Meta-Critiquing）能力。
*   **第二阶段 - 强化学习（Reinforcement Learning, RL）:**
    *   在 SFT 模型基础上通过强化学习进一步提升批判能力。
    *   RL 数据来源有两种：一是使用人工标注数据集 PRM800K；二是通过蒙特卡洛采样（Monte Carlo Sampling）自动生成标注数据，基于每一步推理的正确性估计（截断步骤后多次推演后续路径，判断首个错误步骤）。
    *   RL 阶段通过奖励机制（判断正确奖励为 1.0，否则为 0.0）激励模型提升判断准确性。
*   **关键特点:** 强调深思熟虑的批判过程，通过迭代评估、多视角验证和反思机制，确保批判不仅仅是重复推理，而是真正挑战和分析推理步骤。

## Experiment

*   **有效性:** DeepCritic-7B-SFT 模型相比基础模型 Qwen2.5-7B-Instruct 在 F1 分数上提升约 20 个百分点（34.1 → 54.1），RL 阶段进一步提升性能，DeepCritic-7B-RL-PRM800K 在 6 个测试集中的 5 个上超越所有基线模型（包括 GPT-4o），平均 F1 分数达到 67.1。
*   **优越性:** 与同等规模的 DeepSeek-R1-Distill 系列模型相比，DeepCritic 表现出更强的批判能力，尤其在复杂数学任务（如 OlympiadBench）上优势明显。
*   **测试时扩展性:** 通过多次采样进行多数投票（Maj@8），DeepCritic 的判断准确性进一步提升；作为验证器或反馈提供者，能有效改进生成模型性能，甚至实现弱模型对强模型的监督（如 7B 模型改进 72B 模型输出）。
*   **实验设置合理性:** 实验覆盖多种数据集（从简单到复杂的数学问题），对比模型包括不同规模和能力的 LLM，确保结果全面性和可信度；唯一局限是 RL 阶段因计算资源限制，输入数据步骤数量受限，可能影响更复杂任务的表现。

## Further Thoughts

DeepCritic 框架的多视角批判与元批判机制为提升模型自我纠错能力提供了新思路，可扩展至其他复杂推理领域（如代码生成）；自动化数据生成（蒙特卡洛采样）减少了对人工标注的依赖，未来可结合更多自动化评估方法进一步降低监督成本；弱强监督的潜力（小模型改进大模型）值得在不同任务上进一步探索其泛化性。