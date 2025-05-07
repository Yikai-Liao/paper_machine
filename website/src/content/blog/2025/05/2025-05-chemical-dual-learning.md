---
title: "Enhancing Chemical Reaction and Retrosynthesis Prediction with Large Language Model and Dual-task Learning"
pubDatetime: 2025-05-05T13:31:36+00:00
slug: "2025-05-chemical-dual-learning"
type: "arxiv"
id: "2505.02639"
score: 0.512276469115314
author: "grok-3-latest"
authors: ["Xuan Lin", "Qingrui Liu", "Hongxin Xiang", "Daojian Zeng", "Xiangxiang Zeng"]
tags: ["LLM", "Chemical Synthesis", "Dual Task Learning", "Molecular Representation", "Instruction Tuning"]
institution: ["Xiangtan University", "Hunan University", "Hunan Normal University"]
description: "本文提出 ChemDual 框架，通过构建 4.4M 分子指令数据集、设计多尺度分词器和双任务学习策略，显著提升了基于 LLaMA 的化学反应与逆合成预测性能，并在多个数据集上超越现有方法。"
---

> **Summary:** 本文提出 ChemDual 框架，通过构建 4.4M 分子指令数据集、设计多尺度分词器和双任务学习策略，显著提升了基于 LLaMA 的化学反应与逆合成预测性能，并在多个数据集上超越现有方法。 

> **Keywords:** LLM, Chemical Synthesis, Dual Task Learning, Molecular Representation, Instruction Tuning

**Authors:** Xuan Lin, Qingrui Liu, Hongxin Xiang, Daojian Zeng, Xiangxiang Zeng

**Institution(s):** Xiangtan University, Hunan University, Hunan Normal University


## Problem Background

化学反应预测和逆合成预测是药物发现和合成路线设计中的核心任务，但传统方法依赖专家知识，耗时且效率低。
大型语言模型（LLMs）虽有潜力，却面临两大挑战：一是缺乏大规模化学合成相关指令数据集，二是现有微调策略忽视了反应预测与逆合成预测之间的相关性，导致预测精度受限。

## Method

*   **核心框架：ChemDual**：基于 LLaMA 模型的增强框架，旨在通过创新的数据构建和训练策略提升化学合成预测性能。
*   **数据集构建**：从 ChEMBL-34 数据库提取 20M 个分子 SMILES 序列，经过预处理（如去重、过滤无效分子）后，采用 BRICS 算法生成 4.4M 个分子-片段对数据集，模拟化学合成的重组与片段化过程，以低成本获取大规模指令数据。
*   **多尺度分词器**：扩展 LLaMA 原有分词器，新增三类标记：虚拟原子（如 [1*]）、功能团（如苯环、卤素原子）和特殊标记（如 BOF、EOF），以捕捉分子结构在不同尺度（原子、功能团、片段）上的信息，提升模型对化学数据的理解能力。
*   **双任务学习策略**：将反应预测和逆合成预测视为互逆过程，设计正向任务（分子到反应物/片段）和反向任务（反应物/片段到分子），通过联合优化增强模型对化学合成过程的理解；在预训练阶段基于分子-片段对进行片段化与重组任务学习，在指令微调阶段针对反应与逆合成任务进一步优化，使用交叉熵损失函数衡量预测分布与真实分布的差异。
*   **训练流程**：采用预训练与指令微调两阶段策略，预训练捕捉分子结构的内在关系，微调针对具体任务提升性能，确保模型在化学合成任务上的泛化能力。

## Experiment

*   **有效性**：在 Mol-Instruction 数据集上，ChemDual 在反应预测任务中取得 EXACT 分数 0.869，在逆合成预测中取得 0.670，显著优于基线模型（如 BioT5+ 的 0.864 和 0.642）；在 USPTO-50K 数据集上，ChemDual 的 Top-1 准确率为 46.25%，Top-10 为 77.42%，超越多个基线模型，结合 Retroformer 模块后进一步提升至 Top-1 49.95%。
*   **优越性**：相比通用 LLMs（如 LLaMA）和单任务学习方法，ChemDual 在多个指标（如 BLEU、LEVENSHTEIN、分子指纹相似性）上均表现出色，表明双任务学习和大规模指令数据集的引入显著提升了预测精度和化学相关性。
*   **实验设置合理性**：实验覆盖 Mol-Instruction、USPTO-50K 和 ChemLLMBench 多个数据集，采用 EXACT、BLEU、Top-k 准确率等多样化评价指标；消融实验验证了双任务学习、预训练和指令数据集的重要性，例如双任务学习使 LLaMA 的 EXACT 分数提升 6.3%。
*   **局限性与开销**：ChemDual 在 VALIDITY 指标上略低于部分基线（如 Mol-Instruction 的 1.000），可能因采用 SMILES 格式而非 SELFIES 导致；训练和推理计算成本较高，预训练和微调共耗费 212 GPU 小时。

## Further Thoughts

双任务学习策略是一个亮点，通过联合优化互逆任务（如反应与逆合成）增强模型对化学结构的理解，这种思路可推广至其他领域，如生物信息学中的蛋白质折叠与展开预测；此外，利用算法（如 BRICS）生成大规模指令数据集的方法启发我们可以通过模拟数据弥补实验数据不足，降低数据获取成本，未来可探索更多自动化数据生成技术以支持领域特定任务。