---
title: "Enhancing Chemical Reaction and Retrosynthesis Prediction with Large Language Model and Dual-task Learning"
pubDatetime: 2025-05-05T13:31:36+00:00
slug: "2025-05-chemdual-synthesis-prediction"
type: "arxiv"
id: "2505.02639"
score: 0.5376670769099559
author: "grok-3-latest"
authors: ["Xuan Lin", "Qingrui Liu", "Hongxin Xiang", "Daojian Zeng", "Xiangxiang Zeng"]
tags: ["LLM", "Chemical Synthesis", "Dual-Task Learning", "Instruction Dataset", "Molecular Representation"]
institution: ["Xiangtan University", "Hunan University", "Hunan Normal University"]
description: "本文提出 ChemDual 框架，通过构建 440 万分子指令数据集、设计多尺度分词器和双任务学习策略，显著提升了化学反应和逆合成预测的性能，并在药物设计中展现出强大潜力。"
---

> **Summary:** 本文提出 ChemDual 框架，通过构建 440 万分子指令数据集、设计多尺度分词器和双任务学习策略，显著提升了化学反应和逆合成预测的性能，并在药物设计中展现出强大潜力。 

> **Keywords:** LLM, Chemical Synthesis, Dual-Task Learning, Instruction Dataset, Molecular Representation

**Authors:** Xuan Lin, Qingrui Liu, Hongxin Xiang, Daojian Zeng, Xiangxiang Zeng

**Institution(s):** Xiangtan University, Hunan University, Hunan Normal University


## Problem Background

化学反应预测和逆合成预测是药物发现和合成路线设计中的核心任务，但传统方法依赖专家知识，耗时且资源受限。
大型语言模型（LLMs）在化学领域的应用面临两大挑战：一是缺乏大规模化学合成相关指令数据集，因实验数据获取成本高且规模有限；二是现有微调策略忽视了反应预测与逆合成预测之间的互逆相关性，限制了模型对化学合成过程的深入理解。

## Method

*   **核心框架:** 提出 ChemDual，一种基于 LLaMA 的增强型大型语言模型，专门针对化学合成任务设计，通过多尺度分词器和双任务学习策略提升预测能力。
*   **数据集构建:** 从 ChEMBL-34 数据库中收集 20M 分子 SMILES 序列，利用 BRICS 算法生成分子片段，构建包含 440 万分子及其片段的大型指令数据集，模拟分子分解与重组过程，以低成本生成高质量数据。
*   **多尺度分词器:** 扩展 LLaMA 原有分词器，新增三类标记：虚拟原子（如 [1*] 到 [16*]）、180 个常见功能团（如苯环、卤素原子）以及分子和片段的特殊标记（如 <BOM>、<EOF>），以捕捉分子结构在不同尺度上的信息。
*   **双任务学习:** 设计前向任务（分子到片段、产物到反应物）和后向任务（片段到分子、反应物到产物）对，在预训练和指令微调阶段联合优化，利用任务间的互逆性增强模型对化学合成过程的理解，采用交叉熵损失函数优化概率分布。
*   **训练流程:** 预训练阶段在分子-片段双任务上进行，微调阶段针对反应-逆合成双任务优化，确保模型既掌握通用化学知识，又适应特定任务需求。

## Experiment

*   **有效性:** 在 Mol-Instruction 数据集上，ChemDual 在反应预测任务中取得 EXACT 分数 0.869 和 LEVENSHTEIN 距离 2.099，在逆合成预测中取得 EXACT 分数 0.670，均显著优于 BioT5+ 和 Mol-Instruction 等基准模型；在 USPTO-50K 数据集上，Top-1 准确率达 49.95%（结合 Retroformer 模块），相比 Retroformer 提升 2.06%。
*   **全面性:** 实验覆盖多个数据集（Mol-Instruction、USPTO-50K、ChemLLMBench），评价指标包括 EXACT、BLEU、分子指纹相似度（RDK、MACCS、MORGAN）等，对比模型涵盖通用 LLM 和领域特定模型，设置合理且全面。
*   **消融验证:** 消融实验表明，双任务学习、预训练和指令数据集均对性能提升至关重要，去除任一组件均导致性能下降，验证了方法的有效性。
*   **应用潜力:** 分子对接分析显示，ChemDual 生成的化合物与目标蛋白（如 MAP2）的结合亲和力较高（-6.3 至 -8.4 kcal/mol），展现了在药物设计中的应用价值。

## Further Thoughts

论文通过 BRICS 算法生成大规模指令数据集的思路启发了我，未来可以探索基于分子图或其他化学规则生成更多样化的合成数据；双任务学习策略可扩展至其他化学任务（如分子性质预测）或跨领域任务（如化学与生物学联合学习）；多尺度分词器的设计为处理复杂结构数据提供了新思路，可尝试应用于蛋白质序列或材料科学等领域。