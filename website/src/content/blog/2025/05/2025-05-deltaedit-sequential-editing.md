---
title: "DeltaEdit: Enhancing Sequential Editing in Large Language Models by Controlling Superimposed Noise"
pubDatetime: 2025-05-12T07:11:26+00:00
slug: "2025-05-deltaedit-sequential-editing"
type: "arxiv"
id: "2505.07899"
score: 0.5493286390010227
author: "grok-3-latest"
authors: ["Ding Cao", "Yuchen Cai", "Rongxi Guo", "Xuesong He", "Guiquan Liu"]
tags: ["LLM", "Knowledge Editing", "Sequential Editing", "Noise Control", "Parameter Update"]
institution: ["University of Science and Technology of China", "State Key Laboratory of Cognitive Intelligence"]
description: "本文提出DeltaEdit方法，通过动态正交约束策略优化更新参数，显著降低顺序编辑中的叠加噪声，提升大型语言模型的编辑成功率和稳定性。"
---

> **Summary:** 本文提出DeltaEdit方法，通过动态正交约束策略优化更新参数，显著降低顺序编辑中的叠加噪声，提升大型语言模型的编辑成功率和稳定性。 

> **Keywords:** LLM, Knowledge Editing, Sequential Editing, Noise Control, Parameter Update

**Authors:** Ding Cao, Yuchen Cai, Rongxi Guo, Xuesong He, Guiquan Liu

**Institution(s):** University of Science and Technology of China, State Key Laboratory of Cognitive Intelligence


## Problem Background

大型语言模型（LLMs）在预训练后编码了大量知识，但这些知识可能随时间变得过时或不准确，需要持续更新以保持准确性和可靠性。
传统的微调方法计算成本高且可能导致灾难性遗忘，而顺序知识编辑技术作为高效替代方案，在长期连续编辑后面临编辑成功率显著下降的问题，原因是模型输出逐渐偏离预期目标，论文将其定义为‘叠加噪声（Superimposed Noise）’问题。

## Method

*   **核心思想:** 提出DeltaEdit方法，通过动态正交约束策略优化更新参数，减少顺序编辑中的叠加噪声，降低编辑间的干扰，确保模型性能稳定。
*   **更新参数分解:** 将更新参数∆分解为影响向量（Influence Vector, α）和激活向量（Activation Vector, β），前者决定更新对输出的修改能力，后者控制更新在不同输入下的触发程度。
*   **噪声成因分析:** 通过理论分析，发现叠加噪声主要由激活向量的错误激活（由输入表示引起）和影响向量之间的重叠导致，现有方法多忽视影响向量优化。
*   **动态正交约束策略:** 在训练影响向量α时，利用历史编辑信息构建列空间矩阵，通过奇异值分解（SVD）计算正交空间，将α投影到该空间以减少与历史编辑的干扰；同时引入基于滑动平均的动态阈值，判断是否应用正交约束，避免过度限制训练空间。
*   **实现细节:** 不存储历史向量以降低存储开销，动态调整约束强度以平衡编辑效率与噪声控制，激活向量β的计算沿用AlphaEdit方法，确保与现有技术的兼容性。

## Experiment

*   **有效性:** 在CounterFact数据集上，DeltaEdit在Llama3-8B模型的Efficacy_top指标上比AlphaEdit提升43.52%，在GPT2-XL上也有显著提升（幅度较小，因AlphaEdit已表现较好）；在ZsRE数据集上优势不明显，可能是因对象相似性低导致噪声影响较小。
*   **稳定性:** 在连续3000次编辑后，DeltaEdit保持高性能，尤其在噪声敏感的Llama3-8B上显著优于基线方法（Fine-Tuning, ROME, MEMIT, PRUNE, RECT, AlphaEdit），表明其噪声控制策略有效。
*   **泛化与特异性:** 在CounterFact数据集上，DeltaEdit的Generalization_top和Specificity_top指标均有提升，显示出对编辑知识的更好整合与对未编辑内容的保护。
*   **噪声控制:** 实验显示DeltaEdit显著降低叠加噪声（noise_E），尤其在Llama3-8B上效果更明显，与编辑性能提升直接相关。
*   **通用能力保留:** 通过GLUE基准测试，DeltaEdit在多个任务（如CoLA, MMLU）上保持模型原有性能，F1分数差异较小，表明编辑对通用能力的干扰最小。
*   **实验设置合理性:** 实验覆盖不同规模模型（GPT2-XL和Llama3-8B）和代表性数据集（CounterFact和ZsRE），指标设计全面（Efficacy, Generalization, Specificity），评估了编辑效果、泛化性和稳定性，设置合理。

## Further Thoughts

DeltaEdit通过正交约束减少编辑间干扰的思路可扩展至多任务学习或模型微调，解决任务间干扰问题；动态阈值设计提示自适应策略的潜力，未来可结合强化学习优化调整机制；此外，影响向量与激活向量的联合优化或探索噪声累积与模型架构的关系，可能进一步提升编辑性能。