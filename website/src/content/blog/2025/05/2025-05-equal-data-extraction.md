---
title: "Not All Documents Are What You Need for Extracting Instruction Tuning Data"
pubDatetime: 2025-05-18T06:10:08+00:00
slug: "2025-05-equal-data-extraction"
type: "arxiv"
id: "2505.12250"
score: 0.6558634406480692
author: "grok-3-latest"
authors: ["Chi Zhang", "Huaping Zhong", "Hongtao Li", "Chengliang Chai", "Jiawei Hong", "Yuhao Deng", "Jiacheng Wang", "Tian Tan", "Yizhou Yan", "Jiantao Qiu", "Ye Yuan", "Guoren Wang", "Conghui He", "Lei Cao"]
tags: ["LLM", "Instruction Tuning", "Data Selection", "Contrastive Learning", "Distribution Alignment"]
institution: ["Beijing Institute of Technology", "Sensetime Research", "University of Arizona", "Meta", "Shanghai Artificial Intelligence Laboratory"]
description: "本文提出EQUAL框架，通过对比学习、MAB策略和最优传输分数迭代选择和提取高质量指令微调数据，显著降低计算成本并提升大型语言模型在下游任务上的性能。"
---

> **Summary:** 本文提出EQUAL框架，通过对比学习、MAB策略和最优传输分数迭代选择和提取高质量指令微调数据，显著降低计算成本并提升大型语言模型在下游任务上的性能。 

> **Keywords:** LLM, Instruction Tuning, Data Selection, Contrastive Learning, Distribution Alignment

**Authors:** Chi Zhang, Huaping Zhong, Hongtao Li, Chengliang Chai, Jiawei Hong, Yuhao Deng, Jiacheng Wang, Tian Tan, Yizhou Yan, Jiantao Qiu, Ye Yuan, Guoren Wang, Conghui He, Lei Cao

**Institution(s):** Beijing Institute of Technology, Sensetime Research, University of Arizona, Meta, Shanghai Artificial Intelligence Laboratory


## Problem Background

大型语言模型（LLM）的指令微调（Instruction Tuning）依赖于高质量的训练数据，但公开数据集有限，且利用LLM合成数据的方法往往缺乏多样性，生成的指令数据与下游任务分布不一致。
作者提出从网络语料库中提取指令数据以获取丰富知识，但直接提取面临两大挑战：(1) 使用LLM从大量文档中提取问答对（QA Pairs）计算成本极高；(2) 提取的QA对可能与下游任务无关，甚至损害模型性能。
因此，关键问题是如何以低成本从大规模文档中提取高质量、与下游任务高度相关的指令数据。

## Method

*   **核心思想:** 提出EQUAL框架，通过迭代交替进行文档选择和QA对提取，优化指令微调数据的质量和效率，避免直接处理所有文档。
*   **具体步骤:**
    *   **特征对齐与文档聚类:** 首先通过对比学习（Contrastive Learning）对文档和QA对的特征空间进行对齐，解决文档与QA对特征分布不一致的问题。随机采样少量文档提取QA对，训练嵌入模型（如BAAI/bge-en-v1.5），使相似QA对对应的文档在嵌入空间中更接近，然后基于此对所有文档进行聚类（如k-means）。
    *   **迭代选择与提取:** 采用多臂老虎机（Multi-Armed Bandit, MAB）策略，将每个文档簇视为一个‘臂’，通过文档采样分数（DS Score）平衡探索（数据多样性）和利用（数据质量）。DS Score结合最优传输（Optimal Transport, OT）分数（衡量簇内QA对分布与目标分布的相似性）和采样频率（鼓励探索未充分采样的簇）。
    *   **动态更新:** 在每次迭代中，选择DS Score最高的簇，采样部分文档，使用高性能LLM（如Qwen2.5-72B）提取QA对，并根据提取结果更新簇的OT分数，逐步提高分布估计的准确性。
*   **关键优势:** 不需提取所有文档的QA对，通过迭代采样和动态更新聚焦高质量簇，显著降低计算成本，同时通过OT分数确保数据与下游任务分布对齐。

## Experiment

*   **有效性:** 实验在AutoMathText和StackOverflow数据集上进行，针对数学（GSM8K, MATH）和编程（HUMANEVAL, MBPP）领域的下游任务评估。EQUAL在LLaMA-3.1-8B和Mistral-7B模型上全面优于基线方法（如Random, Mammoth, Rewriting等），例如在LLaMA-3.1-8B全微调设置下，EQUAL在GSM8K准确率提升4.09%，MATH提升2.64%。
*   **效率:** 计算成本（以FLOPs衡量）比基线方法低5-10倍，仅提取5%文档的QA对即可接近或超过提取全部文档的性能，显示出显著的效率优势。
*   **实验设置合理性:** 实验覆盖两种模型、两种微调方式（全微调和LoRA），并在多个下游任务上验证了方法的普适性。消融实验进一步确认了对比学习、MAB和OT分数的必要性，缺少任一组件均导致性能下降。
*   **局限性分析:** 实验未深入探讨不同领域或数据集规模对方法的影响，聚类数量和算法选择可能对结果有潜在影响，但论文通过消融研究部分缓解了这一问题。

## Further Thoughts

论文中使用最优传输（OT）分数来衡量数据分布相似性，提供了一种全局视角评估数据质量的思路，启发我思考是否可以结合其他分布度量（如KL散度）或多维度指标（如困惑度和影响函数）形成更全面的质量评估体系。此外，EQUAL的迭代数据选择框架具有通用性，可推广至其他领域（如图像或多模态数据选择），或者通过引入强化学习优化MAB策略进一步提升采样效率。