---
title: "AdaSTaR: Adaptive Data Sampling for Training Self-Taught Reasoners"
pubDatetime: 2025-05-22T07:24:11+00:00
slug: "2025-05-adaptive-star-sampling"
type: "arxiv"
id: "2505.16322"
score: 0.5891500805497442
author: "grok-3-latest"
authors: ["Woosung Koh", "Wonbeen Oh", "Jaein Jang", "MinHyung Lee", "Hyeongjin Kim", "Ah Yeon Kim", "Joonkee Kim", "Junghyun Lee", "Taehyeon Kim", "Se-Young Yun"]
tags: ["LLM", "Reasoning", "Sampling", "Curriculum Learning", "Self-Training"]
institution: ["Yonsei University", "LG AI Research", "KAIST AI"]
description: "AdaSTaR 通过自适应数据采样机制，显著提升自教式推理者的性能和效率，在六个推理基准上均取得最高准确率，同时平均减少 58.6% 的训练计算量。"
---

> **Summary:** AdaSTaR 通过自适应数据采样机制，显著提升自教式推理者的性能和效率，在六个推理基准上均取得最高准确率，同时平均减少 58.6% 的训练计算量。 

> **Keywords:** LLM, Reasoning, Sampling, Curriculum Learning, Self-Training

**Authors:** Woosung Koh, Wonbeen Oh, Jaein Jang, MinHyung Lee, Hyeongjin Kim, Ah Yeon Kim, Joonkee Kim, Junghyun Lee, Taehyeon Kim, Se-Young Yun

**Institution(s):** Yonsei University, LG AI Research, KAIST AI


## Problem Background

自教式推理者（Self-Taught Reasoners, STaR）是一种大型语言模型（LLMs）自改进推理能力的方法，通过生成推理步骤（Chains-of-Thought, CoT）并验证答案正确性进行微调。
然而，传统 STaR 采用随机数据采样，导致训练数据不平衡：模型在已解决的简单样本上过度训练，而在挑战性样本上训练不足，造成计算资源浪费和收敛速度慢；此外，仅依赖结果验证可能导致学习到错误推理过程（false positives），影响推理质量。
论文旨在解决如何通过平衡多样性学习和数据质量，实现高效且有效的自改进。

## Method

*   **核心思想:** 提出 AdaSTaR（Adaptive STaR），通过自适应数据采样解决传统 STaR 的训练不平衡问题，提升性能和效率，同时避免额外计算开销。
*   **具体实现:** 包含两个主要机制：
    *   **Adaptive Sampling for Diversity (AdaD):** 跟踪每个样本的最后采样迭代（last sampled iteration）和胜率统计（win statistic，基于模型在该样本上的正确率估计难度），使用分层最小堆（Hierarchical Min Heap）优先选择未被充分训练的样本和较难样本，以促进训练多样性，避免过度训练简单样本。
    *   **Adaptive Sampling for Curriculum (AdaC):** 根据当前训练准确率（作为模型强度的代理），动态调整采样难度；当模型较弱时，倾向于采样较易样本，随着模型强度提升，逐渐增加难样本比例，避免早期过度挑战导致推理质量下降。
*   **优化细节:** 采样时仅生成满足批次大小（batch size）所需的数据，避免传统 STaR 中过度采样后过滤的浪费；统计数据（如胜率和准确率）是 STaR 循环中已有的副产品，不引入额外计算开销。
*   **关键优势:** 在不改变模型架构或训练流程的前提下，通过采样策略优化训练数据分布，实现性能提升和计算效率改进。

## Experiment

*   **有效性:** AdaSTaR 在所有六个推理基准数据集（ARC-C, CQA, CLadder 1.5, ANLI, GSM8K, SVAMP）上均取得最高测试准确率（6/6），例如在 ARC-C 上从 STaR-Acc 的 73.2% 提升至 73.8%，在 CQA 上从 74.6% 提升至 78.0%。
*   **效率:** 相比准确率最高的基线，AdaSTaR 平均减少了 58.6% 的训练计算量（PFLOPs），减少幅度从最低 19.2% 到最高 93.7%；学习曲线显示其在相同计算预算下更快达到高准确率。
*   **泛化性:** 效果在不同预训练模型（Llama 3.2 3B, Qwen 2.5 3B）和更大规模模型（Gemma 7B）上保持一致，表明方法具有普适性。
*   **消融研究:** 去掉 AdaC（仅用 AdaD）会导致准确率下降，表明仅追求多样性会增加错误推理（false positives）；去掉胜率统计或反转优先级也会降低性能，验证了两个机制的协同作用。
*   **实验设置合理性:** 实验覆盖多种推理任务（科学、常识、因果、数学等）和模型规模，数据采样不平衡问题通过标准差（SD）量化，基线对比全面（包括标准 STaR 及其变体、SFT 等），验证了方法的改进效果。

## Further Thoughts

AdaSTaR 的自适应采样机制启发了我思考是否可以将基于模型强度的动态难度调整应用到其他自监督或半监督学习场景，例如在强化学习中根据智能体能力动态调整任务难度；此外，胜率统计作为难度代理是否可以结合其他指标（如生成 CoT 的长度或语义复杂性）进一步优化？另一个方向是是否可以通过多模型协作（如教师-学生模型）更精确地估计样本难度，而不仅仅依赖单一模型的表现？