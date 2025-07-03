---
title: "On the Predictive Power of Representation Dispersion in Language Models"
pubDatetime: 2025-06-30T17:53:50+00:00
slug: "2025-06-representation-dispersion"
type: "arxiv"
id: "2506.24106"
score: 0.6995288572597577
author: "grok-3-latest"
authors: ["Yanhong Li", "Karen Livescu", "Ming Li", "Jiawei Zhou"]
tags: ["LLM", "Embedding Space", "Representation Dispersion", "Perplexity", "Model Selection"]
institution: ["University of Chicago", "Toyota Technological Institute at Chicago", "University of Maryland", "Stony Brook University"]
description: "本文揭示了语言模型中表示分散度与预测性能的强负相关性，并通过无监督性能预测、模型选择、层选择和训练优化等多种应用，展示了嵌入空间几何特性的实用价值。"
---

> **Summary:** 本文揭示了语言模型中表示分散度与预测性能的强负相关性，并通过无监督性能预测、模型选择、层选择和训练优化等多种应用，展示了嵌入空间几何特性的实用价值。 

> **Keywords:** LLM, Embedding Space, Representation Dispersion, Perplexity, Model Selection

**Authors:** Yanhong Li, Karen Livescu, Ming Li, Jiawei Zhou

**Institution(s):** University of Chicago, Toyota Technological Institute at Chicago, University of Maryland, Stony Brook University


## Problem Background

大型语言模型（LLMs）的嵌入空间几何特性（如各向异性和秩坍缩）可能限制其表达能力，但其与自回归文本生成性能的具体关系尚不明确。
本文旨在探索嵌入空间的‘广度’（即表示分散度，representation dispersion）与模型预测能力（如困惑度，perplexity）之间的联系，并研究如何利用这一特性解决实际问题，如模型选择和性能优化。

## Method

*   **核心思想:** 通过量化嵌入空间的表示分散度（representation dispersion），揭示其与模型预测性能的关联，并将其应用于无监督性能预测和模型优化。
*   **分散度定义与测量:** 表示分散度定义为隐藏向量之间的平均成对余弦距离（average pairwise cosine distance），通常在模型的最终层计算，但也适用于中间层。通过对大量文本片段提取隐藏表示，计算其成对距离来评估嵌入空间的‘广度’。
*   **验证关联性:** 在多个模型家族（如 LLaMA、Qwen、Mistral）和领域（如 Wikipedia、新闻、科学摘要）上，分析分散度与困惑度之间的关系，验证其负相关性。
*   **实际应用:** 
    *   **无标签性能预测:** 在无标签文本上测量分散度，预测下游任务准确率，用于识别模型在特定数据上的潜在表现。
    *   **模型选择:** 通过计算领域特定token（如数学中的数字、代码中的关键字）嵌入的分散度差距（dispersion gap），快速选择性能最佳的模型变体，无需昂贵的全面评估。
    *   **层选择:** 在检索增强方法（如 kNN-LM）中，通过识别具有最高分散度的层作为数据存储键（datastore key），提升检索效果。
    *   **训练优化:** 在训练中引入辅助‘推开’损失（push-away loss），鼓励隐藏表示更分散，直接降低困惑度，尤其在单领域和跨领域场景中有效。
*   **关键点:** 方法简单且计算高效，许多应用无需标签数据或额外前向计算（如模型选择中的嵌入矩阵分析），具有很强的实用性。

## Experiment

*   **有效性:** 实验表明，表示分散度与困惑度呈显著负相关（即更高分散度对应更低困惑度），这一趋势在多个模型（如 LLaMA、Qwen）和数据集（如 WikiText-103、CNN DailyMail）上均成立（见 Figure 3、4）。
*   **应用表现:** 
    *   **无标签预测:** 在 ARC Challenge 和 MMLU 数据集上，分散度与准确率呈单调正相关，可有效预测下游性能（Figure 8）。
    *   **模型选择:** 在 MATH 和 HumanEval 任务中，分散度差距（dispersion gap）与任务准确率高度相关（Spearman 相关系数 > 0.95），成功排名模型变体（Figure 9）。
    *   **层选择:** 在 kNN-LM 中，注意力子层（attention sub-layer）的分散度始终高于前馈子层（feed-forward sub-layer），选择高分散度层作为键显著提升性能（Table 1）。
    *   **训练优化:** 引入‘推开’损失后，单领域（WikiText）困惑度略有下降（1-4 点），跨领域（WikiText+Code）场景中效果更显著（Table 2）。
*   **实验设置合理性:** 数据集和模型选择覆盖广泛，实验设计（如均匀困惑度采样）确保了结果的代表性，但主要限于英文文本和标准基准，跨语言和高度领域特定数据的适用性待验证。
*   **局限性:** 实验聚焦于最终层表示，可能忽略中间层信息；因果关系未完全确立，某些架构或目标可能影响相关性。

## Further Thoughts

表示分散度作为无监督性能指标的思路令人启发，或许可以扩展到多模态模型中，分析文本与图像嵌入空间的几何交互特性，探索是否能通过几何约束提升跨模态任务性能。此外，‘推开’损失在跨领域训练中的成功应用提示我们，可以在多任务学习中设计类似几何目标，增强任务间表示的区分性，从而提升模型的专项能力。