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
description: "本文通过引入表示分散度这一指标，揭示了语言模型嵌入空间广度与预测性能的强相关性，并展示了其在无标签性能预测、模型选择、层选择和训练优化中的实用价值。"
---

> **Summary:** 本文通过引入表示分散度这一指标，揭示了语言模型嵌入空间广度与预测性能的强相关性，并展示了其在无标签性能预测、模型选择、层选择和训练优化中的实用价值。 

> **Keywords:** LLM, Embedding Space, Representation Dispersion, Perplexity, Model Selection

**Authors:** Yanhong Li, Karen Livescu, Ming Li, Jiawei Zhou

**Institution(s):** University of Chicago, Toyota Technological Institute at Chicago, University of Maryland, Stony Brook University


## Problem Background

大型语言模型（LLMs）的嵌入空间常表现出各向异性或秩坍缩问题，隐藏状态集中在狭窄区域或低维子空间，可能限制模型的表达能力。
本文研究嵌入空间的广度（即表示分散度）与模型文本预测能力（以困惑度衡量）之间的关系，探索如何利用这种关系解决模型性能预测、选择和优化的关键问题。

## Method

*   **核心概念：表示分散度 (Representation Dispersion)**
    *   定义为隐藏向量之间的平均成对余弦距离（average pairwise cosine distance），用于量化嵌入空间的广度。
    *   在任意选定层（默认最后一层）计算，反映模型如何在嵌入空间中区分文本样本。
*   **验证方法：**
    *   在多个模型家族（如 LLaMA、Qwen）和领域（如 Wikipedia、新闻、科学摘要）上，分析分散度与困惑度（perplexity）的相关性。
    *   使用大规模文本片段（例如 100,000 个 512 令牌片段）进行统计分析，确保结果稳健。
*   **应用场景：**
    *   **无标签性能预测：** 在无标注数据上测量分散度，预测下游任务准确率，用于数据高效的模型验证。
    *   **模型选择：** 通过计算领域特定令牌（如数学中的数字、代码中的关键字）嵌入的分散度差距（dispersion gap），快速筛选性能最佳的模型变体。
    *   **层选择：** 在检索增强方法（如 kNN-LM）中，选择分散度最高的隐藏层作为数据存储键，优化检索性能。
    *   **训练优化：** 引入“推开”辅助损失（push-away loss），在训练中增加分散度，直接降低困惑度，适用于单领域和跨领域场景。
*   **关键特点：** 方法简单且无监督，仅依赖模型内部表示，无需额外标注数据，具有高实用性。

## Experiment

*   **相关性验证：** 实验表明表示分散度与困惑度呈显著负相关，即更高的分散度对应更低的困惑度，这一趋势在多个模型（如 LLaMA、Qwen）和数据集（如 WikiText-103、CNN DailyMail）上均成立。
*   **下游性能预测：** 在 ARC Challenge 和 MMLU 任务中，分散度作为无标签指标能有效预测准确率，准确率随分散度单调上升（见 Figure 8）。
*   **模型选择效果：** 提出的分散度差距指标与 MATH 和 HumanEval 任务性能高度相关（Spearman 相关系数 > 0.95），证明其作为快速筛选工具的有效性（见 Figure 9）。
*   **层选择优化：** 在 kNN-LM 中，注意力子层的分散度普遍高于前馈子层，选择高分散度层显著提升检索性能（见 Table 1）。
*   **训练改进：** 加入推开辅助损失后，单领域（WikiText）困惑度略降 1-4 点，跨领域（WikiText+Code）困惑度显著降低（见 Table 2）。
*   **实验设置合理性：** 实验覆盖多种模型、数据集和应用场景，使用大规模数据（100,000 文本片段）和多次重复试验（10 个随机种子），结果稳健，但主要限于英文数据，普适性待验证。

## Further Thoughts

表示分散度作为无监督指标的潜力令人印象深刻，它不仅能预测性能，还能指导模型选择和训练优化。未来是否可以结合其他几何指标（如内在维度）进一步描述嵌入空间特性？此外，分散度是否能用于检测过拟合或泛化问题，例如通过比较训练和测试数据上的分散度差异来诊断模型行为？