---
title: "Pretrained LLMs Learn Multiple Types of Uncertainty"
pubDatetime: 2025-05-27T14:06:15+00:00
slug: "2025-05-uncertainty-llm-probing"
type: "arxiv"
id: "2505.21218"
score: 0.8623570514745006
author: "grok-3-latest"
authors: ["Roi Cohen", "Omri Fahn", "Gerard de Melo"]
tags: ["LLM", "Uncertainty Representation", "Linear Probe", "Pre-Training", "Post-Training"]
institution: ["HPI / University of Potsdam", "Tel Aviv University"]
description: "本文提出了一种探测大型语言模型内部不确定性表示的框架，证明不确定性是可学习且多样的线性概念，为理解和缓解幻觉现象提供了新视角。"
---

> **Summary:** 本文提出了一种探测大型语言模型内部不确定性表示的框架，证明不确定性是可学习且多样的线性概念，为理解和缓解幻觉现象提供了新视角。 

> **Keywords:** LLM, Uncertainty Representation, Linear Probe, Pre-Training, Post-Training

**Authors:** Roi Cohen, Omri Fahn, Gerard de Melo

**Institution(s):** HPI / University of Potsdam, Tel Aviv University


## Problem Background

大型语言模型（LLMs）在生成文本时常出现‘幻觉’（hallucinations），即生成不准确或误导性的内容，严重影响其可靠性和可信度。
尽管 LLMs 通过预训练吸收了大量知识，但它们通常不擅长表达不确定性（Uncertainty），也未被明确训练去捕捉这一特性。
本文的出发点是探索 LLMs 是否在预训练阶段自然学习了不确定性，以及是否可以通过提取这些不确定性表示来预测生成内容的正确性，从而减少幻觉现象。

## Method

*   **核心思想：** 在不调整模型权重的情况下，通过探测 LLMs 潜在空间中的线性向量（Linear Vectors）来提取不确定性表示，并利用这些向量预测模型生成内容的正确性。
*   **具体实现：**
    *   **线性不确定性搜索（Linear Uncertainty Search）：** 在模型的每一层隐藏状态（Hidden State）上，训练一个简单的线性分类器（Logistic Regression Probe）。输入是模型在某层结束时的隐藏状态向量，目标是预测模型生成下一个 token 的正确性（通过与真实答案对比生成标签）。这一过程假设不确定性是潜在空间中的线性概念，可以通过线性分类器分离。
    *   **不确定性向量作为预测器（Uncertainty Vector as Predictor）：** 将训练得到的线性向量应用于未见数据，测试其预测模型生成正确性的能力。具体而言，通过计算隐藏状态与不确定性向量的内积（加上偏置项），判断生成的 token 是否正确。
    *   **技术细节：** 分类器的训练和测试基于多个问答数据集，针对每个模型层和数据集分别搜索不确定性向量，形成数据集特定的线性方向。
*   **关键优势：** 方法简单且计算成本低，无需对模型进行额外训练或微调，保持了模型的原始性能，同时提供了对不确定性表示的可解释性。

## Experiment

*   **有效性：** 实验表明，在多个模型（Llama、Mistral、Qwen）和 16 个问答数据集上，提取的线性不确定性向量在预测生成正确性方面的准确率显著高于随机基线（0.5），例如 Llama-3.1-8B-Instruct 在 GSM8K 数据集上的准确率为 0.737，证明 LLMs 在预训练中确实内化了不确定性。
*   **多样性发现：** LLMs 并非学习单一不确定性表示，而是针对不同数据集学习了多个近乎正交的线性向量，表明不确定性是任务或数据特定的。
*   **层级分析：** 中间层（通常在模型层数的中部）在提取不确定性向量时表现最佳，预测准确率最高，晚期层则显示出模型自信度下降。
*   **模型规模影响：** 模型规模对不确定性表示的影响不大，小模型（如 Llama-3.2-1B）在某些情况下甚至优于大模型，表明规模并非关键因素。
*   **训练策略提升：** 指令微调（Instruction-Tuning）和 [IDK]-Tuning 显著提升了不确定性捕捉能力，且最佳表示出现在更早的层，例如 Llama-3.1-8B-Instruct 相较基础模型有明显改进。
*   **实验设置合理性：** 实验覆盖多种模型（参数规模从 1B 到 14B）、数据集（涵盖常识、数学、代码等多领域）和训练策略，数据分割明确（训练集和测试集分离），结果具有较高可信度。
*   **局限性：** 未报告误差条或统计显著性测试，可能影响结果严谨性；局限于线性探针，未探索非线性方法对不确定性捕捉的潜力。

## Further Thoughts

论文揭示 LLMs 学习了多种类型的不确定性表示，这启发我们思考是否可以设计任务特定的不确定性探测器，以提高模型在特定领域的可靠性，例如针对数学任务的不确定性向量是否能改进数学推理输出？此外，指令微调和 [IDK]-Tuning 对不确定性捕捉的提升表明，未来可以通过引入不确定性相关的损失函数或特殊 token，进一步增强模型的不确定性表达能力，从而减少幻觉现象。