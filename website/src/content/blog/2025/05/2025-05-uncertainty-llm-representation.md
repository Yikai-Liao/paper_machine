---
title: "Pretrained LLMs Learn Multiple Types of Uncertainty"
pubDatetime: 2025-05-27T14:06:15+00:00
slug: "2025-05-uncertainty-llm-representation"
type: "arxiv"
id: "2505.21218"
score: 0.8623570514745006
author: "grok-3-latest"
authors: ["Roi Cohen", "Omri Fahn", "Gerard de Melo"]
tags: ["LLM", "Uncertainty Representation", "Linear Probe", "Pre-Training", "Post-Training"]
institution: ["HPI/University of Potsdam", "Tel Aviv University"]
description: "本文提出一种线性探针框架，揭示了大型语言模型在预训练中内化了多种不确定性表示，并分析了模型层级、规模和训练策略的影响，为提升模型可靠性和缓解幻觉问题提供了新见解。"
---

> **Summary:** 本文提出一种线性探针框架，揭示了大型语言模型在预训练中内化了多种不确定性表示，并分析了模型层级、规模和训练策略的影响，为提升模型可靠性和缓解幻觉问题提供了新见解。 

> **Keywords:** LLM, Uncertainty Representation, Linear Probe, Pre-Training, Post-Training

**Authors:** Roi Cohen, Omri Fahn, Gerard de Melo

**Institution(s):** HPI/University of Potsdam, Tel Aviv University


## Problem Background

大型语言模型（LLMs）在预训练后是否能够捕获不确定性（Uncertainty）是一个关键问题，因为它们常因‘幻觉’（Hallucinations）生成错误或误导性内容，影响准确性和可信度。
论文旨在探究 LLMs 是否在预训练中内化了不确定性表示，以及这种表示是否可以被提取用于预测生成内容的正确性，同时研究不确定性是以单一还是多种形式存在。

## Method

*   **核心思想:** 假设不确定性是 LLMs 潜在空间中的线性概念，可以通过线性探针提取，用于预测模型生成内容的正确性，而无需对模型权重进行额外训练。
*   **线性不确定性搜索（Linear Uncertainty Search）:** 在模型的每一层 Transformer 隐藏状态中，训练一个简单的线性分类器（Logistic Regression Probe）。具体步骤包括：
    *   对于给定数据集的每个问答对，输入问题到模型，获取每一层结束时的隐藏状态。
    *   对比模型预测答案与真实答案，标注正确性标签（正确为正，错误为负）。
    *   使用隐藏状态和正确性标签训练线性分类器，得到一个线性向量（称为不确定性向量），代表该层对该数据集的不确定性表示。
*   **不确定性向量作为预测器（Uncertainty Vector as Predictor）:** 在测试阶段，利用训练得到的不确定性向量，通过计算隐藏状态与向量的点积（加上偏置项），预测模型在未见数据上的生成正确性。
*   **关键特点:** 方法不依赖于模型微调，仅通过分析预训练模型的内部表示即可提取不确定性信息，同时针对不同数据集和模型层级进行独立分析，以捕捉不确定性表示的多样性和层级特性。

## Experiment

*   **有效性:** 实验表明，在多个模型（Llama、Mistral、Qwen）和 16 个问答数据集上，提取的不确定性向量在预测生成正确性方面的准确率显著高于随机基线（0.5），证明 LLMs 在预训练中内化了不确定性表示。
*   **多样性与特异性:** 模型学习了多个数据集特异性的不确定性向量，这些向量几乎正交（Cosine Similarity 接近零），但某些主题（如数学）的不确定性向量在相关数据集间有一定泛化能力。
*   **层级分析:** 中间层（通常在模型层数的中部）的不确定性向量预测准确率最高，表明不确定性信息在中间层最为集中。
*   **模型规模影响:** 模型规模对不确定性表示能力影响不大，小模型（如 Llama-3.2-1B）在某些情况下与大模型表现相当甚至更优。
*   **训练策略影响:** 指令微调（Instruction-Tuning）和 [IDK]-Tuning 显著提升不确定性捕获能力，且峰值准确率出现在更早的层，显示后训练策略对模型可靠性的重要性。
*   **合理性与局限性:** 实验覆盖多种模型和数据集，数据分割明确，但未报告误差条，可能是由于大规模实验统计误差较小；总体设计合理，支持主要结论。

## Further Thoughts

不确定性作为潜在空间中的线性概念，为模型可解释性研究开辟了新方向，或许可以通过类似线性探针方法探测其他抽象概念（如情感、意图）的表示；此外，多个数据集特异性不确定性向量的发现提示 LLMs 内部表示可能是模块化的，这对设计更可靠模型或开发实时幻觉检测工具具有潜在价值；指令微调和 [IDK]-Tuning 的提升效果也表明，针对不确定性的特定训练策略可能比单纯扩大模型规模更有效。