---
title: "ReGraP-LLaVA: Reasoning enabled Graph-based Personalized Large Language and Vision Assistant"
pubDatetime: 2025-05-06T16:00:13+00:00
slug: "2025-05-regrap-llava-personalization"
type: "arxiv"
id: "2505.03654"
score: 0.5638548410809668
author: "grok-3-latest"
authors: ["Yifan Xiang", "Zhenxi Zhang", "Bin Li", "Yixuan Weng", "Shoujun Zhou", "Yangfan He", "Keqin Li"]
tags: ["LLM", "Multimodal Learning", "Personalization", "Knowledge Graph", "Reasoning"]
institution: ["Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences", "School of Engineering, Westlake University", "University of Minnesota – Twin Cities", "University of Toronto"]
description: "本文提出 ReGraP-LLaVA 模型，通过知识图谱和推理链数据的软硬提示方法，显著提升多模态大语言模型在个性化任务中的关系推理能力。"
---

> **Summary:** 本文提出 ReGraP-LLaVA 模型，通过知识图谱和推理链数据的软硬提示方法，显著提升多模态大语言模型在个性化任务中的关系推理能力。 

> **Keywords:** LLM, Multimodal Learning, Personalization, Knowledge Graph, Reasoning

**Authors:** Yifan Xiang, Zhenxi Zhang, Bin Li, Yixuan Weng, Shoujun Zhou, Yangfan He, Keqin Li

**Institution(s):** Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences, School of Engineering, Westlake University, University of Minnesota – Twin Cities, University of Toronto


## Problem Background

多模态大语言模型（MLLMs）在图像分析和问答任务中表现出色，但现有个性化方法存在局限：训练数据缺乏多对象关系信息，模型忽略个性化概念间的关系且无法进行推理，实验评估局限于单一概念的识别和描述任务。
因此，本研究旨在通过引入知识图谱和推理链数据，增强模型在个性化任务中的关系推理能力。

## Method

*   **数据集构建（ReGraP Dataset）**：提出了一个包含 120 个个性化知识集合的数据集，每个集合包括图像、知识图谱（KGs）和推理链问答对（CoT QA pairs）。知识图谱通过数据生成流程构建，捕捉个性化概念、属性及关系；CoT 问答对基于图谱路径生成，支持模型学习多步推理过程。
*   **训练框架（ReGraP-LLaVA）**：基于 LLaVA 模型，设计了两种图谱提示方法以对齐知识图谱与模型语义空间：
    *   **软提示（Soft Prompting）**：使用图神经网络（GNN）编码知识图谱为嵌入向量，并通过多层感知机（MLP）投影到模型的词嵌入空间，作为软提示与视觉和文本输入结合。
    *   **硬提示（Hard Prompting）**：引入新的实体和关系 token，将知识图谱转化为自然语言描述序列，作为硬提示输入，增强模型对结构化知识的理解。
*   **训练细节**：冻结 LLaVA 的视觉编码器和投影器，结合图像、CoT 问答对和图谱提示进行训练，使用标准语言损失函数优化模型，确保既学习个性化知识又保留原有对话能力。

## Experiment

*   **有效性**：在闭合式问答任务中，ReGraP-LLaVA 显著优于基线模型（如 Yo’LLaVA 和 LLaVA），尤其在需要关系推理的复杂任务中，准确率提升明显（相比最佳微调模型提升 5.3%，相比最佳提示模型提升 11.0%）。
*   **开放式问答表现**：在描述性任务和完整图像描述中，ReGraP-LLaVA 在关键点覆盖指标（Point）上表现最佳，生成的回答更全面且贴合上下文。
*   **消融研究**：硬提示方法在准确率上略优于软提示和组合方法（差异小于 0.4%），证明了图谱提示方法的鲁棒性。
*   **实验设置合理性**：ReGraP Benchmark 设计了多样化任务类型（多项选择、填空、判断、描述性问题），涵盖简单识别和复杂推理任务，评估全面；数据集规模（120 个集合）和多对象场景的引入增强了实验代表性。

## Further Thoughts

知识图谱与多模态大语言模型的结合为处理复杂关系推理任务提供了新思路，未来可以探索动态图谱更新或自适应提示生成方法；此外，CoT 数据在推理中的作用提示我们可以在其他多模态任务中引入类似结构化推理数据，以增强模型逻辑性和上下文理解能力；最后，如何在个性化与通用能力间设计更精细的权衡机制，避免过拟合到特定用户数据，是一个值得深入研究的方向。