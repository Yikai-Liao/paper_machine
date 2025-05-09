---
title: "ReGraP-LLaVA: Reasoning enabled Graph-based Personalized Large Language and Vision Assistant"
pubDatetime: 2025-05-06T16:00:13+00:00
slug: "2025-05-regrap-llava-personalization"
type: "arxiv"
id: "2505.03654"
score: 0.5638548410809668
author: "grok-3-latest"
authors: ["Yifan Xiang", "Zhenxi Zhang", "Bin Li", "Yixuan Weng", "Shoujun Zhou", "Yangfan He", "Keqin Li"]
tags: ["LLM", "Multimodal Learning", "Knowledge Graph", "Personalization", "Reasoning"]
institution: ["Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences", "Westlake University", "University of Minnesota – Twin Cities", "University of Toronto"]
description: "本文提出 ReGraP-LLaVA 模型，通过知识图谱和思维链问答数据增强个性化多模态大语言模型的关系推理能力，显著提升了上下文理解和复杂任务表现。"
---

> **Summary:** 本文提出 ReGraP-LLaVA 模型，通过知识图谱和思维链问答数据增强个性化多模态大语言模型的关系推理能力，显著提升了上下文理解和复杂任务表现。 

> **Keywords:** LLM, Multimodal Learning, Knowledge Graph, Personalization, Reasoning

**Authors:** Yifan Xiang, Zhenxi Zhang, Bin Li, Yixuan Weng, Shoujun Zhou, Yangfan He, Keqin Li

**Institution(s):** Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences, Westlake University, University of Minnesota – Twin Cities, University of Toronto


## Problem Background

多模态大语言模型（MLLMs）在图像分析和问答任务中表现出色，但现有个性化 MLLM 方法存在局限：训练数据缺乏多对象集合，无法学习对象间关系；模型忽略个性化概念之间的关系，缺乏推理能力；实验评估局限于单一概念的识别和描述任务，未能考察复杂的上下文理解能力。
因此，研究出发点是构建一个能够学习个性化知识并进行关系推理的模型，超越简单概念识别，达到更接近人类理解的上下文推理水平。

## Method

* **数据集构建（ReGraP Dataset）**：提出一个包含 120 个个性化知识集合的数据集，每个集合包括图像、知识图谱（Knowledge Graphs, KGs）和思维链问答（Chain-of-Thought QA, CoT QA）对。通过数据生成流程，从图像和文本中提取个性化概念、属性和关系，构建知识图谱，并基于图谱生成推理路径和问答对，为模型提供结构化的知识表示。
* **训练框架（Soft and Hard Graph Prompting）**：
  * **Soft Prompting**：使用图神经网络（GNN）编码知识图谱为嵌入向量，并通过多层感知机（MLP）投影到模型的语义空间，作为软提示输入。这种方法将图谱的结构化信息转化为连续向量，便于模型处理。
  * **Hard Prompting**：引入新的实体和关系 token，将知识图谱转化为自然语言描述序列，作为硬提示输入。这种方法更直观，允许模型直接学习图谱中的结构化信息。
* **模型训练**：基于 LLaVA 架构，冻结视觉编码器和投影器，重点训练语言模型部分，利用 CoT QA 数据增强推理能力，确保模型在学习个性化知识的同时保持原有对话能力。

## Experiment

* **有效性**：在闭合式问答任务（多项选择、填空、判断、描述性问题）中，ReGraP-LLaVA 显著优于基线模型（如 Yo’LLaVA 和 LLaVA），尤其在需要关系推理的复杂任务上，准确率提升明显（相比最佳微调模型 LLaVA (CoT) 提升 5.3%，相比最佳提示模型 LLaVA-13B 提升 11.0%）。
* **开放式问答**：在描述性任务和完整图像描述任务中，ReGraP-LLaVA 在关键点覆盖率（Point 指标）上表现最佳，表明其生成的回答更全面且贴合上下文。
* **消融研究**：硬提示方法在闭合式问答任务中略优于软提示和组合方法，但三种方法的性能差异较小（不到 0.4%），表明两种提示方法均有效。
* **实验设置**：ReGraP Benchmark 设计全面，涵盖多种任务类型（多项选择、填空、判断、描述性问题）和难度级别（简单任务和需要推理的复杂任务），同时包括开放式和闭合式设置，合理评估了模型在个性化知识获取和关系推理上的能力。

## Further Thoughts

知识图谱与多模态模型的结合为提升复杂关系理解能力提供了新思路，未来可扩展到医疗诊断或教育领域，利用领域知识图谱增强推理能力；软硬提示方法的灵活性启发了对提示设计的进一步探索，如动态提示调整；此外，CoT 数据在推理中的作用表明，生成更高质量的推理路径或结合强化学习优化推理过程是值得研究的方向。