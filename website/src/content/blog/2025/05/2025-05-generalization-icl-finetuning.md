---
title: "On the generalization of language models from in-context learning and finetuning: a controlled study"
pubDatetime: 2025-05-01T17:02:27+00:00
slug: "2025-05-generalization-icl-finetuning"
type: "arxiv"
id: "2505.00661"
score: 0.8430071071509898
author: "grok-3-latest"
authors: ["Andrew K. Lampinen", "Arslan Chaudhry", "Stephanie C.Y. Chan", "Cody Wild", "Diane Wan", "Alex Ku", "Jörg Bornschein", "Razvan Pascanu", "Murray Shanahan", "James L. McClelland"]
tags: ["LLM", "In-Context Learning", "Finetuning", "Generalization", "Data Augmentation"]
institution: ["Google DeepMind", "Stanford University"]
description: "本文通过控制实验揭示上下文学习在泛化能力上优于微调，并提出通过数据增强将上下文推理融入微调数据的方法，显著提升了语言模型的泛化性能。"
---

> **Summary:** 本文通过控制实验揭示上下文学习在泛化能力上优于微调，并提出通过数据增强将上下文推理融入微调数据的方法，显著提升了语言模型的泛化性能。 

> **Keywords:** LLM, In-Context Learning, Finetuning, Generalization, Data Augmentation

**Authors:** Andrew K. Lampinen, Arslan Chaudhry, Stephanie C.Y. Chan, Cody Wild, Diane Wan, Alex Ku, Jörg Bornschein, Razvan Pascanu, Murray Shanahan, James L. McClelland

**Institution(s):** Google DeepMind, Stanford University


## Problem Background

大型语言模型（LLMs）在预训练后展现出强大的上下文学习（In-Context Learning, ICL）能力，能够通过少量示例泛化到新任务，但通过微调（Finetuning）适配下游任务时，泛化能力往往受限，例如无法从训练数据中推导出简单的逆向关系或逻辑推理，这种泛化失败限制了模型的实际应用；相比之下，ICL 在某些情况下表现出更灵活的泛化能力，论文旨在系统性研究这两种学习模式的泛化差异，并探索改进微调泛化能力的方法。

## Method

* **数据集构建**：设计了多个合成数据集（如简单逆向关系、简单三段论、语义结构基准等），使用无意义词汇（nonsense words）避免与预训练知识重叠，确保测试纯粹性；数据集分为训练集和测试集，测试集包含需要逆向推理、逻辑推导等泛化能力的任务。
* **上下文学习（ICL）评估**：将整个训练数据集或其子集放入模型上下文窗口，测试模型基于上下文的泛化能力，无需调整模型参数，仅依赖提示（prompt）引导模型推理。
* **微调（Finetuning）评估**：使用训练数据集对模型（如 Gemini 1.5 Flash）进行微调，通过调整模型参数适配特定任务，测试其在测试集上的泛化表现，关注是否能从训练数据中学习到系统性推理能力。
* **数据增强（Dataset Augmentation）**：提出利用 ICL 的优势改进微调泛化能力，具体方法是利用上下文学习生成额外的推理数据（如逆向关系、逻辑推导），分为局部增强（句子级别的改写和逆向推理）和全局增强（基于整个数据集的文档级别推理），然后将这些增强数据加入微调训练集，提升模型对未见任务的适应性。
* **句子分割（Sentence Splitting）**：对训练文档进行句子级分割，探索通过打破句子间上下文关联性是否能提升微调效果，分为独立分割（每个句子独立作为训练样本）和累积分割（目标句子包含前文句子作为上下文），以分析上下文依赖对学习的影响。

## Experiment

* **ICL 优于微调**：在多个数据集（如逆向关系、三段论、语义结构基准）上，ICL 表现出比微调更强的泛化能力，尤其是在逆向关系和逻辑推理任务上，例如在‘逆向诅咒’数据集上，ICL 接近满分（准确率近 1.0），而微调几乎为零。
* **数据增强效果显著**：通过将 ICL 生成的推理数据加入微调数据集，微调的泛化能力显著提高，甚至在某些任务（如语义结构基准的逆向和三段论任务）上超过 ICL，表明增强策略有效弥补了微调的不足。
* **实验设置合理性**：实验设计了多种数据集，覆盖不同泛化类型（如逆向、逻辑推理、类别保留），并通过无意义词汇避免预训练知识干扰，确保控制性；同时对比了不同模型规模（如 Gemini 1.5 Flash 和 Flash-8B）和增强策略，设置全面。
* **局限性与不足**：使用无意义词汇可能干扰 ICL 的长上下文推理能力（如在逆向任务上的表现下降），且实验仅限于 Gemini 系列模型，未验证其他模型的泛化性；此外，某些任务（如类别保留）对 ICL 和增强微调仍具挑战性，改进空间有限。

## Further Thoughts

论文提出的‘训练时推理扩展’（Train-Time Inference Scaling）概念令人启发，即通过在训练阶段利用上下文学习生成额外数据来提升模型泛化能力，这不仅揭示了 ICL 和微调的互补性，还为如何高效利用计算资源提供了新思路；此外，‘通过思考学习’（Learning by Thinking）的理论视角，即通过计算使隐含信息更易访问，也为理解模型如何从数据中提取深层结构提供了新的研究方向，未来可探索是否能通过设计更智能的增强策略进一步提升泛化能力。