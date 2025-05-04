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
description: "本文通过控制实验揭示上下文学习在系统性泛化任务上优于微调，并提出通过上下文推理增强微调数据的方法，显著提升了微调的泛化能力。"
---

> **Summary:** 本文通过控制实验揭示上下文学习在系统性泛化任务上优于微调，并提出通过上下文推理增强微调数据的方法，显著提升了微调的泛化能力。 

> **Keywords:** LLM, In-Context Learning, Finetuning, Generalization, Data Augmentation

**Authors:** Andrew K. Lampinen, Arslan Chaudhry, Stephanie C.Y. Chan, Cody Wild, Diane Wan, Alex Ku, Jörg Bornschein, Razvan Pascanu, Murray Shanahan, James L. McClelland

**Institution(s):** Google DeepMind, Stanford University


## Problem Background

大型语言模型（LLMs）在预训练后展现出强大的上下文学习（In-Context Learning, ICL）能力，能够通过少量示例泛化到新任务，但通过微调（Finetuning）适配下游任务时，泛化能力往往受限，例如无法处理简单的关系反转或逻辑推理任务，这种泛化失败限制了模型的实际应用；相比之下，上下文学习在某些情况下表现出更强的泛化能力，作者旨在研究两种学习模式的泛化差异及其背后的归纳偏见，并探索改进微调泛化能力的方法。

## Method

* **核心思想：** 通过控制实验对比上下文学习和微调在系统性泛化任务上的表现差异，并提出一种结合上下文学习优势的增强微调方法，以提升泛化能力。
* **数据集设计：** 构建多个合成数据集（如简单反转、简单三段论、语义结构基准等），使用无意义词汇（Nonsense Words）避免与预训练知识重叠，确保实验控制性；数据集包含不同类型的泛化测试拆分，如关系反转、逻辑推理（三段论）和类别保留等。
* **学习模式对比：**
  * **上下文学习（ICL）：** 将整个训练数据集或其子集置于模型上下文窗口中，提示模型基于上下文回答测试问题，无需参数更新。
  * **微调（Finetuning）：** 使用训练数据集对模型（如 Gemini 1.5 Flash）进行参数更新，测试其在未见数据上的泛化能力。
* **增强微调方法：**
  * **数据集增强（Dataset Augmentation）：** 利用上下文学习生成额外的推理数据（如反转、重新表述、三段论推理等），分为局部增强（针对单个句子生成多种表述）和全局增强（基于整个数据集生成跨文档推理），将这些增强数据加入微调训练集。
  * **句子分割（Sentence Splitting）：** 将训练文档拆分为独立句子作为训练样本，探索独立分割（每个句子独立）和累积分割（包含前文句子作为上下文）两种方式，以打破句子间相关性，提升学习效果。
* **评估方式：** 使用多选似然评分（Multiple-Choice Likelihood Scoring）评估模型在测试集上的表现，关注不同泛化任务的准确率。

## Experiment

* **泛化能力对比：** 在数据量匹配的情况下，上下文学习（ICL）在大多数系统性泛化任务（如关系反转、三段论推理）上显著优于微调，例如在 Reversal Curse 数据集上，ICL 准确率接近满分，而微调接近零，表明 ICL 具有更灵活的归纳偏见。
* **增强微调效果：** 通过上下文推理增强微调数据后，微调的泛化能力显著提升，在语义结构基准的反转和三段论任务上，增强微调甚至超越了 ICL，显示出结合两种学习模式优势的潜力。
* **实验设置合理性：** 实验覆盖多种数据集和泛化拆分（如反转、三段论、类别保留），使用无意义词汇避免预训练知识干扰，确保控制性；同时考虑了模型规模（Gemini 1.5 Flash vs Flash-8B）和数据效率（低样本量表现），设置全面合理。
* **局限性与挑战：** ICL 在处理无意义词汇的长上下文时表现下降，可能由于缺乏语义先验；某些复杂任务（如类别保留）对两种方法都具挑战性，改进空间有限；此外，实验主要基于单一模型，未广泛验证其他模型的通用性。

## Further Thoughts

论文揭示了上下文学习和微调在归纳偏见上的差异，启发我们针对不同任务选择合适的学习模式，或设计混合策略；训练时推理扩展（Train-Time Inference Scaling）的概念提示，可以通过在训练阶段投入更多计算资源模拟推理过程，减少对大规模标注数据的依赖；此外，‘通过思考学习’和信息可访问性的思想，可以扩展到生成中间推理步骤以改进复杂任务表现，不仅适用于语言模型，也可能启发视觉或多模态模型的泛化能力提升。