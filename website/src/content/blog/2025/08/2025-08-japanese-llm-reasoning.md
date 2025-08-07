---
title: "Enhancing Japanese Large Language Models with Reasoning Vectors"
pubDatetime: 2025-08-04T21:31:20+00:00
slug: "2025-08-japanese-llm-reasoning"
type: "arxiv"
id: "2508.02913"
score: 0.759925135717495
author: "grok-3-latest"
authors: ["Carolina Minami Oguchi", "Leo Wei", "Koyo Kobayashi", "Hsin-Tai Wu", "Dipak Ghosal"]
tags: ["LLM", "Reasoning", "Post-Training", "Task Vector", "Cross-Lingual Transfer"]
institution: ["University of California, Davis", "Santa Clara University", "NTT DOCOMO, INC", "DOCOMO Innovations, Inc"]
description: "本文提出通过从主流语言模型中提取推理向量并注入日语模型，显著提升其推理能力，为资源受限语言模型改进提供了一种简单有效的路径。"
---

> **Summary:** 本文提出通过从主流语言模型中提取推理向量并注入日语模型，显著提升其推理能力，为资源受限语言模型改进提供了一种简单有效的路径。 

> **Keywords:** LLM, Reasoning, Post-Training, Task Vector, Cross-Lingual Transfer

**Authors:** Carolina Minami Oguchi, Leo Wei, Koyo Kobayashi, Hsin-Tai Wu, Dipak Ghosal

**Institution(s):** University of California, Davis, Santa Clara University, NTT DOCOMO, INC, DOCOMO Innovations, Inc


## Problem Background

日语大型语言模型（Japanese LLMs）由于缺乏大规模公开数据集、专家标注资源以及强大的本土模型用于数据质量过滤，难以通过传统的后训练方法（如监督微调和强化学习）显著提升性能；此外，依赖英语翻译数据可能丢失语言和文化细微差别，影响模型表现。本文旨在解决资源受限情况下如何有效增强日语LLM推理能力的问题。

## Method

* **核心思想：** 从主流（英语）大型语言模型中提取推理向量（Reasoning Vector），并将其注入到日语模型中，以增强其推理能力，而无需额外训练或标注数据。
* **具体实现：** 
  - 首先，计算预训练模型（pre-trained model）和后训练模型（post-trained model）之间的权重差异，得到推理向量，表示后训练带来的推理能力提升方向。
  - 然后，将该推理向量以可调的标量权重（scalar weight, w）添加到目标日语模型的权重中，形成增强模型。
  - 该过程不涉及对目标模型的进一步训练，仅通过参数空间中的向量加法操作实现能力迁移。
* **关键优势：** 方法简单且低成本，适合资源受限场景；通过调整权重 w，可以控制推理能力的增强程度，灵活适应不同需求。
* **技术基础：** 受任务向量（Task Vector）研究的启发，利用权重差异在参数空间中表示特定能力的提升方向。

## Experiment

* **有效性：** 在 AIME24 数据集（包含日语和英语数学题目）上，目标日语模型（EZO）初始性能较低（日语/英语题目分别答对 4/7 题），但通过注入推理向量后，性能随权重 w 增加而持续提升，在 w=1.00 时分别答对 10/16 题，甚至超越后训练模型 s1-32B（9/14 题）。
* **实验设置：** 采用 Qwen-32B 作为预训练模型，s1-32B 作为后训练模型，EZO 作为目标模型；评估方法为简单启发式（检查答案是否包含正确标签），数据集规模较小（30 题）。
* **合理性与局限：** 实验验证了推理向量方法的有效性，但模型架构需一致，且缺乏验证集优化权重 w，评估方法较为初步，未来需更鲁棒的策略。

## Further Thoughts

推理向量方法启发我们思考如何在参数空间中解耦和迁移不同能力（如推理、对话、领域知识），不仅限于跨语言迁移，还可能通过多任务向量组合进一步提升模型综合能力；此外，这种轻量化增强路径对其他低资源语言模型开发具有借鉴意义，未来可探索动态调整权重 w 以适应不同任务需求。