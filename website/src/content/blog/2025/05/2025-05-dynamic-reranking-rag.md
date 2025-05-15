---
title: "DynamicRAG: Leveraging Outputs of Large Language Model as Feedback for Dynamic Reranking in Retrieval-Augmented Generation"
pubDatetime: 2025-05-12T05:19:01+00:00
slug: "2025-05-dynamic-reranking-rag"
type: "arxiv"
id: "2505.07233"
score: 0.5197437650451477
author: "grok-3-latest"
authors: ["Jiashuo Sun", "Xianrui Zhong", "Sizhe Zhou", "Jiawei Han"]
tags: ["LLM", "Retrieval-Augmented Generation", "Reranking", "Reinforcement Learning", "Feedback Mechanism"]
institution: ["University of Illinois Urbana-Champaign"]
description: "DynamicRAG 提出了一种基于强化学习的动态重排序框架，利用大型语言模型输出质量作为反馈，显著提升检索增强生成系统的性能和效率，达到最先进的成果。"
---

> **Summary:** DynamicRAG 提出了一种基于强化学习的动态重排序框架，利用大型语言模型输出质量作为反馈，显著提升检索增强生成系统的性能和效率，达到最先进的成果。 

> **Keywords:** LLM, Retrieval-Augmented Generation, Reranking, Reinforcement Learning, Feedback Mechanism

**Authors:** Jiashuo Sun, Xianrui Zhong, Sizhe Zhou, Jiawei Han

**Institution(s):** University of Illinois Urbana-Champaign


## Problem Background

检索增强生成（RAG）系统通过结合大型语言模型（LLMs）和外部知识检索，在知识密集型任务中表现出色，但传统重排序器（reranker）采用静态方法，无法根据查询复杂性动态调整检索文档的数量（k）和顺序，导致生成质量下降或效率低下。
论文旨在解决这一问题，探索如何利用 LLMs 输出质量作为反馈信号，动态优化重排序决策，提升 RAG 系统的整体性能。

## Method

*   **核心思想:** 提出 DynamicRAG 框架，将重排序器建模为一个通过强化学习（Reinforcement Learning, RL）优化的智能体，根据查询动态调整检索文档的顺序和数量，同时利用生成器输出质量作为反馈信号进行优化。
*   **具体实现:** 训练过程分为两阶段：
    *   **行为克隆（Behavioral Cloning）阶段:** 通过监督微调（Supervised Fine-Tuning, SFT），利用专家轨迹（如 MonoT5 模型生成的排序结果）训练重排序器，使其具备基础的动态排序能力，减少动作空间的复杂性。
    *   **交互学习阶段:** 将生成器视为环境，重排序器通过与环境交互生成多个轨迹（trajectories），并基于生成质量计算奖励，使用直接偏好优化（Direct Preference Optimization, DPO）方法优化策略，优先选择高奖励轨迹。
*   **奖励函数设计:** 奖励函数综合了五个维度：精确匹配（Exact Match, EM）、语义相似度（Semantic Similarity, SS）、文本流畅性（Textual Fluency, TF）、长度惩罚（Length Penalty, LP）和基于 LLM 的评估（LLM-Eval），以全面评估生成质量。
*   **系统架构:** 框架包括三个组件：冻结的检索器（负责初始文档检索）、可训练的重排序器（动态选择和排序文档）和可训练的生成器（基于重排序文档生成答案），通过联合训练实现整体优化。
*   **关键创新:** 不依赖静态 k 值，而是根据查询难度动态调整文档数量；利用生成质量作为直接反馈，而非仅依赖 LLMs 内部知识。

## Experiment

*   **有效性:** DynamicRAG 在七个知识密集型数据集（NQ, TriviaQA, HotpotQA, 2WikimQA, ASQA, FEVER, ELI5）上显著优于基线方法，例如在 NQ 数据集上基于 LLaMA3-8B 的 EM 得分为 48.4，超越 GPT-4o（40.0）和 RankRAG（42.4），实现 SOTA 性能。
*   **重排序性能:** 在 NQ 和 HotpotQA 数据集上，重排序器的召回率（R@5, R@10, R@20）平均得分达 73.7，优于其他开源模型如 RankLLaMA 和 monoT5，且仅用 20k 训练样本即可媲美需 50k 样本的 RankRAG。
*   **效率优势:** DynamicRAG 仅需两次 LLM 调用即可生成答案，相较于 RankRAG 的多次调用（约 17 倍延迟），计算效率显著提升；与无重排序的 Vanilla-RAG 相比，仅增加 2.3 倍延迟，但性能提升明显（如 NQ 上提升 9.6 个百分点）。
*   **实验设置合理性:** 实验覆盖多种模型规模（LLaMA2-7B/13B, LLaMA3-8B）、数据集类型和消融研究（如移除 RL 或检索组件），验证了各组件的重要性；此外，测试了不同检索器（如 DPR, Contriever）的影响，证明方法鲁棒性。
*   **潜在不足:** 训练数据量（约 150k）较少于部分基线（如 RankRAG 的 470k），但仍取得优异结果，显示方法的高效性；对闭源模型（如 GPT-4o）的适配性验证较少，尽管初步结果显示性能提升。

## Further Thoughts

DynamicRAG 将重排序器视为 RL 智能体并利用生成质量作为反馈的思路非常启发性，提示我们可以在其他 NLP 任务（如对话系统或文本摘要）中探索类似的动态调整机制；此外，多维度奖励函数的设计启发我们可以在模型优化中引入更全面的评估指标，而不仅仅依赖单一准确率；一个发散性想法是，是否可以利用更强大的闭源模型（如 GPT-4o）生成的输出质量作为奖励信号，指导开源模型（如 LLaMA）的优化，从而在资源受限场景下进一步提升性能。