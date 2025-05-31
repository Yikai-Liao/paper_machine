---
title: "Beyond path selection: Better LLMs for Scientific Information Extraction with MimicSFT and Relevance and Rule-induced(R$^2$)GRPO"
pubDatetime: 2025-05-28T07:47:46+00:00
slug: "2025-05-scientific-ie-mimicsft-r2grpo"
type: "arxiv"
id: "2505.22068"
score: 0.6970760476288594
author: "grok-3-latest"
authors: ["Ran Li", "Yuchen Liu", "Chen Jing", "Shimin Di", "Yu Qiu", "Lei Chen"]
tags: ["LLM", "Information Extraction", "Supervised Fine-Tuning", "Reinforcement Learning", "Reasoning", "Structured Templates", "Post-Training"]
institution: ["Hong Kong University of Science and Technology (HKUST)", "Hong Kong University of Science and Technology (Guangzhou Campus)", "Southeast University (SEU)", "Zhipu AI"]
description: "本文提出 MimicSFT 和 R²GRPO 两阶段训练方法，通过结构化推理模板和复合奖励机制显著提升了大型语言模型在科学信息提取任务上的推理能力和知识整合效果，并在 SciER 数据集上超越了监督基线模型。"
---

> **Summary:** 本文提出 MimicSFT 和 R²GRPO 两阶段训练方法，通过结构化推理模板和复合奖励机制显著提升了大型语言模型在科学信息提取任务上的推理能力和知识整合效果，并在 SciER 数据集上超越了监督基线模型。 

> **Keywords:** LLM, Information Extraction, Supervised Fine-Tuning, Reinforcement Learning, Reasoning, Structured Templates, Post-Training

**Authors:** Ran Li, Yuchen Liu, Chen Jing, Shimin Di, Yu Qiu, Lei Chen

**Institution(s):** Hong Kong University of Science and Technology (HKUST), Hong Kong University of Science and Technology (Guangzhou Campus), Southeast University (SEU), Zhipu AI


## Problem Background

大型语言模型（LLMs）在科学信息提取（SciIE）任务上的表现不如小型 BERT 基线模型，原因是 SciIE 同时需要知识记忆和上下文推理，而 LLMs 的训练目标与此需求不完全对齐。
论文旨在探索监督微调（SFT）和强化学习（RLVR）是否能通过后训练提升 LLMs 的推理能力和知识整合效果，挑战 RLVR 仅优化推理路径而不提升推理能力的传统观点。

## Method

*   **核心思想:** 提出一个两阶段训练框架，通过结构化推理和任务感知的奖励机制，提升 LLMs 在 SciIE 任务上的表现。
*   **MimicSFT（第一阶段）:** 一种监督微调方法，利用伪推理模板（pseudo reasoning templates）引导模型生成结构化推理步骤，而无需高质量链式思维（CoT）数据。具体做法是将信息提取任务分解为多个子任务（如仅 NER、仅 RE、端到端 IE），采用多任务学习策略，训练模型在输出最终结果前生成一个标记为 `<reasoning>...</reasoning>` 的推理块。这种方法旨在通过结构化提示激活模型的推理能力，增强其对模式约束和事实约束的满足能力。
*   **R²GRPO（第二阶段）:** 一种基于强化学习的训练方法，扩展了 Group Relative Policy Optimization (GRPO)，通过一个复合奖励函数（包括 F1 分数奖励、实体边界奖励、相关性奖励和规则模式奖励）优化推理路径和知识整合。R²GRPO 引入分层推理机制，将复杂任务分解为多个推理阶段（如 `<reasoning>` 和 `<think>` 两个层次），并采用课程学习（从简单到复杂任务）和数据选择策略（优先选择 SFT 表现差但奖励信号清晰的样本）提高训练效率。
*   **理论支持:** 作者从约束生成视角（constrained generation view）解释方法有效性，认为通过分层推理逐步满足模式约束和事实约束，可以将复杂问题转化为更易处理的形式，并通过数学推导证明分层推理能提高约束满足概率。
*   **实现细节:** 所有微调均采用 LoRA（Low-Rank Adaptation）方法以降低计算成本，训练基于 Qwen2.5-7B-Instruct 模型，参数设置包括不同的学习率和批量大小。

## Experiment

*   **有效性:** 在 SciER 数据集上，MimicSFT 和 R²GRPO 显著提升了模型性能，R²GRPO*（两阶段结合）在关系提取（Rel）任务上的 F1 分数达到 66.81，超越所有监督基线模型（如 HGERE 的 62.32）；在 Best@5 指标下，Rel F1 进一步提升至 74.38，显示出模型在多次生成中的潜力。
*   **对比分析:** MimicSFT 相较标准 SFT 在关系提取任务上提升明显（SciER Rel F1 从 42.22 提升至 56.02），证明伪推理模板能有效激活推理能力；R²GRPO 相较基本 GRPO 也有显著改进（Rel F1 从 48.84 提升至 54.59），显示复合奖励函数的作用；两阶段结合（R²GRPO*）实现了知识获取和推理优化的协同效应。
*   **合理性与全面性:** 实验设置覆盖了 SciER 和 OOD 数据集，评估了 NER 和 RE 任务，采用了 Best F1@K 和 Avg@K 指标分析模型的上限能力和平均表现，K 值从 1 到 1024，验证了结构化推理对知识整合和泛化能力的提升；此外，消融研究和温度敏感性分析进一步确认了各组件贡献和模型对确定性生成的偏好。
*   **局限性:** 实验主要聚焦 SciIE 任务，未验证方法在其他信息提取任务或语言上的适应性；R²GRPO 的训练对计算资源需求较高（如大组大小需要更多显存），可能限制其在更大模型上的应用。

## Further Thoughts

MimicSFT 的伪推理模板设计启发我们可以在其他任务中探索低成本的结构化提示方法，以激活模型推理能力，而无需依赖高质量标注数据；R²GRPO 的复合奖励函数为强化学习在多约束任务中的应用提供了新思路，未来可以设计更多任务特定的奖励机制来平衡探索与利用；分层推理机制对复杂任务的分解方式值得借鉴，特别是在需要逐步满足多重约束的场景中，可以进一步研究自动化模板生成或多层次推理优化方法。