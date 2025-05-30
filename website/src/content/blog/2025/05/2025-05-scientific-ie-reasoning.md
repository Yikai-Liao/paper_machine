---
title: "Beyond path selection: Better LLMs for Scientific Information Extraction with MimicSFT and Relevance and Rule-induced(R$^2$)GRPO"
pubDatetime: 2025-05-28T07:47:46+00:00
slug: "2025-05-scientific-ie-reasoning"
type: "arxiv"
id: "2505.22068"
score: 0.6970760476288594
author: "grok-3-latest"
authors: ["Ran Li", "Yuchen Liu", "Chen Jing", "Shimin Di", "Yu Qiu", "Lei Chen"]
tags: ["LLM", "Information Extraction", "Reasoning", "Supervised Fine-Tuning", "Reinforcement Learning"]
institution: ["Hong Kong University of Science and Technology (HKUST)", "Hong Kong University of Science and Technology (Guangzhou) (HKUST-GZ)", "Southeast University (SEU)", "Zhipu AI"]
description: "本文提出 MimicSFT 和 R² GRPO 两阶段训练方法，通过结构化推理模板和复合奖励函数显著提升大型语言模型在科学信息提取任务中的推理能力和性能，超越监督基线并挑战 RLVR 仅优化路径的观点。"
---

> **Summary:** 本文提出 MimicSFT 和 R² GRPO 两阶段训练方法，通过结构化推理模板和复合奖励函数显著提升大型语言模型在科学信息提取任务中的推理能力和性能，超越监督基线并挑战 RLVR 仅优化路径的观点。 

> **Keywords:** LLM, Information Extraction, Reasoning, Supervised Fine-Tuning, Reinforcement Learning

**Authors:** Ran Li, Yuchen Liu, Chen Jing, Shimin Di, Yu Qiu, Lei Chen

**Institution(s):** Hong Kong University of Science and Technology (HKUST), Hong Kong University of Science and Technology (Guangzhou) (HKUST-GZ), Southeast University (SEU), Zhipu AI


## Problem Background

大型语言模型（LLM）在科学信息提取（SciIE）任务中表现不佳，尤其是在命名实体识别（NER）和关系提取（RE）方面，常常不如小型 BERT 基线模型。
SciIE 任务同时需要知识记忆和上下文推理能力，而现有 LLM 的训练目标与此需求不完全对齐，导致在实体边界检测和关系推理上存在不足。
论文旨在探索如何通过监督微调（SFT）和强化学习（RLVR）等后训练方法提升 LLM 在 SciIE 任务上的表现，并挑战 RLVR 仅优化推理路径而不提升推理能力的传统观点。

## Method

*   **整体框架:** 提出了一种两阶段训练方法，结合 MimicSFT 和 R² GRPO，旨在提升 LLM 在 SciIE 任务中的推理能力和性能。
*   **MimicSFT（第一阶段）:**
    *   一种改进的监督微调方法，通过引入结构化的伪推理模板（pseudo reasoning templates），引导模型在生成最终输出前生成推理步骤（标记为 `<reasoning>...</reasoning>`）。
    *   不需要高质量链式思维（CoT）数据，而是使用通用的信息提取流程模板（如‘识别实体’、‘考虑关系’、‘制定提取’）。
    *   将 SciIE 任务分解为多个子任务（如 NER、RE with Gold Entities、End-to-End IE），采用多任务学习策略以提升泛化能力。
    *   使用低秩适应（LoRA）进行高效微调，优化条件概率以适应目标任务。
*   **R² GRPO（第二阶段）:**
    *   一种基于强化学习的训练方法，扩展了 Group Relative Policy Optimization (GRPO)，通过分层推理进一步优化模型输出。
    *   引入复合奖励函数，包括：F1 分数奖励（衡量预测与真实提取的匹配度）、实体边界奖励（鼓励精确边界检测）、相关性奖励（促进基于证据的提取）、规则模式奖励（鼓励遵循逻辑或领域模式，如‘导致’、‘规则暗示’等）。
    *   采用课程学习（从简单到复杂任务）和数据选择策略（优先选择 SFT 表现差但奖励信号清晰的样本），以提高训练效率。
    *   通过分层推理（hierarchical reasoning）分解复杂任务，生成多级推理步骤（如 `<think>...</think>` 细化推理），确保满足模式约束和事实约束。
*   **理论支持:** 从约束生成（constrained generation）视角分析，通过任务分解和分层推理，模型能更有效地满足复杂约束，提升输出质量。

## Experiment

*   **有效性:** 基于 Qwen2.5-7B-Instruct 模型，在 SciER 数据集上，R² GRPO*（结合 MimicSFT 和 R² GRPO）显著提升性能，NER F1 达到 84.36，关系提取（Rel F1）达到 66.81，Rel+ F1 达到 65.95，超越所有监督基线（如 HGERE 的 Rel F1 62.32）；在 Best@5 指标下，Rel F1 进一步提升至 74.38。
*   **对比分析:** MimicSFT 相较标准 SFT 在关系提取上有明显提升（Rel F1 从 42.22 提升至 56.02），表明伪推理模板有效激活推理能力；R² GRPO 相较基本 GRPO 也有改进（Rel F1 从 48.84 提升至 54.59），显示复合奖励函数的作用；两阶段结合（R² GRPO*）实现了最佳效果。
*   **泛化能力:** 在 OOD 数据集上，R² GRPO* 保持较好性能（Rel F1 55.08，Best@5 达到 66.74），表明分层推理有助于学习泛化模式。
*   **实验设置合理性:** 实验对比了多种基线（零样本 LLM、蒸馏模型、监督 BERT 模型），通过 Best F1@K 和 Avg@K 指标分析推理能力和一致性；消融研究验证了各组件贡献；温度敏感性分析显示 SciIE 任务更适合低温度生成（<0.6），避免噪声干扰。
*   **不足与权衡:** 实验主要集中于 SciER 数据集，缺乏多领域验证；RLVR 模型在高 K 值下的 Best F1@K 略低于 MimicSFT，反映出探索能力的轻微劣势，但 Avg@K 和 Best F1@1 更高，显示其在实际应用中的可靠性。

## Further Thoughts

MimicSFT 的伪推理模板表明，即使缺乏高质量 CoT 数据，结构化推理也能显著提升约束生成任务的表现，启发我们探索自动化模板生成或结合领域知识进一步优化，尤其在数据稀缺场景下；
R² GRPO 的复合奖励函数设计提示我们在其他任务中可以考虑多目标优化，而不仅仅依赖单一指标；
SFT 和 RLVR 的互补性表明后训练中可以探索动态混合策略，例如根据任务阶段调整训练方法比例，以平衡知识扩展和输出优化。