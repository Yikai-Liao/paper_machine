---
title: "Improving LLM-Generated Code Quality with GRPO"
pubDatetime: 2025-06-02T19:50:16+00:00
slug: "2025-06-llm-code-quality-grpo"
type: "arxiv"
id: "2506.02211"
score: 0.48696283678750696
author: "grok-3-latest"
authors: ["Maxime Robeyns", "Laurence Aitchison"]
tags: ["LLM", "Code Quality", "Reinforcement Learning", "Reward Design", "Synthetic Dataset"]
institution: ["University of Bristol"]
description: "本文通过开发 `codequal_analyzer` 库并将其集成到 GRPO 框架中，利用代码质量作为奖励信号，显著提升了 LLM 生成代码的质量，同时维持了功能正确性。"
---

> **Summary:** 本文通过开发 `codequal_analyzer` 库并将其集成到 GRPO 框架中，利用代码质量作为奖励信号，显著提升了 LLM 生成代码的质量，同时维持了功能正确性。 

> **Keywords:** LLM, Code Quality, Reinforcement Learning, Reward Design, Synthetic Dataset

**Authors:** Maxime Robeyns, Laurence Aitchison

**Institution(s):** University of Bristol


## Problem Background

大型语言模型（LLMs）在代码生成中广泛应用，但现有训练方法主要以功能正确性（如单元测试通过率）作为奖励信号，忽视了代码的可维护性、安全性、可靠性和性能等质量维度，导致生成的代码可能难以维护、存在安全漏洞或资源浪费。本文旨在通过引入代码质量作为奖励信号，提升 LLM 生成代码的整体质量。

## Method

* **核心思想**：通过强化学习（RL）框架，将代码质量度量作为奖励信号，训练 LLM 生成不仅功能正确且质量更高的代码。
* **代码质量评估工具**：开发了 `codequal_analyzer` 库，基于 CISQ 标准，结合现有工具（如 Pylint、Radon、MyPy）和自定义分析器，评估代码在可维护性、安全性、性能和可靠性四个方面的质量，并将问题映射到 CWE ID，最终生成一个 0 到 1 的质量分数作为奖励信号。
* **训练框架**：采用 GRPO（Group Relative Policy Optimization）算法，通过采样多个候选输出并基于相对奖励更新策略。奖励信号包括三部分：格式奖励（确保代码格式正确）、正确性奖励（基于单元测试通过率）和质量奖励（基于 `codequal_analyzer` 分数），其中质量奖励占比略高。
* **数据集设计**：由于现有基准数据集（如 MBPP、HumanEval）问题复杂度不足以体现代码质量差异，作者设计了一个合成数据集生成流程，生成 200 个多样化的 Python 编码问题，覆盖算法优化、错误处理、安全性等多个类别，以暴露模型在代码质量上的弱点。
* **实现细节**：质量分数通过对不同严重度问题（信息、低、中、高、关键）的加权求和并归一化计算，确保奖励信号反映代码质量的综合表现。

## Experiment

* **实验设置**：在多个开源 LLM（如 Qwen 2.5 3B、Llama 3.2 3B、OLMo 2 1B）上进行实验，使用 200 个合成 Python 编码问题作为数据集，比较加入质量奖励和不加入质量奖励的模型表现。
* **效果显著**：加入质量奖励的模型在验证集上的代码质量分数显著提升，例如 Qwen 2.5 3B 的质量分数从 0.766 提升到 0.878（+0.112）；同时，功能正确性分数维持或略有提升（如 Qwen 2.5 3B 从 0.567 提升到 0.690）。
* **人类评估验证**：通过盲测，专家人类评估者在 78.6% 的比较中更偏好加入质量奖励的模型生成的代码，效果显著（p < 0.001），表明方法确实提升了代码质量。
* **计算开销**：`codequal_analyzer` 库的分析时间小于 1 秒/次，计算成本低，适合 RL 训练流程。
* **合理性与全面性**：实验涵盖多个模型和多样化数据集，通过人类评估避免了奖励机制的‘作弊’问题，设置合理且结果可信。

## Further Thoughts

本文将传统软件工程中的代码质量度量引入 LLM 训练，启发我们可以在其他生成任务中引入领域特定质量度量作为奖励信号；合成数据集的设计方法提示我们可以通过复杂任务暴露模型弱点，为现实场景中的模型评估和改进提供新思路；多维度奖励设计也表明未来的 RL 训练可以进一步探索多目标优化，平衡不同性能需求。