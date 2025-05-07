---
title: "Bielik 11B v2 Technical Report"
pubDatetime: 2025-05-05T07:03:41+00:00
slug: "2025-05-bielik-polish-llm"
type: "arxiv"
id: "2505.02410"
score: 0.658684619098032
author: "grok-3-latest"
authors: ["Krzysztof Ociepa", "Łukasz Flis", "Remigiusz Kinas", "Krzysztof Wróbel", "Adrian Gwoździej"]
tags: ["LLM", "Language Adaptation", "Parameter Efficiency", "Instruction Tuning", "Data Quality"]
institution: ["SpeakLeash", "ACK Cyfronet AGH", "Jagiellonian University", "Azurro", "Enelpol"]
description: "本文提出 Bielik 11B v2，一个针对波兰语优化的高效语言模型，通过深度扩展、创新训练方法和高质量数据，在波兰语任务上实现与更大模型相当的性能，同时保持参数效率和部署灵活性。"
---

> **Summary:** 本文提出 Bielik 11B v2，一个针对波兰语优化的高效语言模型，通过深度扩展、创新训练方法和高质量数据，在波兰语任务上实现与更大模型相当的性能，同时保持参数效率和部署灵活性。 

> **Keywords:** LLM, Language Adaptation, Parameter Efficiency, Instruction Tuning, Data Quality

**Authors:** Krzysztof Ociepa, Łukasz Flis, Remigiusz Kinas, Krzysztof Wróbel, Adrian Gwoździej

**Institution(s):** SpeakLeash, ACK Cyfronet AGH, Jagiellonian University, Azurro, Enelpol


## Problem Background

在自然语言处理（NLP）领域，针对资源较少语言（如波兰语）的高性能语言模型开发面临数据稀缺和计算资源不足的挑战。
Bielik 11B v2 的目标是通过构建一个针对波兰语优化的高效模型，解决波兰语文本处理能力不足的问题，同时保持跨语言能力和计算效率，以便在资源受限的环境中部署。

## Method

*   **架构设计：** 基于 Mistral 7B v0.2 模型，采用深度扩展（Depth Up-Scaling）方法，将层数从 32 层增加到 50 层，参数规模达到 11B，在性能和计算效率之间取得平衡。
*   **预训练数据：** 使用 SpeakLeash 项目提供的 1980 亿 token 数据集，其中包括 900 亿 token 的波兰语语料，通过数据清洗和质量评估（使用 XGBoost 分类器，准确率达 94%）确保训练数据的高质量和多样性。
*   **后训练创新：** 引入加权指令交叉熵损失（Weighted Instruction Cross-Entropy Loss, WICEL），根据样本质量分配权重，优先学习高质量数据；采用自适应学习率（Adaptive Learning Rate, ALR），根据上下文长度动态调整学习率，提升训练稳定性。
*   **强化学习：** 使用改进的直接偏好优化（DPO-Positive, DPO-P），通过增加惩罚项避免首选响应的概率下降，提升模型对用户偏好的对齐能力，训练数据包括 7.2 万条偏好指令。
*   **模型合并与量化：** 通过线性合并（Linear Merging）整合不同训练阶段的模型，捕捉多样化能力；提供多种量化选项（如 GGUF, GPTQ），支持在资源受限设备上的高效部署。

## Experiment

*   **有效性：** Bielik 11B v2 在多个波兰语基准测试中表现出色，例如 Open PL LLM Leaderboard（平均分 58.14，指令调优版本达 65.71），Polish MT-Bench（8.56 分），显著优于许多参数规模更大的模型（如 Meta-Llama-3-70B），与前代 Bielik 7B v0.1 相比性能提升明显。
*   **全面性：** 实验覆盖了广泛的波兰语任务（情感分析、问答、文化理解等），并在跨语言任务中测试了泛化能力，包含 0-shot 和 5-shot 评估，量化版本测试显示性能保持率高（如 Q8_0 版本达 65.76%）。
*   **合理性与局限：** 实验设计合理，数据支持结论可信，但在某些领域（如医学任务）与超大规模模型仍有差距，这与训练数据和模型规模限制有关。

## Further Thoughts

加权指令交叉熵损失（WICEL）通过为训练样本分配质量权重，优先学习高质量数据，这一策略可推广到其他资源稀缺语言的模型训练中，尤其在数据质量参差不齐时；自适应学习率（ALR）根据上下文长度动态调整，可能在多语言模型的长序列任务中优化训练难度平衡；此外，针对特定语言的深度优化思路为小语种模型开发提供了参考，表明通过高质量本地化数据和架构调整可在较小参数规模下实现高性能。