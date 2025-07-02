---
title: "Predicting thinking time in Reasoning models"
pubDatetime: 2025-06-29T15:01:01+00:00
slug: "2025-06-thinking-time-prediction"
type: "arxiv"
id: "2506.23274"
score: 0.6753121303109939
author: "grok-3-latest"
authors: ["Hans Peter Lyngsøe Raaschou-Jensen", "Constanza Fierro", "Anders Søgaard"]
tags: ["LLM", "Reasoning", "Progress Prediction", "User Interaction", "Hidden States"]
institution: ["Department of Computer Science, University of Copenhagen"]
description: "本文提出并验证了通过模型隐藏状态预测推理时间的方法，开发‘推理进度条’以改善用户体验，尽管在推理质量和泛化能力上存在权衡。"
---

> **Summary:** 本文提出并验证了通过模型隐藏状态预测推理时间的方法，开发‘推理进度条’以改善用户体验，尽管在推理质量和泛化能力上存在权衡。 

> **Keywords:** LLM, Reasoning, Progress Prediction, User Interaction, Hidden States

**Authors:** Hans Peter Lyngsøe Raaschou-Jensen, Constanza Fierro, Anders Søgaard

**Institution(s):** Department of Computer Science, University of Copenhagen


## Problem Background

推理模型在生成隐藏推理链时，用户无法预测模型的思考时间（Thinking Time），这导致用户体验问题，如等待时间不确定引发的沮丧或任务放弃；
随着大型语言模型（LLMs）应用于更长、更异步的任务，这种问题愈发突出，论文旨在通过开发‘推理进度条’（Progress Bar for Reasoning）来解决这一用户交互挑战。

## Method

*   **核心思想:** 开发在线和离线方法预测推理模型的思考时间，利用模型内部隐藏状态中编码的进度信息，为用户提供实时反馈。
*   **在线预测方法:**
    *   **提示（Prompting）:** 使用现有LLM（如 Qwen3-14B）通过提示估计推理进度，作为基线，但效果较差。
    *   **探针（Probe）:** 设计轻量级多层感知机（MLP）和Transformer预测头，从模型隐藏状态中提取进度信息；MLP探针将隐藏状态映射到进度桶的概率分布，Transformer探针则考虑隐藏状态随时间的变化。
    *   **监督微调（SFT）与强化学习（RL）:** 在推理轨迹中插入进度标记（如 `<progressbar>`），通过监督微调训练模型生成进度信息；随后使用强化学习（如 GRPO）优化进度预测准确性、标记格式一致性和数学推理正确性，通过多目标奖励函数平衡不同目标。
*   **离线预测方法:** 仅基于输入问题预测总推理时间，但由于缺乏上下文信息，效果不如在线预测。
*   **关键点:** 方法不改变模型核心架构，仅通过探针或轻量级微调实现进度预测，确保与现有推理引擎（如 VLLM）的兼容性；同时探索了进度预测与推理质量之间的权衡。

## Experiment

*   **有效性:** MLP探针在分布内数据上表现较好（准确率约45%，MAE 0.82-0.97），表明模型隐藏状态中确实编码了进度信息；在线预测优于离线预测，特别是在利用上下文信息时。
*   **局限性:** 在分布外（OOD）数据上，探针性能显著下降（MAE 2.87-3.08），显示泛化能力不足；通过SFT和RL插入进度标记后，数学推理准确率下降（如 MATH500 上从63%降至53.8%-64.2%），表明进度预测与任务性能存在冲突。
*   **实验设置:** 实验在数学推理数据集（如 MATH500, AMC23, OlympiadBench）上进行，使用 DeepScaleR 1.5B 模型；方法覆盖提示、探针、微调和RL，较为全面；但受硬件限制，上下文长度受限（RL训练时最大完成长度为8096 tokens，原始模型支持24K），且未测试非数学领域，限制了结果的通用性评估。

## Further Thoughts

模型隐藏状态中天然编码进度信息的发现，启发我们可以在预训练阶段设计特定任务以增强这种编码能力；进度预测与任务性能的权衡问题，提示未来可探索多任务学习或动态奖励调整来优化用户体验与核心任务表现；在线预测的上下文依赖性优势，表明类似方法可应用于对话系统或代码生成等动态任务中，预测任务完成时间或复杂性。