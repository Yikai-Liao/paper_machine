---
title: "Point-It-Out: Benchmarking Embodied Reasoning for Vision Language Models in Multi-Stage Visual Grounding"
pubDatetime: 2025-09-30T05:05:54+00:00
slug: "2025-09-visual-grounding-benchmark"
type: "arxiv"
id: "2509.25794"
score: 0.6950525337709305
author: "grok-3-latest"
authors: ["Haotian Xue", "Yunhao Ge", "Yu Zeng", "Zhaoshuo Li", "Ming-Yu Liu", "Yongxin Chen", "Jiaojiao Fan"]
tags: ["Vision-Language Model", "Embodied Reasoning", "Visual Grounding", "Benchmarking", "Task Planning"]
institution: ["Georgia Tech", "NVIDIA"]
description: "本文提出 Point-It-Out (PIO) 基准测试，通过分层评估框架系统性地测试视觉-语言模型在具身推理中的精确视觉定位能力，揭示了当前模型在细粒度定位和多步规划中的局限性。"
---

> **Summary:** 本文提出 Point-It-Out (PIO) 基准测试，通过分层评估框架系统性地测试视觉-语言模型在具身推理中的精确视觉定位能力，揭示了当前模型在细粒度定位和多步规划中的局限性。 

> **Keywords:** Vision-Language Model, Embodied Reasoning, Visual Grounding, Benchmarking, Task Planning

**Authors:** Haotian Xue, Yunhao Ge, Yu Zeng, Zhaoshuo Li, Ming-Yu Liu, Yongxin Chen, Jiaojiao Fan

**Institution(s):** Georgia Tech, NVIDIA


## Problem Background

视觉-语言模型（Vision-Language Models, VLMs）在具身推理（Embodied Reasoning, ER）任务中的评估目前主要依赖多选题或语言规划，缺乏对模型在物理世界中‘感知-行动’闭环能力的直接测试，尤其是在需要精确视觉定位（Visual Grounding）的情况下。
论文指出，现有基准测试无法有效捕捉模型在真实世界交互中的细粒度定位和规划能力，因此提出一个关键问题：如何设计一个系统性的基准测试，直接评估 VLMs 在不同复杂度的具身任务中的视觉定位能力？

## Method

*   **核心思想:** 提出 Point-It-Out (PIO) 基准测试，通过分层评估协议系统性地测试 VLMs 在具身推理中的精确视觉定位能力。
*   **具体框架:** 将评估分为三个阶段：
    *   **S1: 指称对象定位（Referred Object Localization）**：根据语言指令定位图像中的特定对象，可能包含空间、颜色或材料等约束，测试模型的基本视觉-语言映射能力。
    *   **S2: 任务驱动定位（Task-Driven Grounding）**：在 S1 基础上，要求模型推断与任务相关的交互点（如物体可供操作的部位），即使这些点未在指令中明确提及，涉及对物体功能（Affordance）和任务上下文的理解。
    *   **S3: 视觉轨迹预测（Visual Trace Prediction）**：结合 S1 和 S2 的能力，要求模型生成完成任务的粗略 2D 视觉轨迹，引入时间维度，测试模型的多步规划和时空推理能力。
*   **数据与场景:** 收集了超过 600 个标注数据点，覆盖室内环境、厨房场景、驾驶场景和机器人操作四个关键领域，确保任务多样性和现实性。
*   **评估方式:** 测试了超过 10 个最先进的 VLMs，输出格式包括点和边界框，并设计归一化的 IoU 指标以公平比较不同输出格式；对于 S3 阶段，采用人工评分和基于 GPT 的评估方法。

## Experiment

*   **有效性:** 在 S1 和 S2 阶段，专门为定位任务微调的模型（如 RoboRefer、MoLMO、Qwen2.5-VL）表现优于通用模型（如 GPT-4o、Claude-3.7），表明微调对细粒度定位任务至关重要；在 S3 阶段，通用模型（如 Gemini-2.5-Pro、GPT-o3）表现更好，显示出在多步规划和时间推理方面的优势。
*   **局限性:** 所有模型在细粒度任务（如对象部件定位和功能性接触点预测）上表现不佳，准确率普遍较低（如 MoLMO 在功能性任务中得分低于 0.4），揭示了当前 VLMs 在具身推理中的不足。
*   **实验设置:** 实验覆盖多个场景和任务类型，数据量充足（超过 600 个标注数据点），评估指标设计合理（如归一化 IoU 和 S3 的人工评分），考虑了不同输出格式的公平性，整体设置全面且合理。

## Further Thoughts

论文提出的分层评估框架（S1-S3）不仅适用于 VLMs 的具身推理评估，还可能推广到其他多模态任务中，如视频理解或交互式学习；此外，视觉轨迹预测作为中间表示的潜力启发我们可以在模型训练中引入更多时间维度的数据，以增强规划能力；同时，通用模型和微调模型在不同任务上的能力差异提示我们可以通过混合训练策略（结合通用预训练和任务特定微调）来平衡定位精度和规划能力。