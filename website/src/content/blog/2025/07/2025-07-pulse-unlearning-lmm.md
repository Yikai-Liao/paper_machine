---
title: "PULSE: Practical Evaluation Scenarios for Large Multimodal Model Unlearning"
pubDatetime: 2025-07-02T01:13:08+00:00
slug: "2025-07-pulse-unlearning-lmm"
type: "arxiv"
id: "2507.01271"
score: 0.5736446727394493
author: "grok-3-latest"
authors: ["Tatsuki Kawakami", "Kazuki Egashira", "Atsuyuki Miyai", "Go Irie", "Kiyoharu Aizawa"]
tags: ["LLM", "Multimodal Model", "Unlearning", "Pre-Training", "Sustainability"]
institution: ["The University of Tokyo"]
description: "本文提出 PULSE 协议，为大型多模态模型（LMMs）的遗忘技术提供实用评估框架，揭示现有方法在预训练知识遗忘和多次遗忘场景下的不足。"
---

> **Summary:** 本文提出 PULSE 协议，为大型多模态模型（LMMs）的遗忘技术提供实用评估框架，揭示现有方法在预训练知识遗忘和多次遗忘场景下的不足。 

> **Keywords:** LLM, Multimodal Model, Unlearning, Pre-Training, Sustainability

**Authors:** Tatsuki Kawakami, Kazuki Egashira, Atsuyuki Miyai, Go Irie, Kiyoharu Aizawa

**Institution(s):** The University of Tokyo


## Problem Background

大型多模态模型（LMMs）在训练中可能包含隐私数据或版权内容，引发隐私和知识产权问题，因此需要‘遗忘’（Unlearning）技术来删除特定信息，同时尽量保留其他任务性能。
现有 LMMs 遗忘基准（如 MLLMU-Bench）仅关注微调阶段知识的遗忘，忽略预训练知识，且未考虑现实中多次顺序性遗忘请求的场景，缺乏实用性评估框架。

## Method

*   **核心思想：** 提出 PULSE 协议（Practical Unlearning Scenarios Evaluation），一个针对 LMMs 遗忘的实用评估框架，从两个关键视角评估现有遗忘方法的表现。
*   **具体设计：**
    *   **Pre-trained Knowledge Unlearning（预训练知识遗忘）：** 评估模型遗忘预训练阶段获取知识的能力，区别于微调知识遗忘，关注更深层次嵌入模型中的信息，实验中选择预训练模型已熟知的目标（如名人数据）作为遗忘对象。
    *   **Long-term Sustainability Evaluation（长期可持续性评估）：** 模拟现实中多次顺序性遗忘请求的场景，将遗忘目标数据分成多个子集，顺序执行多次遗忘操作，观察模型性能随操作次数变化。
*   **评估指标：** 使用遗忘效果（Effectiveness，即对目标数据 _D_unlearn 的准确率下降）和泛化能力（Generality，即对保留数据 _D_retain 和标准基准如 MMBench 的性能保持）作为衡量标准。
*   **实验管道：** 基于公开数据集（如 MLLMU-Bench），设计不同实验设置，覆盖微调知识遗忘、预训练知识遗忘和可持续性评估，测试多种遗忘方法（如 Gradient Ascent, GA with KL Regularization, Negative Preference Optimization）。

## Experiment

*   **有效性：** 实验表明，现有遗忘方法（如 GA, GA+KLR, NPO）在遗忘微调知识时效果尚可（MMBench 性能下降约 10%），但在遗忘预训练知识时泛化能力大幅下降（MMBench 性能下降超 90%），显示预训练知识遗忘难度更大。
*   **可持续性：** 在多次顺序性遗忘实验中，随着遗忘操作次数增加（5 次），模型对目标数据准确率下降（遗忘有效），但对保留数据和 MMBench 的性能也显著下降，几乎完全丧失泛化能力，表明现有方法无法应对现实中的多次遗忘请求。
*   **任务差异：** 多模态任务比纯文本任务更容易被遗忘，但可能是通过破坏图像与知识对齐而非真正遗忘，效果存疑。
*   **实验设置合理性：** 实验覆盖微调、预训练和可持续性三个维度，数据选择（如 MLLMU-Bench）和指标设计（Effectiveness 和 Generality）较为全面，但结果显示现有方法在遗忘效果与泛化能力权衡上存在明显局限。

## Further Thoughts

预训练与微调知识遗忘难度的差异启发我们，未来可针对不同训练阶段设计差异化遗忘策略，如在预训练阶段引入遗忘友好的正则化机制；可持续性评估揭示多次遗忘导致的灾难性遗忘问题，提示可引入记忆保护机制（如参数冻结）减少对保留任务影响；多模态与文本任务遗忘效果差异表明，LMMs 遗忘技术可能需要模块化设计，分别处理不同模态表示层。