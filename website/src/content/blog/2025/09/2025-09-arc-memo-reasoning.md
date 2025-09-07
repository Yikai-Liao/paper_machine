---
title: "ArcMemo: Abstract Reasoning Composition with Lifelong LLM Memory"
pubDatetime: 2025-09-04T17:54:19+00:00
slug: "2025-09-arc-memo-reasoning"
type: "arxiv"
id: "2509.04439"
score: 0.7536217999906712
author: "grok-3-latest"
authors: ["Matthew Ho", "Chen Si", "Zhaoxiang Feng", "Fangxu Yu", "Zhijian Liu", "Zhiting Hu", "Lianhui Qin"]
tags: ["LLM", "Memory Augmentation", "Abstract Reasoning", "Continual Learning", "Concept Abstraction"]
institution: ["University of California, San Diego", "University of Maryland"]
description: "本文提出 ArcMemo 框架，通过抽象概念级记忆增强大型语言模型在组合推理任务中的持续学习能力，在 ARC-AGI 基准上实现显著性能提升。"
---

> **Summary:** 本文提出 ArcMemo 框架，通过抽象概念级记忆增强大型语言模型在组合推理任务中的持续学习能力，在 ARC-AGI 基准上实现显著性能提升。 

> **Keywords:** LLM, Memory Augmentation, Abstract Reasoning, Continual Learning, Concept Abstraction

**Authors:** Matthew Ho, Chen Si, Zhaoxiang Feng, Fangxu Yu, Zhijian Liu, Zhiting Hu, Lianhui Qin

**Institution(s):** University of California, San Diego, University of Maryland


## Problem Background

大型语言模型 (LLMs) 在推理密集型任务中表现出色，但推理过程中发现的模式和策略在上下文重置后会被丢弃，无法持续利用。
这与人类通过积累经验和抽象模式解决复杂问题的能力形成对比，限制了 LLMs 在跨问题推理中的表现。
论文提出通过外部记忆机制持久化这些发现，超越传统的实例级记忆（与具体问题紧密耦合），转向概念级记忆，以支持更广泛的复用和组合推理。

## Method

*   **核心思想:** 设计一个名为 ArcMemo 的外部记忆框架，通过存储抽象概念级记忆（而非具体问题实例），增强 LLMs 在组合推理任务中的持续学习能力，支持测试时动态更新而无需权重调整。
*   **记忆格式 (Memory Format):** 提出了两种记忆格式以存储抽象概念：
    *   **开放式 (Open-Ended, OE):** 以‘情境-建议’对的形式存储，结构约束最小，强调从原始问题情境中解耦核心思想，便于跨问题复用。
    *   **程序合成式 (Program Synthesis, PS):** 受软件工程和函数式编程启发，将概念定义为类型、结构和例程，支持参数化和类型注解，鼓励高阶函数以实现更高层次的抽象和模块化组合。
*   **记忆写入 (Memory Write):** 从推理轨迹中提取抽象概念并更新记忆：
    *   OE 方式通过模型反思总结生成情境-建议对，必要时通过后验推导重建推理过程。
    *   PS 方式通过预处理解决方案为伪代码，优先提取高层次操作，并结合现有记忆更新概念描述、参数列表等，减少冗余并提升通用性。
*   **记忆读取 (Memory Read):** 为新问题选择相关记忆条目以辅助推理：
    *   OE 方式通过预处理问题描述（例如使用视觉语言模型将空间推理问题转为自然语言描述），并采用 top-k 选择机制挑选相关条目。
    *   PS 方式采用推理驱动的选择机制，利用 LLMs 的长篇推理和回溯能力（系统2式思考），根据相关性线索和类型注解探索和匹配概念。
*   **持续学习:** 支持测试时动态更新记忆，通过反馈机制（如测试用例或自我反思）筛选有效模式，确保只保留有用的概念，模拟人类学习的积累过程。

## Experiment

*   **数据集与设置:** 在 ARC-AGI-1 基准上评估，该基准专注于流体智能和抽象推理，包含像素网格变换谜题；实验使用 OpenAI 的 o4-mini 作为主要推理模型，GPT-4.1 辅助概念抽象和选择；对比了无记忆基线、动态备忘录 (cheatsheet)、ArcMemo-OE 和 ArcMemo-PS 四种设置。
*   **主要结果:** ArcMemo-PS 在标准推理计算下取得最佳官方得分（59.33，相对基线 55.17 提升 7.5%），且在所有推理规模下持续优于基线；随着重试次数增加（0到2次），性能进一步提升至 70.83，显示出良好的计算扩展性。
*   **消融分析:** 去除选择机制后性能下降（例如 ArcMemo-PS 在 oracle@2 从 59.33 降至 55.17），且 token 使用量增加，表明选择机制对性能和效率至关重要。
*   **持续学习效果:** 测试时动态更新记忆（每10个问题更新一次）在高重试深度下提升性能（例如 ArcMemo-OE 在2次重试时从 67.67 提升至 70.00），验证了持续学习的有效性。
*   **合理性与局限:** 实验设置全面，考虑了不同推理规模和多次运行以减少采样方差；但受限于评估子集（100个谜题）和基线已解决大部分问题，潜在提升空间可能被低估；此外，记忆更新对问题顺序敏感，可能影响结果一致性。

## Further Thoughts

论文中抽象概念级记忆的设计令人印象深刻，通过从具体问题中解耦通用模式，为 LLMs 的跨任务迁移和持续学习开辟了新方向；程序合成式 (PS) 格式的参数化和模块化组合机制，类似于软件工程中的代码复用，可启发其他领域（如规划或代码生成）中结构化记忆系统的构建；此外，测试时持续学习和推理驱动的选择机制（利用长篇推理匹配抽象概念）也为构建更接近人类学习过程的终身学习系统提供了宝贵思路，特别是在复杂推理任务中可能比传统嵌入相似性检索更具潜力。