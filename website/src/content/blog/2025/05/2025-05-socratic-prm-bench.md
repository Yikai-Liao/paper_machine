---
title: "Socratic-PRMBench: Benchmarking Process Reward Models with Systematic Reasoning Patterns"
pubDatetime: 2025-05-29T14:26:53+00:00
slug: "2025-05-socratic-prm-bench"
type: "arxiv"
id: "2505.23474"
score: 0.567807667812798
author: "grok-3-latest"
authors: ["Xiang Li", "Haiyang Yu", "Xinghua Zhang", "Ziyang Huang", "Shizhu He", "Kang Liu", "Jun Zhao", "Fei Huang", "Yongbin Li"]
tags: ["LLM", "Reasoning Patterns", "Process Reward", "Benchmarking", "Error Detection"]
institution: ["Institute of Automation, Chinese Academy of Sciences", "School of Artificial Intelligence, University of Chinese Academy of Sciences", "Tongyi Lab, Alibaba Group"]
description: "本文提出 SOCRATIC-PRMBENCH 基准，从推理模式视角系统性评估过程奖励模型（PRMs），揭示其在不同推理模式下的性能不足，为未来 PRM 开发提供重要参考。"
---

> **Summary:** 本文提出 SOCRATIC-PRMBENCH 基准，从推理模式视角系统性评估过程奖励模型（PRMs），揭示其在不同推理模式下的性能不足，为未来 PRM 开发提供重要参考。 

> **Keywords:** LLM, Reasoning Patterns, Process Reward, Benchmarking, Error Detection

**Authors:** Xiang Li, Haiyang Yu, Xinghua Zhang, Ziyang Huang, Shizhu He, Kang Liu, Jun Zhao, Fei Huang, Yongbin Li

**Institution(s):** Institute of Automation, Chinese Academy of Sciences, School of Artificial Intelligence, University of Chinese Academy of Sciences, Tongyi Lab, Alibaba Group


## Problem Background

大型语言模型（LLMs）在复杂推理和长距离决策任务中表现出强大能力，而过程奖励模型（PRMs）通过为推理过程中的每一步提供奖励信号，起到关键的指导作用。
然而，LLMs 在推理时会采用多种推理模式（如分解、演绎等），这些模式下可能出现不同类型的错误，现有 PRM 评估基准主要关注步级别的正确性，缺乏对不同推理模式下错误检测能力的系统性评估，导致无法全面揭示 PRMs 的局限性。

## Method

*   **核心思想：** 构建一个系统性基准 SOCRATIC-PRMBENCH，从推理模式视角评估 PRMs 的错误检测能力，覆盖六种推理模式（Transformation, Decomposition, Regather, Deduction, Verification, Integration）及 20 种细粒度错误类型。
*   **推理模式设计：** 基于苏格拉底逻辑理论，定义六种推理模式，并为每种模式设计特定错误类型（如 Transformation Inconsistency, Decomposition Unsoundness），确保评估的全面性和细粒度。
*   **数据生成与模型训练：** 首先通过 GPT-4o 将现有数学数据集（如 MATH-Hard, Open-o1）的链式推理（CoT）标注转化为苏格拉底推理过程，然后微调 Qwen2.5-72B-Instruct 模型生成新的推理路径，构建包含正确推理过程的数据池。
*   **测试用例构建：** 从正确推理数据池中随机抽取样本，利用 GPT-4o 控制性地注入特定类型的错误，生成 2995 个包含错误的推理路径，确保每个测试用例对应特定的推理模式和错误类型。
*   **质量控制：** 采用规则过滤（如格式检查）和基于 LLM 的过滤（如 Gemini2.5-Pro 评估错误合理性），并通过与人类标注者 93.3% 的一致性验证，确保数据质量和可靠性。
*   **评估方法：** 使用 PRM-Score 作为指标（综合 F1 分数和负 F1 分数），在多个开源 PRMs 和作为评论模型的 LLMs 上进行测试，分析模型在不同推理模式下的性能差异。

## Experiment

*   **有效性：** 实验结果显示现有 PRMs 性能不足，最高得分（Qwen2.5-Math-PRM-7B）仅为 68.0，而 LLMs（如 o3-mini）达到 75.7，表明 PRMs 在推理模式下的错误检测能力亟需提升。
*   **模式差异：** 模型在 Transformation, Decomposition, Regather 模式上的表现明显弱于 Deduction, Integration, Verification 模式，例如 Qwen2.5-Math-PRM-7B 在 Integration 上接近 80.0，但在 Decomposition 上不足 60.0，反映出训练数据中推理模式分布不均的问题。
*   **深入分析：** 实验发现模型存在错误检测延迟（预测错误位置晚于实际位置）和奖励偏见（倾向于正向或负向奖励），如 Qwen2.5-Math-PRM-7B 对正确步骤的准确率为 90.8%，但对错误步骤仅为 42.9%。
*   **合理性与局限：** 实验设置覆盖了六种推理模式和多种模型，设计合理且数据支持结论，但主要聚焦数学推理任务，论文也提到未来需扩展到其他领域（如文学、法律）以验证普适性。

## Further Thoughts

推理模式的细化设计为模型评估提供了新视角，未来可探索更多跨领域推理模式以提升 PRMs 泛化能力；自动化数据生成与质量控制的结合（如 LLM 注入错误与人工验证）可推广到其他评估任务；此外，LLMs 作为评论模型的优越性能提示我们，是否可以通过知识蒸馏或迁移学习，将 LLMs 的推理能力注入 PRMs，值得进一步研究。