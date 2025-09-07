---
title: "Intermediate Languages Matter: Formal Languages and LLMs affect Neurosymbolic Reasoning"
pubDatetime: 2025-09-04T10:25:50+00:00
slug: "2025-09-intermediate-language-reasoning"
type: "arxiv"
id: "2509.04083"
score: 0.8061918508864002
author: "grok-3-latest"
authors: ["Alexander Beiser", "David Penz", "Nysret Musliu"]
tags: ["LLM", "Neurosymbolic Reasoning", "Formal Language", "Logical Reasoning", "In-Context Learning"]
institution: ["TU Wien", "Johannes Kepler University Linz"]
description: "本文通过实验验证了形式语言选择对神经符号推理性能的显著影响，证明一阶逻辑（FOL）在多种 LLMs 和数据集上表现最佳，为中间语言挑战提供了实证支持。"
---

> **Summary:** 本文通过实验验证了形式语言选择对神经符号推理性能的显著影响，证明一阶逻辑（FOL）在多种 LLMs 和数据集上表现最佳，为中间语言挑战提供了实证支持。 

> **Keywords:** LLM, Neurosymbolic Reasoning, Formal Language, Logical Reasoning, In-Context Learning

**Authors:** Alexander Beiser, David Penz, Nysret Musliu

**Institution(s):** TU Wien, Johannes Kepler University Linz


## Problem Background

大型语言模型（LLMs）在逻辑推理任务中存在抽象推理能力不足的问题，常导致错误结论。
神经符号推理（Neurosymbolic LLM Reasoning）通过将自然语言翻译为形式语言并结合符号推理器求解，试图提升推理可靠性，但其成功因素尚未明确。
本文聚焦于形式语言选择的影响，提出‘中间语言挑战’（Intermediate Language Challenge），即不同形式语言的语法和语义特性如何影响翻译和推理性能。

## Method

*   **核心思想：** 验证形式语言选择对神经符号推理性能的影响，通过比较不同形式语言在 LLMs 上的表现，探索中间语言挑战的解决方案。
*   **具体步骤：**
    *   **框架设计：** 采用神经符号推理框架，将自然语言问题通过 LLMs 的上下文学习（In-Context Learning, ICL）翻译为形式语言，再由符号推理器求解。
    *   **形式语言选择：** 选取四种形式语言进行对比，包括逻辑编程语言 Pyke 和 ASP，以及一阶逻辑相关语言 NLTK 和 FOL，每种语言具有不同的语法和语义特性。
    *   **提示策略：** 为每种形式语言设计一致的提示风格，提供 ICL 指令和示例，确保翻译过程的可比性，避免提示差异对结果的干扰。
    *   **符号求解：** 使用特定符号推理器处理翻译后的形式语言表示，例如 Clingo 用于 ASP，Prover9 用于 NLTK 和 FOL，确保求解过程的准确性。
    *   **控制变量：** 将 LLMs 的温度设为 0 以实现近似确定性输出，限制最大输出 token 数（通常为 2048，部分模型如 DeepSeek-R1 设为 20480），避免使用备份策略，聚焦形式语言本身的影响。
*   **创新点：** 通过系统性实验，首次量化了形式语言选择对推理性能的影响，为中间语言挑战提供了实证支持。

## Experiment

*   **有效性：** 实验结果表明形式语言选择显著影响推理性能，一阶逻辑（FOL）整体准确率最高（65.29%），优于 NLTK（60.36%）、ASP（53.94%）和 Pyke（45.83%），FOL 在执行率和执行准确率上均表现突出。
*   **模型差异：** 不同 LLMs 对形式语言的适应性差异较大，FOL 在小型模型（如 GPT-4o-mini，平均 72.85%）上表现最佳，而 ASP 在 DeepSeek-32B 和 DeepSeek-R1 (20480 token) 上表现较强（最高 88.82%）；小型模型性能波动大，但提升潜力显著。
*   **实验设置：** 实验覆盖三个逻辑推理数据集（ProntoQA、ProofWriter、FOLIO），七个 LLMs（8B 至 671B 参数），以及四种形式语言，设置全面合理，考虑了不同推理深度和世界假设（闭世界与开世界）。
*   **对比基线：** 相较标准提示和思维链（CoT）提示，神经符号方法在大多数情况下表现更优，尤其在小型模型上提升明显。
*   **局限性：** 部分结果因标准误差（SEM）重叠而不够确定，DeepSeek-R1 等模型需要更多输出 token（20480）以避免截断，增加了计算成本。

## Further Thoughts

形式语言的语法特性对 LLMs 翻译能力的影响值得深入探索，例如 ASP 的简洁语法在特定模型上表现出的 token 效率优势，是否可以通过设计混合形式语言（如结合 FOL 的语义能力和 ASP 的语法简洁性）进一步提升性能？
小型模型在神经符号推理中的潜力巨大，是否可以通过针对性微调或定制化形式语言数据增强其推理能力，从而在资源受限场景下实现高效推理？
训练数据中形式语言分布可能影响模型性能，是否可以通过分析 LLMs 训练语料，针对性选择或设计形式语言以匹配模型的‘语言偏好’？