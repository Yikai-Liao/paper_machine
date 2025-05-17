---
title: "Are Large Language Models Robust in Understanding Code Against Semantics-Preserving Mutations?"
pubDatetime: 2025-05-15T16:04:25+00:00
slug: "2025-05-llm-code-robustness"
type: "arxiv"
id: "2505.10443"
score: 0.7228935665915713
author: "grok-3-latest"
authors: ["Pedro Orvalho", "Marta Kwiatkowska"]
tags: ["LLM", "Code Understanding", "Semantic Robustness", "Reasoning Quality", "Code Mutation"]
institution: ["Department of Computer Science, University of Oxford"]
description: "本文通过手动专家分析和语义保持不变的代码变异，揭示了大型语言模型在代码理解中推理质量不高且对语法变化缺乏鲁棒性的缺陷，为改进模型语义理解能力提供了方向。"
---

> **Summary:** 本文通过手动专家分析和语义保持不变的代码变异，揭示了大型语言模型在代码理解中推理质量不高且对语法变化缺乏鲁棒性的缺陷，为改进模型语义理解能力提供了方向。 

> **Keywords:** LLM, Code Understanding, Semantic Robustness, Reasoning Quality, Code Mutation

**Authors:** Pedro Orvalho, Marta Kwiatkowska

**Institution(s):** Department of Computer Science, University of Oxford


## Problem Background

大型语言模型（LLMs）在编程任务中被广泛应用，但其代码理解能力是否真正基于语义而非语法模式仍存疑问。
现有研究多关注预测准确性，忽视推理过程的逻辑合理性，而 LLMs 在数学推理中常通过错误逻辑得出正确答案，提示代码理解可能存在类似问题。
论文旨在评估 LLMs 是否能基于合理推理预测程序输出，并测试其对语义保持不变的代码变异（semantics-preserving mutations）的鲁棒性。

## Method

*   **核心思想：** 通过手动专家分析和语义保持不变的代码变异，评估 LLMs 在代码理解任务中的推理质量和语义鲁棒性。
*   **具体步骤：**
    *   **手动专家分析：** 在 LiveCodeBench 数据集上，人工评估 LLMs 的正确预测是否基于合理推理、错误推理或纯粹猜测，揭示模型推理过程的逻辑质量。
    *   **语义保持不变的代码变异：** 设计五种变异方式，包括变量重命名、比较表达式镜像、if-else 分支交换、for-to-while 循环转换和循环部分展开。这些变异不改变程序运行行为，但改变语法结构，用于测试 LLMs 是否依赖语法特征而非语义理解。
    *   **迭代查询策略：** 通过多次交互和反馈，观察 LLMs 是否能修正错误预测并改进推理逻辑，评估交互对模型性能的提升效果。
*   **评估对象：** 针对六个 LLMs（包括专门为编码任务训练的模型如 Qwen 2.5-Coder 和通用模型如 Llama 3.2），在 LiveCodeBench 和 CRUXEval 两个基准数据集上进行测试。
*   **关键创新：** 不仅关注预测结果，还深入分析推理过程，并通过变异测试揭示模型对语义理解的真实能力。

## Experiment

*   **推理质量：** Qwen 2.5-Coder 表现最佳，正确预测率达 62%，且基于错误推理的正确预测比例较低（12.79%）；Llama 3.2 正确预测率为 41%，但 61% 的正确预测基于错误推理，显示出较强的猜测倾向。
*   **交互查询效果：** 迭代查询显著提升部分模型性能，如 Granite Code 和 Llama 3.2 在多次反馈后基于合理推理的正确预测比例分别提升至 14.72% 和 14.22%。
*   **鲁棒性测试：** 在语义保持不变的代码变异下，模型预测波动明显，例如 Qwen 2.5-Coder 在 LiveCodeBench 上的正确率从 62% 提升至 93.1%（组合所有变异后），SemCoder 从 48% 提升至 84.6%，表明模型对语法变化敏感，缺乏语义稳定性。
*   **数据集差异：** 在 CRUXEval 上模型表现略微稳定，可能因数据集公开且部分模型训练中接触过类似数据，但整体仍显示出对语法变异的敏感性。
*   **实验设置评价：** 实验覆盖了不同类型模型和数据集，变异种类多样，设置较为全面合理；但未深入探讨变异组合的复杂影响，且限于 8B 参数以下模型，可能限制结论普适性。

## Further Thoughts

论文通过语义保持不变的变异测试 LLMs 鲁棒性，这一思路可扩展至其他领域，如自然语言处理中的同义句替换，测试模型对语义等价表达的理解能力；此外，交互查询提升推理质量的发现提示在实际应用中设计人机交互机制，可能有效弥补模型语义理解缺陷，尤其在代码调试或教育场景中。