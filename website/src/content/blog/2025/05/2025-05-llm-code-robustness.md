---
title: "Are Large Language Models Robust in Understanding Code Against Semantics-Preserving Mutations?"
pubDatetime: 2025-05-15T16:04:25+00:00
slug: "2025-05-llm-code-robustness"
type: "arxiv"
id: "2505.10443"
score: 0.7228935665915713
author: "grok-3-latest"
authors: ["Pedro Orvalho", "Marta Kwiatkowska"]
tags: ["LLM", "Code Understanding", "Semantic Robustness", "Reasoning", "Data Augmentation"]
institution: ["Department of Computer Science, University of Oxford"]
description: "本文通过推理质量分析和语义保持变异测试，揭示了大型语言模型在代码理解任务中缺乏真正的语义鲁棒性，强调了推理一致性和鲁棒性评估的重要性。"
---

> **Summary:** 本文通过推理质量分析和语义保持变异测试，揭示了大型语言模型在代码理解任务中缺乏真正的语义鲁棒性，强调了推理一致性和鲁棒性评估的重要性。 

> **Keywords:** LLM, Code Understanding, Semantic Robustness, Reasoning, Data Augmentation

**Authors:** Pedro Orvalho, Marta Kwiatkowska

**Institution(s):** Department of Computer Science, University of Oxford


## Problem Background

大型语言模型（LLMs）在编程任务中被广泛应用，但其是否真正理解代码语义仍存疑问。
现有研究多关注 LLMs 在代码输出预测任务上的准确性，忽视了其推理过程是否合理，以及面对语义保持不变但语法变化的代码变异时是否表现出鲁棒性。
论文旨在解决这一关键问题：LLMs 是否能基于语义理解代码，而非仅依赖语法模式进行预测？这一问题对 LLMs 在软件开发中的可靠性和可信度至关重要。

## Method

*   **核心目标：** 评估 LLMs 在代码理解任务中的推理能力和语义鲁棒性，揭示其是否真正理解代码而非依赖语法特征。
*   **推理质量评估：** 通过专家手动分析，在 LIVE CODE BENCH 数据集上对 LLMs 的输出预测进行分类，判断正确预测是否基于合理推理、错误推理或纯粹猜测，以揭示模型推理的逻辑基础。
*   **语义保持变异测试：** 设计并应用五种语义保持的代码变异，包括：
    *   变量重命名（Renaming Variables）：更改变量名但保持作用域一致性。
    *   比较表达式镜像（Mirroring Comparison Expressions）：交换比较操作数并使用逻辑等价的反向操作符。
    *   if-else 分支交换（Swapping If-Else Branches）：交换 if 和 else 块并对条件取反。
    *   for 循环转 while 循环（For-to-While Loop Conversion）：将 for 循环改写为等价的 while 循环，引入显式索引。
    *   循环部分展开（Partial Loop Unrolling）：提取循环最后几次迭代并在循环后顺序执行，调整循环条件。
  这些变异在 LIVE CODE BENCH 和 CRUX EVAL 数据集上测试，评估模型预测的一致性。
*   **交互式查询策略：** 对于预测错误的模型，提供反馈并允许多次迭代（最多五次或超时），观察是否能通过反馈改进推理和预测结果。
*   **评估对象：** 选择了六个参数规模不超过 8B 的开源 LLMs，包括五个编码专用模型（Code Gemma、Granite Code、Qwen 2.5-Coder、Mistral、SemCoder）和一个通用模型（Llama 3.2），以确保实验可在消费级硬件上运行。

## Experiment

*   **推理质量：** 在 LIVE CODE BENCH 上，Qwen 2.5-Coder 正确预测率最高（62%），SemCoder 在单次迭代中基于合理推理的正确率最高（84%），但 Llama 3.2 有 61% 的正确预测基于错误推理，显示出较强的猜测倾向。
*   **交互式查询效果：** 支持交互式查询的模型（如 Granite Code、Llama 3.2、Qwen 2.5-Coder）在反馈后基于合理推理的正确预测比例提升明显，例如 Granite Code 提升至 14.72%，表明反馈机制有助于改进推理。
*   **鲁棒性测试：** 在语义保持变异测试中，模型预测一致性较差，例如 Qwen 2.5-Coder 在 LIVE CODE BENCH 上正确率从 62% 提升至 93.1%（组合所有变异后），SemCoder 从 48% 提升至 84.6%，表明模型对语法变化敏感，缺乏语义鲁棒性；CRUX EVAL 上结果类似，但波动略小。
*   **实验设置合理性：** 实验覆盖多个模型、两个基准数据集和五种变异类型，并结合专家分析进行定性评估，设置较为全面；但局限性在于未对 CRUX EVAL 进行深入推理分析，且未测试更大规模模型的表现。

## Further Thoughts

论文通过语义保持变异测试模型鲁棒性的方法具有广泛适用性，可扩展至自然语言处理等领域，评估模型是否依赖表面特征而非深层语义；交互式查询对推理改进的潜力提示我们可以在模型部署中引入用户反馈或强化学习机制；此外，模型对语法变化的敏感性可能与训练数据多样性不足有关，未来可通过数据增强或对抗性训练提高泛化能力。