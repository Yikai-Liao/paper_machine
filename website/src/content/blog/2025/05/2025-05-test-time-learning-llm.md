---
title: "Test-Time Learning for Large Language Models"
pubDatetime: 2025-05-27T02:18:59+00:00
slug: "2025-05-test-time-learning-llm"
type: "arxiv"
id: "2505.20633"
score: 0.8319750922063133
author: "grok-3-latest"
authors: ["Jinwu Hu", "Zitian Zhang", "Guohao Chen", "Xutao Wen", "Chao Shuai", "Wei Luo", "Bin Xiao", "Yuanqing Li", "Mingkui Tan"]
tags: ["LLM", "Test Time Adaptation", "Self-Supervised Learning", "Parameter Efficiency", "Domain Adaptation"]
institution: ["South China University of Technology", "Pazhou Laboratory", "Zhejiang University", "South China Agricultural University", "Chongqing University of Posts and Telecommunications", "Key Laboratory of Big Data and Intelligent Robot"]
description: "本文提出测试时学习方法 TLM，通过输入困惑度最小化、样本高效策略和 LoRA 实现大型语言模型在分布偏移场景下的动态适配，显著提升性能并降低计算成本。"
---

> **Summary:** 本文提出测试时学习方法 TLM，通过输入困惑度最小化、样本高效策略和 LoRA 实现大型语言模型在分布偏移场景下的动态适配，显著提升性能并降低计算成本。 

> **Keywords:** LLM, Test Time Adaptation, Self-Supervised Learning, Parameter Efficiency, Domain Adaptation

**Authors:** Jinwu Hu, Zitian Zhang, Guohao Chen, Xutao Wen, Chao Shuai, Wei Luo, Bin Xiao, Yuanqing Li, Mingkui Tan

**Institution(s):** South China University of Technology, Pazhou Laboratory, Zhejiang University, South China Agricultural University, Chongqing University of Posts and Telecommunications, Key Laboratory of Big Data and Intelligent Robot


## Problem Background

大型语言模型（LLMs）在实际部署中面临分布偏移（distribution shifts）的问题，包括领域特定术语和语言多样性变异，导致模型性能下降。
论文旨在解决如何在测试时仅使用无标签测试数据动态适配模型到目标领域，同时避免灾难性遗忘和高计算开销。

## Method

*   **核心思想:** 提出测试时学习范式 TLM（Test-Time Learning for LLMs），通过自监督方式动态调整模型参数以适应分布偏移。
*   **输入困惑度最小化（Input Perplexity Minimization）:** 基于输入困惑度与输出困惑度的正相关性，将测试时学习目标定义为最小化输入困惑度，通过优化模型对输入数据的理解间接提升输出质量。这种方法无需标签数据，利用自监督学习目标调整模型对目标分布的适配能力。
*   **样本高效学习策略（Sample Efficient Learning Strategy）:** 观察到高困惑度样本对模型更新贡献更大，设计基于困惑度的加权方案，优先选择高困惑度样本进行反向传播，减少对低困惑度样本的计算开销，提高适配效率。
*   **低秩适配（Low-Rank Adaptation, LoRA）:** 为避免灾难性遗忘和降低计算成本，采用 LoRA 进行参数更新，仅调整模型参数的一小部分（通过低秩矩阵分解），实现轻量级训练并保留原始知识。
*   **实现细节:** 在测试时，模型根据输入困惑度计算样本权重，通过 LoRA 更新参数，优化过程不依赖训练数据或外部知识库，适用于动态环境。

## Experiment

*   **有效性:** TLM 在 AdaptEval 基准的多个数据集（DomainBench, InstructionBench, ReasoningBench）上显著优于原始 LLMs，特别是在领域知识适配任务中，性能提升至少 20%。例如，在 Geography 数据集上，TLM 相较于 Llama3.2-3B-Instruct 提升了 20.79%。
*   **对比优越性:** 相较于基线方法（如 Tent, EATA, COME），TLM 表现更优，尤其在领域适配和指令任务中。例如，在 Qwen2.5-7B-Instruct 的 Agriculture 数据集上，TLM 相较于 EATA 提升了 37.32%。
*   **实验设置合理性:** AdaptEval 基准覆盖了多种分布偏移场景，数据集选择和评估指标（如 Rouge-Lsum 和 Exact Match）全面，消融实验验证了各组件的有效性，例如输入困惑度最小化在 Medicine 数据集上带来 83.9% 的相对提升。
*   **计算开销:** 通过样本高效策略和 LoRA，TLM 显著降低计算成本，例如在线设置中反向传播次数减少了 69.7%。

## Further Thoughts

输入困惑度作为自监督目标的思路启发我们探索其他内在度量（如置信度或一致性）作为测试时学习的优化目标，尤其在跨模态场景中是否可结合多模态数据的困惑度设计适配方法；高困惑度样本优先策略提示是否可通过强化学习动态调整样本选择标准；LoRA 在测试时学习的成功应用启发我们尝试其他参数高效方法（如适配器或提示学习）以进一步优化适配效率。