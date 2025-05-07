---
title: "FormalMATH: Benchmarking Formal Mathematical Reasoning of Large Language Models"
pubDatetime: 2025-05-05T15:37:00+00:00
slug: "2025-05-formalth-benchmark"
type: "arxiv"
id: "2505.02735"
score: 0.5545446165758466
author: "grok-3-latest"
authors: ["Zhouliang Yu", "Ruotian Peng", "Keyi Ding", "Yizhe Li", "Zhongyuan Peng", "Minghao Liu", "Yifan Zhang", "Zheng Yuan", "Huajian Xin", "Wenhao Huang", "Yandong Wen", "Ge Zhang", "Weiyang Liu"]
tags: ["LLM", "Formal Reasoning", "Benchmark", "Autoformalization", "Semantic Verification"]
institution: ["The Chinese University of Hong Kong", "Numina", "Westlake University", "M-A-P", "2077AI", "University of California, Los Angeles", "Max Planck Institute for Intelligent Systems, Tübingen"]
description: "本文提出 FormalMATH，一个包含 5,560 个形式化数学问题的 Lean4 基准，通过人在回路的自动形式化流程降低构建成本，并揭示了当前 LLM 在形式化推理中的显著局限性。"
---

> **Summary:** 本文提出 FormalMATH，一个包含 5,560 个形式化数学问题的 Lean4 基准，通过人在回路的自动形式化流程降低构建成本，并揭示了当前 LLM 在形式化推理中的显著局限性。 

> **Keywords:** LLM, Formal Reasoning, Benchmark, Autoformalization, Semantic Verification

**Authors:** Zhouliang Yu, Ruotian Peng, Keyi Ding, Yizhe Li, Zhongyuan Peng, Minghao Liu, Yifan Zhang, Zheng Yuan, Huajian Xin, Wenhao Huang, Yandong Wen, Ge Zhang, Weiyang Liu

**Institution(s):** The Chinese University of Hong Kong, Numina, Westlake University, M-A-P, 2077AI, University of California, Los Angeles, Max Planck Institute for Intelligent Systems, Tübingen


## Problem Background

形式化数学推理（Formal Mathematical Reasoning, FMR）是人工智能领域的一个关键挑战，现有基准数据集如 MiniF2F 和 ProofNet 在范围、规模和性能饱和方面存在局限，难以全面评估大型语言模型（LLM）在多样化数学领域中的推理能力。
论文旨在构建一个大规模、覆盖广泛数学领域和难度级别的形式化数学基准，以揭示当前 LLM 的不足并推动其在形式化推理能力上的发展。

## Method

* **核心目标**：构建 FormalMATH，一个包含 5,560 个在 Lean4 系统中形式化验证的数学问题基准，覆盖从高中奥林匹克竞赛到本科水平的多种数学领域（如代数、微积分、离散数学等）。
* **数据构建流程**：
  * **多 LLM 自动形式化**：利用多个专门的 LLM（如 Qwen2.5-7B-Coder 和 DeepSeek-Prover-Base）通过最佳-N 采样（Best-of-N Sampling）策略生成多个形式化语句候选，确保多样性和质量。
  * **自动验证与过滤**：包括三层验证机制：首先通过 Lean4 编译器进行语法正确性检查；其次利用多个通用 LLM（如 o1-mini、Claude-3.5-Sonnet）进行语义验证，将形式化语句回译为自然语言并与原始问题比对，确保语义一致性；最后采用基于否定的反证策略，使用现成的 LLM 证明器尝试证明语句的否定形式，以过滤不可证明的语句。
  * **专家手动验证**：由 12 名奥林匹克级别的数学专家对经过自动过滤的语句进行最终语义一致性检查，保留了 72.09% 的语句，显著降低了人工标注成本。
* **关键创新**：通过多层次自动化流程减少人工负担，同时保证形式化语句的质量和语义准确性，为大规模形式化数据集的构建提供了一种高效、可扩展的解决方案。

## Experiment

* **挑战性验证**：在 FormalMATH 上评估了多种最先进的 LLM 形式化定理证明器（如 Kimina-Prover、BFS-Prover），结果显示即使最强模型在 Pass@32 指标下也仅达到 16.46% 的成功率，表明基准的高难度。
* **领域偏差**：模型在代数和应用数学领域表现较好（例如 Kimina-Prover 在代数领域的成功率较高），但在微积分和离散数学等领域表现较差，反映出训练数据分布不均和跨领域泛化能力的不足。
* **测试时扩展效果**：在 FormalMATH-Lite 子集上增加采样预算（如从 Pass@32 到 Pass@3200）带来的性能提升有限，例如 STP 模型仅提升了 4.58%，表明形式化推理中单一错误即可导致整个证明失败，限制了采样扩展的有效性。
* **CoT 策略影响**：朴素的链式思维（Naive CoT）比自然语言增强的 CoT（NL-Augmented CoT）表现更好，例如 DeepSeek-V1.5-SFT 在 Pass@3200 下分别达到 50.6% 和 49.2%，表明自然语言指导可能引入噪声，增加模型不确定性。
* **实验设置合理性**：实验覆盖了最佳优先树搜索（BFS）和单次生成（SPG）两种证明策略，并在 FormalMATH-Lite 上进行了测试时扩展分析，设置较为全面；但由于计算资源限制，部分高预算实验仅在子集上进行，可能影响结果的全面性。

## Further Thoughts

论文的多 LLM 协作语义验证和人在回路策略启发我们可以在其他高精度语义一致性任务（如法律文本或代码生成）中应用类似方法；领域偏差问题提示未来 LLM 训练需关注数据分布均衡或领域自适应技术；自然语言指导在 CoT 中的负面效应表明需设计更结构化的中间推理表示，避免噪声；测试时扩展的局限性则启发探索基于中间状态反馈的智能搜索策略或容错机制，以提升形式化推理的鲁棒性。