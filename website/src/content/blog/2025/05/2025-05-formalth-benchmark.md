---
title: "FormalMATH: Benchmarking Formal Mathematical Reasoning of Large Language Models"
pubDatetime: 2025-05-05T15:37:00+00:00
slug: "2025-05-formalth-benchmark"
type: "arxiv"
id: "2505.02735"
score: 0.5504693799303635
author: "grok-3-latest"
authors: ["Zhouliang Yu", "Ruotian Peng", "Keyi Ding", "Yizhe Li", "Zhongyuan Peng", "Minghao Liu", "Yifan Zhang", "Zheng Yuan", "Huajian Xin", "Wenhao Huang", "Yandong Wen", "Ge Zhang", "Weiyang Liu"]
tags: ["LLM", "Formal Reasoning", "Benchmarking", "Autoformalization", "Test Time Scaling"]
institution: ["The Chinese University of Hong Kong", "Numina", "Westlake University", "M-A-P", "2077AI", "University of California, Los Angeles", "Max Planck Institute for Intelligent Systems, Tübingen"]
description: "本文提出 FormalMATH，一个包含 5560 个形式化数学问题的 Lean4 基准测试，通过高效的‘人在回路中’自动化形式化流程构建，并揭示了当前大型语言模型在形式化推理中的显著局限性。"
---

> **Summary:** 本文提出 FormalMATH，一个包含 5560 个形式化数学问题的 Lean4 基准测试，通过高效的‘人在回路中’自动化形式化流程构建，并揭示了当前大型语言模型在形式化推理中的显著局限性。 

> **Keywords:** LLM, Formal Reasoning, Benchmarking, Autoformalization, Test Time Scaling

**Authors:** Zhouliang Yu, Ruotian Peng, Keyi Ding, Yizhe Li, Zhongyuan Peng, Minghao Liu, Yifan Zhang, Zheng Yuan, Huajian Xin, Wenhao Huang, Yandong Wen, Ge Zhang, Weiyang Liu

**Institution(s):** The Chinese University of Hong Kong, Numina, Westlake University, M-A-P, 2077AI, University of California, Los Angeles, Max Planck Institute for Intelligent Systems, Tübingen


## Problem Background

形式化数学推理（Formal Mathematical Reasoning, FMR）是人工智能领域的重要挑战，但现有基准测试在范围、规模和性能饱和方面存在局限，难以全面评估大型语言模型（LLMs）在多样化数学领域和难度级别上的推理能力。
论文旨在解决如何构建一个更大规模、更全面的基准测试，以揭示当前模型在形式化推理中的局限性，并推动更通用、更强大的 FMR 系统发展。

## Method

*   **核心目标：** 构建 FormalMATH，一个包含 5560 个经过形式化验证的数学问题的 Lean4 基准测试，覆盖从高中奥林匹克竞赛到本科水平的多个数学领域（如代数、几何、微积分、数论、离散数学等）。
*   **数据构建流程：** 提出一种‘人在回路中’（human-in-the-loop）的自动化形式化框架，以降低手动形式化的成本：
    *   **多 LLM 自动化形式化：** 使用多个专门的大型语言模型（如 Qwen2.5-7B-Coder 和 DeepSeek-Prover-Base），通过最佳-N 采样（Best-of-N Sampling）策略生成多个形式化语句候选。
    *   **自动化验证：** 包括三层验证机制：首先通过 Lean4 编译器进行语法正确性检查；其次利用多 LLM 进行语义验证，将 Lean4 语句回译为自然语言并与原始问题比对，确保语义一致性；最后采用基于否定的反证策略，尝试证明语句的否定形式以过滤不可证明的语句。
    *   **人工验证：** 由 12 名奥林匹克竞赛奖牌获得者级别的专家进行最终语义对齐检查，确保形式化语句与原始问题的保真度。
*   **效率提升：** 该流程保留了 72.09% 的语句进入人工验证阶段，大幅减少了专家标注的工作量，同时保证了数据集的高质量。
*   **评估方法：** 在 FormalMATH 上测试多种最先进的 LLM 形式化定理证明器（如 Kimina-Prover、BFS-Prover），分析其性能、领域偏见及测试时扩展效果。

## Experiment

*   **挑战性验证：** 即使最先进的 Kimina-Prover 在 Pass@32 指标下成功率仅为 16.46%，BFS-Prover 在 Pass@1×32×100 下仅为 11.13%，远低于现有基准（如 MiniF2F）上的表现，表明 FormalMATH 的高难度。
*   **领域偏见：** 模型在代数和应用数学领域表现较好（例如 Goedel-Prover 在本科代数上达到 50%），但在微积分（5.21%）和离散数学（0%）等领域表现较差，反映出训练数据分布不均导致的泛化能力不足。
*   **测试时扩展效果：** 在 FormalMATH-Lite 子集上增加采样预算（如从 Pass@32 到 Pass@3200）仅带来有限提升，例如 STP 模型从 48.59% 提升到 53.17%，远不如非形式化推理中的线性提升，表明形式化推理对错误的高度敏感性。
*   **CoT 策略分析：** 朴素链式思维（Naive CoT）比自然语言增强的 CoT（NL-Augmented CoT）表现更好，例如 DeepSeek-V1.5-SFT 在 Pass@3200 下分别为 50.6% 和 49.2%，表明自然语言指导可能引入噪声。
*   **实验设置合理性：** 实验覆盖了不同证明策略（最佳优先树搜索和单次生成）、不同采样预算及领域分布分析，设置较为全面，但也揭示了模型在复杂推理和跨领域泛化上的显著局限性。

## Further Thoughts

论文中‘人在回路中’的自动化形式化流程启发了我，这种多 LLM 协作与自动化验证结合的方式可以推广到其他高精度形式化任务（如法律文本或编程规范）；此外，领域偏见问题提示未来模型训练需更均衡的数据分布或领域自适应技术；CoT 的反直觉结果（自然语言指导降低性能）则启发我们设计更贴近形式化逻辑的推理引导策略，而非依赖直观描述；我还认为可以探索利用形式化推理中间状态（如子目标）作为反馈信号，以提升模型学习效率。