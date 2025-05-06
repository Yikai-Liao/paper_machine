---
title: "FormalMATH: Benchmarking Formal Mathematical Reasoning of Large Language Models"
pubDatetime: 2025-05-05T15:37:00+00:00
slug: "2025-05-formalth-benchmark"
type: "arxiv"
id: "2505.02735"
score: 0.5545446165758466
author: "grok-3-latest"
authors: ["Zhouliang Yu", "Ruotian Peng", "Keyi Ding", "Yizhe Li", "Zhongyuan Peng", "Minghao Liu", "Yifan Zhang", "Zheng Yuan", "Huajian Xin", "Wenhao Huang", "Yandong Wen", "Ge Zhang", "Weiyang Liu"]
tags: ["LLM", "Formal Reasoning", "Benchmarking", "Autoformalization", "Theorem Proving"]
institution: ["The Chinese University of Hong Kong", "Numina", "Westlake University", "M-A-P", "2077AI", "University of California, Los Angeles", "Max Planck Institute for Intelligent Systems, Tübingen"]
description: "本文提出 FormalMATH，一个包含5560个形式化数学问题的 Lean4 基准，通过人机协同自动形式化流程构建，并揭示了当前 LLM 定理证明器在成功率、领域泛化和推理策略上的显著局限性。"
---

> **Summary:** 本文提出 FormalMATH，一个包含5560个形式化数学问题的 Lean4 基准，通过人机协同自动形式化流程构建，并揭示了当前 LLM 定理证明器在成功率、领域泛化和推理策略上的显著局限性。 

> **Keywords:** LLM, Formal Reasoning, Benchmarking, Autoformalization, Theorem Proving

**Authors:** Zhouliang Yu, Ruotian Peng, Keyi Ding, Yizhe Li, Zhongyuan Peng, Minghao Liu, Yifan Zhang, Zheng Yuan, Huajian Xin, Wenhao Huang, Yandong Wen, Ge Zhang, Weiyang Liu

**Institution(s):** The Chinese University of Hong Kong, Numina, Westlake University, M-A-P, 2077AI, University of California, Los Angeles, Max Planck Institute for Intelligent Systems, Tübingen


## Problem Background

形式化数学推理（Formal Mathematical Reasoning, FMR）是人工智能领域的重要挑战，但现有基准测试（如 MiniF2F 和 ProofNet）在范围和规模上存在局限，数据集小、领域狭窄，且性能饱和问题严重，限制了对大型语言模型（LLM）在多样化数学领域中推理能力的全面评估。
为此，论文提出了 FormalMATH，一个包含5560个形式化验证问题的 Lean4 基准，覆盖从高中奥林匹克到本科水平的广泛数学领域，旨在解决现有基准的不足，提供更具挑战性和全面性的评估平台。

## Method

*   **核心目标：** 构建一个大规模、多样化的形式化数学推理基准 FormalMATH，并评估 LLM 定理证明器的性能。
*   **构建流程：** 提出了一种人机协同的自动形式化管道，具体步骤包括：
    *   **多 LLM 自动形式化：** 使用多个专门的 LLM（如 Qwen2.5-7B-Coder 和 DeepSeek-Prover-Base），通过最佳-N 采样（Best-of-N Sampling）生成多个候选 Lean4 形式化语句，确保多样性和初步质量。
    *   **自动验证三层机制：** 首先通过 Lean4 编译器进行语法检查，确保语句格式正确；其次利用多 LLM 进行语义验证，将 Lean4 语句回译为自然语言并与原始问题比对，判断语义一致性；最后采用基于否定的反证过滤策略，通过证明语句的否定来剔除不可证明的语句。
    *   **专家手动验证：** 由12名国际数学奥林匹克奖牌获得者级别的专家进行最终审查，确保形式化语句与原始问题的语义完全对齐，保留72.09%的语句通过自动验证，显著降低手动成本。
*   **评估方法：** 在 FormalMATH 上测试多种最先进的 LLM 定理证明器（如 Kimina-Prover, BFS-Prover），分析其性能、领域偏差、测试时扩展效果及链式思维（CoT）策略的影响，使用 Pass@K 指标衡量证明成功率。
*   **关键创新：** 通过自动化与人工结合的形式化流程，解决了手动形式化成本高、现有工具不足的问题，同时 FormalMATH 的多样性和规模为评估提供了新标准。

## Experiment

*   **挑战性验证：** 即使最强的模型 Kimina-Prover 在 FormalMATH-Full 数据集上的成功率也仅为16.46%（Pass@32），BFS-Prover 仅为11.13%（Pass@1×32×100），远低于现有基准上的表现，表明 FormalMATH 的高难度和区分度。
*   **领域偏差：** 模型在代数和应用数学领域表现较好（如 Godel-Prover 在本科代数上达50%），但在微积分（5.21%）和离散数学（0%）等领域表现较差，揭示了跨领域泛化能力的不足。
*   **测试时扩展效果：** 在 FormalMATH-Lite 子集上，即使采样预算增加100倍，STP 模型性能提升仅4.58%（从48.59%到53.17%），远低于非形式化推理中的线性提升，反映了形式化证明中错误不容忍性导致的扩展效率低下。
*   **CoT 推理效果：** 朴素链式思维（Naive CoT）比自然语言增强 CoT（NL-Augmented CoT）表现更好（如 DeepSeek-V1.5-SFT 在 Pass@3200 下分别为50.6%和49.2%），表明自然语言指导可能引入噪声而非帮助。
*   **实验设置合理性：** 实验覆盖了多种证明策略（BFS 和单次生成 SPG）、多种模型、不同采样预算和推理模式，FormalMATH-Lite 通过平衡难度和领域分布设计，确保了测试时扩展分析的有效性和代表性。

## Further Thoughts

人机协同形式化流程通过多 LLM 自动验证和专家审查结合，显著降低成本，这种方法可推广至其他高精度标注领域；
领域偏差揭示了 LLM 训练数据分布不均的问题，未来可探索领域自适应机制或平衡数据集设计；
自然语言指导在形式化推理中的反直觉负面效果提示我们需重新思考提示设计，避免模糊性干扰精确推理；
测试时扩展的低效性启发我们是否可以通过中间状态验证或错误容忍机制改进证明搜索策略。