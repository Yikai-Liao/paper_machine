---
title: "Saturation-Driven Dataset Generation for LLM Mathematical Reasoning in the TPTP Ecosystem"
pubDatetime: 2025-09-08T15:43:29+00:00
slug: "2025-09-saturation-reasoning-data"
type: "arxiv"
id: "2509.06809"
score: 0.6567350772462135
author: "grok-3-latest"
authors: ["Valentin Quesnel", "Damien Sileo"]
tags: ["LLM", "Mathematical Reasoning", "Synthetic Data", "Automated Theorem Proving", "Task Decomposition"]
institution: ["Univ. Lille", "Inria", "CNRS", "Centrale Lille", "UMR 9189 - CRIStAL"]
description: "本文提出了一种基于符号化饱和的框架，利用 TPTP 公理库和自动化定理证明工具生成大规模、逻辑正确的数学推理任务数据集，通过三种难度可控的任务评估 LLMs 的逻辑推理能力，揭示其在多步骤结构推理上的局限性。"
---

> **Summary:** 本文提出了一种基于符号化饱和的框架，利用 TPTP 公理库和自动化定理证明工具生成大规模、逻辑正确的数学推理任务数据集，通过三种难度可控的任务评估 LLMs 的逻辑推理能力，揭示其在多步骤结构推理上的局限性。 

> **Keywords:** LLM, Mathematical Reasoning, Synthetic Data, Automated Theorem Proving, Task Decomposition

**Authors:** Valentin Quesnel, Damien Sileo

**Institution(s):** Univ. Lille, Inria, CNRS, Centrale Lille, UMR 9189 - CRIStAL


## Problem Background

大型语言模型（LLMs）在数学推理，特别是多步骤逻辑推导方面的能力较弱，主要原因是高质量、逻辑严谨的训练数据稀缺。
传统数学证明数据集规模小、创建成本高，且依赖专业人力，无法大规模众包，而现有合成数据方法（如基于 LLM 的自动形式化）可能引入事实错误，或生成缺乏数学相关性的随机逻辑公式。
论文旨在解决如何生成大规模、逻辑正确且具有数学意义的训练数据，以提升 LLMs 的数学推理能力。

## Method

*   **核心思想:** 利用自动化定理证明（ATP）工具的符号化推理能力，从 TPTP 公理库中系统性生成逻辑正确的定理语料库，并将其转化为难度可控的推理任务，用于训练和评估 LLMs 的数学推理能力。
*   **具体步骤:**
    *   **饱和生成:** 使用 E-prover 在 TPTP 公理库上运行饱和模式，通过穷尽应用推理规则（如叠加和参数调制），生成公理的逻辑推导结果，形成一个有向无环图（DAG），记录每个推导步骤及其依赖关系。
    *   **兴趣过滤:** 借助 AGInTRater 系统对生成的定理进行评分，基于复杂性、意外性（surprisingness）和有用性（usefulness）等指标，筛选出数学上‘有趣’的定理，避免琐碎或冗余结果。
    *   **任务构建:** 将筛选后的推导图转化为三种推理任务：
        - **猜想蕴含验证（Conjecture Entailment Verification）:** 测试模型是否能正确判断一组前提是否蕴含某个定理（真/假任务）。
        - **最小前提选择（Minimal Premise Selection）:** 测试模型从包含干扰项的前提池中识别证明定理所需的最小必要前提集。
        - **证明图重构（Proof Graph Reconstruction）:** 测试模型是否能从打乱的推导步骤中重构证明的全局结构。
    *   **难度控制:** 通过调整证明深度（proof depth）和干扰项数量（distractors/perturbations）等参数，控制任务复杂性。
    *   **逻辑验证:** 使用 Vampire 定理证明器作为基准，确保生成任务的逻辑正确性。
*   **优势:** 完全符号化的方法避免了 LLM 生成数据可能引入的错误，生成的定理和任务在逻辑上严谨，且框架具有可扩展性，可按需生成无限任务。

## Experiment

*   **有效性:** 在 3000 个问题的基准测试中，随着证明深度和干扰项数量增加，所有模型（gpt-5-nano、gpt-5-mini、gpt-5）的性能显著下降，表明框架能有效区分推理任务难度，并揭示 LLMs 在多步骤推理上的弱点。
*   **任务差异:** 证明图重构任务对所有模型最具挑战性，尤其是小规模模型性能几乎崩溃，表明 LLMs 在全局结构推理上存在根本缺陷；蕴含验证和前提选择任务相对较易，但高难度设置下性能仍下降明显。
*   **模型规模影响:** 较大规模模型（如 gpt-5）在高难度任务上表现更好，但在结构推理任务上仍不理想，说明单纯扩大模型规模无法完全解决深层推理问题。
*   **实验设置合理性:** 实验覆盖五个 TPTP 领域（代数、几何等），任务设计细致（通过证明深度和干扰项控制难度），使用 Vampire 验证逻辑正确性，确保结果严谨；但实验规模（3000 个问题）较小，仅作为概念验证，未来需更大规模测试以确认普适性。

## Further Thoughts

符号化数据生成的思路非常值得关注，完全依赖自动化定理证明工具生成逻辑正确的数据，避免了 LLM 可能引入的错误，是否可以扩展到其他领域（如自然语言推理或法律推理）？
任务分解为三种互补推理任务（蕴含验证、前提选择、证明重构）提供了细粒度评估工具，这种分解是否适用于其他复杂认知任务？
此外，论文提到的迭代饱和（将‘有趣’定理作为新公理输入下一轮生成）是否会带来逻辑结构的‘深度爆炸’，从而生成超出现有模型能力范围的任务，探索 LLMs 的推理极限？