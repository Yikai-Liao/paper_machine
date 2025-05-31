---
title: "Revisiting Multi-Agent Debate as Test-Time Scaling: A Systematic Study of Conditional Effectiveness"
pubDatetime: 2025-05-29T01:02:55+00:00
slug: "2025-05-mad-test-time-scaling"
type: "arxiv"
id: "2505.22960"
score: 0.5533349451707595
author: "grok-3-latest"
authors: ["Yongjin Yang", "Euiin Yi", "Jongwoo Ko", "Kimin Lee", "Zhijing Jin", "Se-Young Yun"]
tags: ["LLM", "Multi-Agent Systems", "Test-Time Scaling", "Collaborative Refinement", "Diverse Exploration", "Reasoning", "Safety"]
institution: ["KAIST AI", "MPI for Intelligent Systems", "University of Toronto", "Vector Institute"]
description: "本文通过系统性实验揭示了多代理辩论（MAD）作为测试时扩展方法的条件有效性，特别是在数学推理中对高难度任务和低能力模型的帮助，以及在安全任务中通过异构代理降低攻击成功率的潜力。"
---

> **Summary:** 本文通过系统性实验揭示了多代理辩论（MAD）作为测试时扩展方法的条件有效性，特别是在数学推理中对高难度任务和低能力模型的帮助，以及在安全任务中通过异构代理降低攻击成功率的潜力。 

> **Keywords:** LLM, Multi-Agent Systems, Test-Time Scaling, Collaborative Refinement, Diverse Exploration, Reasoning, Safety

**Authors:** Yongjin Yang, Euiin Yi, Jongwoo Ko, Kimin Lee, Zhijing Jin, Se-Young Yun

**Institution(s):** KAIST AI, MPI for Intelligent Systems, University of Toronto, Vector Institute


## Problem Background

大型语言模型（LLM）的能力显著提升，促使研究者探索多代理系统，其中多代理辩论（MAD）框架被认为是一种通过代理间的协作、批评和精炼来增强问题解决能力的方法。
然而，MAD 相对于单代理测试时扩展方法（如自一致性和自精炼）的有效性尚未被系统性理解，尤其是在不同任务类型、难度、模型规模和代理多样性条件下的表现差异。
论文旨在回答：在什么条件下 MAD 能真正优于高能力的单代理方法？

## Method

*   **核心思想:** 将多代理辩论（MAD）概念化为一种测试时计算扩展技术，通过协作精炼和多样化探索两大特性，与单代理方法（如自一致性 Self-Consistency 和自精炼 Self-Refinement）进行系统性对比。
*   **具体实现:** 
    *   **协作精炼（Collaborative Refinement）:** 在每一轮辩论中，多个代理共享前一轮的所有上下文信息，共同精炼答案。这种方式不同于单代理自精炼仅依赖自身历史输出的迭代改进，MAD 强调代理间的上下文共享和联合优化。
    *   **多样化探索（Diverse Exploration）:** 特别是在异构设置中，通过使用不同模型（如 Qwen2.5, LLaMA3, Gemma2）或不同角色（Personas），在并行生成阶段探索更广泛的解空间，相比单代理自一致性仅通过单一模型多次采样实现的多样性。
    *   **实验设计:** 包括同构（Homogeneous）和异构（Heterogeneous）两种代理设置。同构设置中，所有代理基于同一模型；异构设置中，代理来自不同模型家族或具有不同角色。控制总生成次数（如 16 次）以确保公平对比，并针对数学推理和安全任务设计不同输出选择策略（如多数投票或裁判选择）。
*   **关键点:** MAD 不修改模型本身，仅通过测试时的代理交互实现性能提升，重点在于协作和多样性对不同任务的影响。

## Experiment

*   **数学推理效果:** MAD 在一般情况下未显著优于单代理并行扩展方法（如 Self-Consistency），但在任务难度较高（如 AIME）和模型能力较弱（如 Qwen2.5-3B）时，协作精炼展现相对优势，特别是在验证和共识达成方面，准确率提升明显（如 Qwen2.5-3B 在 AIME 上准确率从 8.9% 提升至 11.1%）。异构代理的多样化探索未带来显著收益，甚至因能力差异导致性能下降。
*   **安全推理效果:** 在同构设置下，MAD 的协作精炼增加攻击成功率（ASR），显示脆弱性，但增幅小于单代理自精炼（如 Qwen2.5-3B 在 Anthropic 数据集上 ASR 增幅较小）。异构设置通过多样化探索显著降低 ASR（如 Qwen2.5-7B 与 LLaMA3.1-8B 组合后 ASR 下降），表明不同模型的安全视角能相互补充。
*   **实验设置合理性:** 实验覆盖任务类型（数学推理、安全推理）、任务难度（GSM8K 到 AIME）、模型规模（1.5B 到 32B 参数）及代理多样性（同构 vs 异构）多个维度，数据全面且对比公平，但未探索更复杂 MAD 框架或更大规模实验。

## Further Thoughts

MAD 的多样化探索特性在安全任务中的显著效果启发我们思考：是否可以通过设计针对性更强的代理角色或模型组合，进一步优化特定任务表现？例如，在安全任务中引入专门训练的安全模型作为代理，是否能进一步降低攻击成功率？此外，MAD 作为测试时扩展方法的视角提示，未来可以探索动态调整代理数量或角色，以适应任务难度和模型能力变化，提升系统灵活性和鲁棒性。