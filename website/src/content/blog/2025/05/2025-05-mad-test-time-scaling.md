---
title: "Revisiting Multi-Agent Debate as Test-Time Scaling: A Systematic Study of Conditional Effectiveness"
pubDatetime: 2025-05-29T01:02:55+00:00
slug: "2025-05-mad-test-time-scaling"
type: "arxiv"
id: "2505.22960"
score: 0.5533349451707595
author: "grok-3-latest"
authors: ["Yongjin Yang", "Euiin Yi", "Jongwoo Ko", "Kimin Lee", "Zhijing Jin", "Se-Young Yun"]
tags: ["LLM", "Test Time Scaling", "Reasoning", "Sampling", "Collaborative Refinement"]
institution: ["KAIST AI", "MPI for Intelligent Systems", "University of Toronto", "Vector Institute"]
description: "本文系统研究了多智能体辩论（MAD）作为测试时计算扩展方法的条件有效性，揭示其协作改进和多样化探索特性在数学推理和安全任务中的不同表现，为未来 MAD 系统设计提供指导。"
---

> **Summary:** 本文系统研究了多智能体辩论（MAD）作为测试时计算扩展方法的条件有效性，揭示其协作改进和多样化探索特性在数学推理和安全任务中的不同表现，为未来 MAD 系统设计提供指导。 

> **Keywords:** LLM, Test Time Scaling, Reasoning, Sampling, Collaborative Refinement

**Authors:** Yongjin Yang, Euiin Yi, Jongwoo Ko, Kimin Lee, Zhijing Jin, Se-Young Yun

**Institution(s):** KAIST AI, MPI for Intelligent Systems, University of Toronto, Vector Institute


## Problem Background

随着大型语言模型（LLMs）能力的显著提升，多智能体系统特别是多智能体辩论（MAD）框架被认为是增强问题解决能力的潜在方法。
MAD 通过多个智能体协作提出、批评和改进论点，旨在提供比单一模型更强的推理能力、鲁棒性和多样化视角。
然而，现有研究缺乏对 MAD 效果的系统性理解，尤其是在不同条件下的表现与单一智能体方法的对比，因此本文试图填补这一空白，解决的关键问题是：在什么条件下 MAD 真正优于高能力的单一智能体方法？

## Method

*   **核心思想:** 将多智能体辩论（MAD）概念化为一种测试时计算扩展（Test-Time Scaling）技术，区别于单一智能体方法，通过协作改进和多样化探索提升性能。
*   **具体实现:** 
    *   **协作改进（Collaborative Refinement）**：在每一轮辩论中，智能体共享前一轮所有智能体的输出上下文，基于此进行联合改进，而非单一路径的迭代改进（如 Self-Refinement）。
    *   **多样化探索（Diverse Exploration）**：通过同质（Homogeneous）设置（使用相同模型的不同实例）和异质（Heterogeneous）设置（使用不同模型或不同角色配置），探索更广泛的解空间，区别于单一模型的并行采样（如 Self-Consistency）。
    *   **实验对比设计**：对比 MAD 与单一智能体扩展方法（如 Self-Consistency 和 Self-Refinement），并在任务类型（数学推理和安全任务）、任务难度、模型规模和智能体多样性等维度上评估 MAD 的表现。
*   **关键点:** MAD 不改变模型本身，而是在测试时通过多智能体交互动态调整输出，旨在利用协作和多样性提升推理或安全性能，同时控制计算成本。

## Experiment

*   **数学推理任务效果:** 在一般情况下，MAD 对单一智能体扩展方法（如 Self-Consistency）无明显优势，准确率提升有限；但在任务难度增加（如 AIME 数据集）和模型能力较低（如 Qwen2.5-3B）时，MAD 的协作改进特性表现出相对优势，准确率提升显著（例如 Qwen2.5-3B 在 AIME 上准确率从 8.9% 提升至 11.1%）。
*   **安全任务效果:** 在同质智能体设置下，MAD 的协作改进增加了攻击成功率（ASR），显示出脆弱性（例如 Qwen2.5-3B 在 Anthropic 数据集上 ASR 上升）；但在异质智能体设置下，多样化探索显著降低 ASR（例如 Qwen2.5-7B 与 LLaMA3.1-8B 组合后 ASR 下降），表明多样性在安全任务中更有效。
*   **实验设置合理性:** 实验覆盖了多种任务类型（数学推理：GSM8K, MATH500, AIME；安全：Anthropic Harmful Prompts, MultiJail）、模型规模（1.5B 到 32B 的 Qwen2.5 系列及 LLaMA3, Gemma2 等）和智能体配置（同质与异质），设置全面且对比公平（统一生成次数为 16），数据支持结论可信。
*   **局限性:** MAD 的提升并非普遍显著，更多在特定条件（如高难度任务或低能力模型）下显现，且计算成本因顺序生成而高于并行方法。

## Further Thoughts

MAD 的多样化探索在数学推理和安全任务中的表现差异启发我们：是否可以根据任务特性设计智能体配置？例如，在需要一致性答案的任务中减少多样性以避免性能下降，而在需要防御或创新的任务中增加多样性以提升鲁棒性。此外，MAD 作为测试时扩展的视角提示，未来的多智能体系统可能需要在计算资源分配和智能体角色设计上进行更精细的优化，例如动态调整智能体数量或角色以适应任务需求。