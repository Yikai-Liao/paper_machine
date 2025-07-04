---
title: "Rethinking the Illusion of Thinking"
pubDatetime: 2025-07-01T23:10:02+00:00
slug: "2025-07-rethinking-illusion-reasoning"
type: "arxiv"
id: "2507.01231"
score: 0.5498102173589046
author: "grok-3-latest"
authors: ["Iñaki Dellibarda Varela", "Pablo Romero-Sorozabal", "Eduardo Rocon", "Manuel Cebrian"]
tags: ["LLM", "Reasoning", "Benchmark Design", "Multi-Agent Systems", "Task Complexity"]
institution: ["Center for Automation and Robotics, Spanish National Research Council (CSIC-UPM), Madrid, Spain"]
description: "本文通过重新设计实验和引入逐步求解、代理对话及配置修正方法，揭示了大型推理模型（LRMs）推理能力的复杂性，强调任务设计对评估结果的关键影响。"
---

> **Summary:** 本文通过重新设计实验和引入逐步求解、代理对话及配置修正方法，揭示了大型推理模型（LRMs）推理能力的复杂性，强调任务设计对评估结果的关键影响。 

> **Keywords:** LLM, Reasoning, Benchmark Design, Multi-Agent Systems, Task Complexity

**Authors:** Iñaki Dellibarda Varela, Pablo Romero-Sorozabal, Eduardo Rocon, Manuel Cebrian

**Institution(s):** Center for Automation and Robotics, Spanish National Research Council (CSIC-UPM), Madrid, Spain


## Problem Background

本文针对 Apple 发表的《The Illusion of Thinking》引发的争议，旨在澄清大型推理模型（LRMs）是否真正缺乏推理能力，还是受到实验设计、任务配置或输出限制的影响。
这一争论源于原始研究中 LRMs 在复杂任务（如 Towers of Hanoi 和 River Crossing）上的失败表现，被批评者视为缺乏推理能力的证据，而辩护者认为实验设计存在缺陷，结论过于绝对。

## Method

*   **逐步求解（Stepwise Resolution）**：针对 Towers of Hanoi 任务，将复杂问题分解为多个子阶段，每次提示模型仅生成固定数量的步骤（p 步），从当前配置开始逐步推进，旨在测试输出长度限制是否是性能瓶颈，而非推理能力不足。通过这种方式，减轻单次输出的负担，观察模型在较短推理范围内的表现。
*   **代理协作对话（Agentic Collaborative Dialogue）**：同样针对 Towers of Hanoi，设计两个 LRM 代理通过对话协作解决问题，每个代理基于共享记忆和对方提供的最新状态，轮流提出下一步的 p 个行动，探索多代理系统是否能通过交互和迭代改进提升长程规划和符号推理能力。
*   **任务配置修正（Configuration Viability）**：针对 River Crossing 任务，修正原始实验中包含数学上不可解配置的缺陷，仅测试符合可解条件的实例（如 boat capacity k ≥ 4 或 N ≤ 2k-1），以确保评估结果反映模型的真实推理能力，而非任务设计问题导致的失败。

## Experiment

*   **Towers of Hanoi 结果**：逐步求解方法未显著提升成功率，当磁盘数达到 8 个时模型仍一致失败，表明复杂性而非输出限制是主要障碍；代理对话方法表现更差，成功率在 N=4 时即下降，但 token 使用量持续高企，显示模型未放弃任务而是陷入无效循环。实验覆盖了 N=3 到 10 的多种配置，并通过 10 次独立试验确保结果稳健。
*   **River Crossing 结果**：修正配置后，LRMs 在可解实例上表现优异，甚至能处理涉及 100 个 agent pairs 的大型实例，成功率高且 token 使用量稳定；但特定中间配置（如 N=5, k=3）因解空间受限导致难度峰值，成功率最低。实验设置合理，覆盖多种可解配置，关注成功率与资源消耗的权衡。
*   **总体评估**：方法改进在 River Crossing 上效果显著，表明任务设计对评估结果影响巨大；但在 Towers of Hanoi 上未见明显提升，显示 LRMs 在长程规划和符号推理上的内在局限性依然存在。

## Further Thoughts

任务设计的合理性对 AI 评估至关重要，River Crossing 实验表明不合理的配置可能导致对模型能力的误判，未来基准测试应注重逻辑可行性；此外，多代理协作虽在当前实验中表现不佳，但其持续努力的特性提示通过优化交互机制可能提升长程推理能力；最后，任务难度的非线性分布（如中间配置的‘相变’区域）启发我们关注解空间结构，而非单纯任务规模。