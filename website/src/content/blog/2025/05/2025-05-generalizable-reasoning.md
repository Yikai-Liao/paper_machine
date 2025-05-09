---
title: "X-Reasoner: Towards Generalizable Reasoning Across Modalities and Domains"
pubDatetime: 2025-05-06T21:08:27+00:00
slug: "2025-05-generalizable-reasoning"
type: "arxiv"
id: "2505.03981"
score: 0.48399096200296776
author: "grok-3-latest"
authors: ["Qianchu Liu", "Sheng Zhang", "Guanghui Qin", "Timothy Ossowski", "Yu Gu", "Ying Jin", "Sid Kiblawi", "Sam Preston", "Mu Wei", "Paul Vozila", "Tristan Naumann", "Hoifung Poon"]
tags: ["LLM", "Reasoning", "Multimodal", "Post-Training", "Generalization"]
institution: ["Microsoft Research"]
description: "本文提出 X-REASONER，通过仅基于通用领域文本的两阶段后训练策略（SFT + RL），成功实现推理能力跨模态和跨领域泛化，并在多个通用和医学基准测试中超越现有 SOTA。"
---

> **Summary:** 本文提出 X-REASONER，通过仅基于通用领域文本的两阶段后训练策略（SFT + RL），成功实现推理能力跨模态和跨领域泛化，并在多个通用和医学基准测试中超越现有 SOTA。 

> **Keywords:** LLM, Reasoning, Multimodal, Post-Training, Generalization

**Authors:** Qianchu Liu, Sheng Zhang, Guanghui Qin, Timothy Ossowski, Yu Gu, Ying Jin, Sid Kiblawi, Sam Preston, Mu Wei, Paul Vozila, Tristan Naumann, Hoifung Poon

**Institution(s):** Microsoft Research


## Problem Background

当前开源研究主要集中于文本推理模型的训练与评估，局限于数学和通用领域任务，而对于如何将推理能力扩展到多模态输入（如视觉-语言）和特定领域（如医学）仍缺乏深入探索。
论文提出一个核心问题：推理能力是否可以通过仅基于通用领域文本的后训练（post-training）实现跨模态和跨领域的泛化？
这一问题不仅具有科学意义（探究推理的本质是否独立于模态），还具有实际意义（文本数据易获取且计算成本低于多模态数据，避免了构建特定领域数据集的复杂性）。

## Method

*   **核心思想：** 通过仅基于通用领域文本数据的后训练，构建一个视觉-语言模型（Vision-Language Model, VLM），使其推理能力能够泛化到多模态和特定领域任务，而无需依赖多模态或领域特定数据。
*   **具体实现：** 采用两阶段训练策略：
    *   **第一阶段 - 监督微调（Supervised Fine-Tuning, SFT）：** 使用通用领域文本数据（如 OpenThoughts-114k，包含数学、科学推理任务），通过蒸馏的长链式推理（Long Chain-of-Thought, CoT）数据进行微调，旨在让模型学习结构化的推理模式，包括自我反思、验证和纠错等能力。训练过程中使用 AdamW 优化器，学习率为 1e-5，训练 4 个 epoch。
    *   **第二阶段 - 强化学习（Reinforcement Learning with Verifiable Rewards, RLVR）：** 在数学文本问题（如 Orz-math-57k）上进一步优化模型，采用 GRPO（Group Relative Policy Optimization）算法，通过直接使用任务准确率作为奖励信号（而非依赖奖励模型），避免奖励黑客问题。训练设置包括学习率 3e-6，采样 8 个响应，最大响应长度 4096 token。
    *   **辅助机制：** 引入强制退出机制（Forced-Exiting），通过在输出达到预定长度时添加停止标记（如 '</think>'），解决长 CoT 推理中模型无休止生成的问题。
*   **关键创新：** 完全依赖文本数据训练，探索推理能力的通用性，挑战传统上依赖多模态数据的训练范式，并通过数学任务作为推理泛化的‘锚点’来增强跨领域迁移能力。

## Experiment

*   **有效性：** X-REASONER 在仅使用通用领域文本训练的情况下，显著提升了多模态任务性能，例如在 MMMU-Pro 上从基线 38.3% 提升到 43.0%，在 MathVista 上从 62.8% 提升到 69.0%，超越了使用多模态数据训练的现有最优模型（SOTA）。
*   **泛化性：** 模型在医学领域的文本和多模态任务上表现出色，如在 MedQA (4-ops) 上从 71.6% 提升到 80.0%，表明推理能力可以跨模态和跨领域迁移。X-REASONER-MED（医学领域微调变体）进一步在医学任务上设置新 SOTA。
*   **合理性：** 实验通过消融研究验证了模型并非仅依赖文本捷径解决多模态任务（去掉文本可解样本后性能提升依然显著，如 MMMU-Pro 从 33.4% 提升到 36.4%），证明了真正的多模态推理能力。评估覆盖四种场景（通用文本、通用多模态、医学文本、医学多模态），数据全面，设置合理。
*   **开销与局限：** 训练计算开销较高（RL 阶段需 32 个 A100 GPU 训练 56 小时），模型规模限于 7B 参数，未探索更大规模或不同骨干网络的影响。

## Further Thoughts

论文提出的‘数学作为推理泛化锚点’的观点令人启发，数学任务因其结构化、长链式推理特性，可能比其他领域  领域更适合作为推理能力的训练基础，是否可以在其他领域寻找类似的‘锚点’任务（如逻辑推理或编程），以进一步提升泛化能力？
仅用文本数据训练即可实现多模态推理泛化挑战了传统观念，是否推理的核心是一种抽象的模式，与输入模态无关？这可能推动未来研究探索更通用的推理框架。
X-REASONER-MED 的成功表明，通用推理基础结合领域特定微调可能是实现专业化模型的高效路径，是否可以推广到其他领域（如法律、金融），通过少量领域数据结合通用推理模型快速构建专业模型？