---
title: "Enigmata: Scaling Logical Reasoning in Large Language Models with Synthetic Verifiable Puzzles"
pubDatetime: 2025-05-26T12:40:31+00:00
slug: "2025-05-enigmata-logical-reasoning"
type: "arxiv"
id: "2505.19914"
score: 0.6579586336015478
author: "grok-3-latest"
authors: ["Jiangjie Chen", "Qianyu He", "Siyu Yuan", "Aili Chen", "Zhicheng Cai", "Weinan Dai", "Hongli Yu", "Qiying Yu", "Xuefeng Li", "Jiaze Chen", "Hao Zhou", "Mingxuan Wang"]
tags: ["LLM", "Logical Reasoning", "Synthetic Data", "Reinforcement Learning", "Generalization"]
institution: ["ByteDance Seed", "Fudan University", "Institute for AI Industry Research (AIR), Tsinghua University", "Nanjing University", "Shanghai Jiao Tong University", "SIA-Lab of Tsinghua AIR and ByteDance Seed"]
description: "本文提出 Enigmata 套件，通过可扩展的合成谜题数据和 RLVR 训练框架，显著提升大型语言模型的逻辑推理能力，并展示出跨领域的泛化潜力。"
---

> **Summary:** 本文提出 Enigmata 套件，通过可扩展的合成谜题数据和 RLVR 训练框架，显著提升大型语言模型的逻辑推理能力，并展示出跨领域的泛化潜力。 

> **Keywords:** LLM, Logical Reasoning, Synthetic Data, Reinforcement Learning, Generalization

**Authors:** Jiangjie Chen, Qianyu He, Siyu Yuan, Aili Chen, Zhicheng Cai, Weinan Dai, Hongli Yu, Qiying Yu, Xuefeng Li, Jiaze Chen, Hao Zhou, Mingxuan Wang

**Institution(s):** ByteDance Seed, Fudan University, Institute for AI Industry Research (AIR), Tsinghua University, Nanjing University, Shanghai Jiao Tong University, SIA-Lab of Tsinghua AIR and ByteDance Seed


## Problem Background

大型语言模型（LLMs）在数学、STEM 和编程等复杂推理任务上通过可验证奖励的强化学习（RLVR）取得了显著进展，但现有模型在纯逻辑推理的谜题任务上表现不佳，这些任务不需要领域知识，却对人类来说往往简单直观。
当前谜题数据集缺乏多样性和可扩展性，且缺乏针对现代 LLMs 的训练方法和资源，论文旨在解决如何提升 LLMs 逻辑推理能力并实现跨任务泛化的问题。

## Method

*   **核心思想:** 提出 Enigmata 套件，通过一个全面、可控、可扩展的谜题数据集和训练框架，提升 LLMs 的逻辑推理能力。
*   **Enigmata-Data:** 包含 36 个任务，覆盖 7 大类逻辑推理谜题（如 Crypto Puzzle、Arithmetic Puzzle、Logic Puzzle 等），每个任务配备自动生成器和验证器，支持无限样本生成和难度控制，适配 RLVR 框架。
*   **Enigmata-Eval:** 一个严格的评估基准，包含 4758 个谜题实例，覆盖不同难度，用于测试模型逻辑推理能力。
*   **Enigmata-Model 训练方案:** 采用两阶段方法：
    *   **阶段 1 - 拒绝微调（Rejection Fine-Tuning, RFT）:** 通过监督微调结合高质量谜题和数学问题解决方案，建立基础推理模式，确保训练稳定性。
    *   **阶段 2 - 多任务强化学习（RL）:** 使用 VC-PPO 算法（一种 PPO 变体），通过自动验证器提供奖励信号，优化模型推理能力；探索两种多任务策略：
        *   **混合训练（Mix-Training RL）:** 同时训练多个谜题类型和数学任务，促进泛化能力。
        *   **多阶段训练（Multi-Stage RL）:** 采用课程学习方式，先训练核心技能，再逐步引入复杂任务，避免遗忘。
*   **关键细节:** 数据生成和验证完全自动化，支持长链推理（Chain-of-Thought）训练；通过控制数据量和难度分布，优化多任务平衡和性能。

## Experiment

*   **有效性:** 基于 Qwen2.5-32B 训练的 Qwen2.5-32B-Enigmata 在 Enigmata-Eval 上得分 62.6%，显著优于 o1（54.9%）和 DeepSeek-R1（49.2%）；在 ARC-AGI 1 上从基础模型的 6.0% 提升至 32.8%，超越 o1（29.0%）。
*   **泛化性:** 模型在域外（OOD）谜题任务（如 KOR-Bench）和数学推理任务（如 AIME）上表现出色；在大规模模型 Seed1.5-Thinking（20B/200B）上，Enigmata 数据进一步提升了 AIME 和 GPQA Diamond 等任务的表现（提升 0.4%-1.9%），显示出跨领域泛化潜力。
*   **实验设置合理性:** 评估覆盖多个基准（Enigmata-Eval、ARC-AGI、KOR-Bench、AIME），对比了多种 SoTA 模型；消融研究分析了训练数据量、难度分布和多任务策略的影响，设置全面。
*   **局限性:** 部分任务（如空间和序列谜题）仍具挑战性，代码利用未有效提升性能，显示出未来优化的方向。

## Further Thoughts

Enigmata 的合成谜题数据不仅提升了逻辑推理能力，还在大规模模型上意外改善了数学和 STEM 推理任务的表现，这种跨领域泛化效应提示我们，逻辑推理可能是通用推理能力的核心，未来可以探索更多类型的合成数据（如代码、知识推理）对不同任务的影响；此外，生成器-验证器设计为 RLVR 提供了可扩展资源，这种自动化框架是否能推广到其他领域也值得深入研究。