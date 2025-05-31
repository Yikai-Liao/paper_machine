---
title: "Skywork Open Reasoner 1 Technical Report"
pubDatetime: 2025-05-28T12:56:04+00:00
slug: "2025-05-skywork-or1-reasoning"
type: "arxiv"
id: "2505.22312"
score: 0.7592791303909512
author: "grok-3-latest"
authors: ["Jujie He", "Jiacai Liu", "Chris Yuhao Liu", "Rui Yan", "Chaojie Wang", "Peng Cheng", "Xiaoyu Zhang", "Fuxiang Zhang", "Jiacheng Xu", "Wei Shen", "Siyuan Li", "Liang Zeng", "Tianwen Wei", "Cheng Cheng", "Bo An", "Yang Liu", "Yahui Zhou"]
tags: ["LLM", "Reinforcement Learning", "Chain of Thought", "Entropy Control", "Reasoning"]
institution: ["Skywork AI", "Kunlun Inc"]
description: "本文提出 Skywork-OR1，通过 MAGIC 框架为长 CoT 模型设计高效、可扩展的强化学习方案，显著提升数学和编码推理能力，并在 AIME24/25 上超越 DeepSeek-R1 和 Qwen3-32B，同时深入研究并缓解熵崩塌问题。"
---

> **Summary:** 本文提出 Skywork-OR1，通过 MAGIC 框架为长 CoT 模型设计高效、可扩展的强化学习方案，显著提升数学和编码推理能力，并在 AIME24/25 上超越 DeepSeek-R1 和 Qwen3-32B，同时深入研究并缓解熵崩塌问题。 

> **Keywords:** LLM, Reinforcement Learning, Chain of Thought, Entropy Control, Reasoning

**Authors:** Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, Siyuan Li, Liang Zeng, Tianwen Wei, Cheng Cheng, Bo An, Yang Liu, Yahui Zhou

**Institution(s):** Skywork AI, Kunlun Inc


## Problem Background

近年来，强化学习（RL）在提升大型语言模型（LLM）的推理能力（尤其是在数学和编码任务中）方面取得了显著成功，如 DeepSeek-R1 的突破。然而，现有 RL 方法多针对基础模型，而对于已经过监督微调（SFT）的长链式推理（Chain-of-Thought, CoT）模型，如何高效且可扩展地应用 RL 仍是一个挑战。长 CoT 模型生成的推理序列极长（例如在 AIME24 上超过 10K token），导致训练成本高、收敛慢和方差大。此外，探索与利用的平衡问题（如过早的策略熵崩塌）也影响训练效果。本文旨在设计一个针对长 CoT 模型的高效 RL 框架，解决上述问题并提升推理性能。

## Method

*   **核心框架：MAGIC**：提出了一种基于 Group Relative Policy Optimization (GRPO) 的改进框架，称为 MAGIC（Multi-stage Adaptive entropy scheduling for GRPO In Convergence），通过数据收集、训练策略和损失函数的优化来提升长 CoT 模型的推理能力。
*   **数据收集策略**：
    *   **严格数据预处理**：从 NuminaMath-1.5 和 LeetCode 等来源收集数学和编码数据，通过可验证性、正确性和挑战性标准进行筛选（例如排除全对或全错的问题），并结合人-LLM 联合评估确保数据质量。
    *   **离线与在线过滤**：训练前移除过于简单或困难的问题，训练中动态丢弃已解决的问题，确保模型始终面对挑战性数据。
    *   **拒绝采样（Rejection Sampling）**：过滤掉零优势组（即组内响应全对或全错），避免对训练的干扰，提高样本效率。
*   **训练策略**：
    *   **多阶段训练**：受 DeepScaleR 启发，逐步增加上下文长度（例如从 8K 到 32K），早期使用较短上下文降低计算成本，后期扩展以提升性能。
    *   **高温度采样**：设置采样温度为 1.0，增强模型探索能力，避免低温度导致的熵快速崩塌和组内响应多样性不足。
    *   **优势掩码策略（未采用）**：针对截断响应（因上下文长度限制无法生成完整答案）可能引入的噪声，测试了优势掩码策略，但实验表明不使用掩码也能保持后期性能提升并提高 token 效率。
    *   **在线策略训练**：优先采用在线策略更新（即每次训练步仅执行一次 SGD），减缓熵崩塌并提升测试性能。
*   **损失函数优化**：
    *   **去除 KL 损失**：发现 KL 损失会将策略拉回参考模型，阻碍后期性能提升，因此在训练中完全去除。
    *   **自适应熵控制**：引入目标熵（tgt-ent）和调整步长（∆）动态调整熵损失系数，防止过早熵崩塌，保持探索能力。
    *   **无长度归一化**：在策略损失中去除响应长度的归一化项，避免隐式长度偏差，损失在批次所有 token 上平均计算。
*   **关键创新**：MAGIC 框架通过上述策略平衡探索与利用，针对长 CoT 模型的高成本和训练不稳定性问题，提供了一个高效、可扩展的 RL 训练方案。

## Experiment

*   **性能提升显著**：基于 DeepSeek-R1-Distill 模型系列，Skywork-OR1-32B 在 AIME24 和 AIME25 上分别达到 82.2 和 73.3 的 avg@32 分数，超越 DeepSeek-R1 和 Qwen3-32B，在 LiveCodeBench 上得分 63.0，与竞品相当；Skywork-OR1-7B 在 AIME24、AIME25 和 LiveCodeBench 上分别得 70.2、54.6 和 47.6，表现出与同规模模型的竞争力。相比基线，32B 模型平均准确率从 57.8% 提升至 72.8%（+15.0%），7B 模型从 43.6% 提升至 57.5%（+13.9%）。
*   **实验设置全面合理**：实验覆盖数学和编码两大领域，使用 AIME24、AIME25 和 LiveCodeBench 等挑战性基准，评估指标包括 avg@32 和 avg@4，生成长度上限设为 32K token，确保公平性。训练配置采用多阶段上下文长度（8K 到 32K），并测试了多种超参数（如采样温度、批次大小、SGD 步数）对性能和熵动态的影响。
*   **消融实验验证有效性**：多阶段训练显著提高训练效率（早期短上下文节省约 100 小时）；高温度采样（τ=1.0）增强早期学习信号；自适应熵控制有效防止熵崩塌，优于固定熵损失系数；去除 KL 损失避免策略过早收敛；在线策略更新优于离线更新，避免熵快速下降。
*   **局限性与成本**：尽管性能提升明显，方法对计算资源需求较高（使用 32-256 个 H800 GPU），且未探讨在非数学/编码任务上的泛化性；熵控制在高 SGD 步数下仍不稳定。

## Further Thoughts

论文中关于熵崩塌的研究提供了深刻启发，探索与利用的平衡是 RL 训练的核心问题，自适应熵控制的动态调整机制可以进一步优化，例如根据任务难度或训练阶段设计分层目标熵值，以更精细地控制探索行为。此外，多阶段训练的思路不仅适用于长 CoT 模型，也可能推广到其他需要逐步复杂化的任务（如对话生成或多步决策），通过早期简化任务降低训练成本，后期扩展复杂性提升性能。最后，数据质量对 RL 训练的影响显著，模型感知的难度估计和人-LLM 联合评估的数据筛选方法，启发我们在其他领域设计类似机制，确保训练数据的挑战性和多样性，从而提升模型的泛化能力。