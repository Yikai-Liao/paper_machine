---
title: "Trust, But Verify: A Self-Verification Approach to Reinforcement Learning with Verifiable Rewards"
pubDatetime: 2025-05-19T17:59:31+00:00
slug: "2025-05-rise-self-verification"
type: "arxiv"
id: "2505.13445"
score: 0.5233483949519526
author: "grok-3-latest"
authors: ["Xiaoyuan Liu", "Tian Liang", "Zhiwei He", "Jiahao Xu", "Wenxuan Wang", "Pinjia He", "Zhaopeng Tu", "Haitao Mi", "Dong Yu"]
tags: ["LLM", "Reinforcement Learning", "Self-Verification", "Reasoning", "Verifiable Rewards"]
institution: ["Tencent"]
description: "本文提出 RISE 框架，通过在线强化学习同时提升大型语言模型的问题解决和自我验证能力，显著改善推理性能和自我意识。"
---

> **Summary:** 本文提出 RISE 框架，通过在线强化学习同时提升大型语言模型的问题解决和自我验证能力，显著改善推理性能和自我意识。 

> **Keywords:** LLM, Reinforcement Learning, Self-Verification, Reasoning, Verifiable Rewards

**Authors:** Xiaoyuan Liu, Tian Liang, Zhiwei He, Jiahao Xu, Wenxuan Wang, Pinjia He, Zhaopeng Tu, Haitao Mi, Dong Yu

**Institution(s):** Tencent


## Problem Background

大型语言模型（LLMs）在复杂推理任务中展现出巨大潜力，但通过强化学习（特别是使用可验证奖励的 RLVR）训练的模型常表现出‘表面自我反思’问题，即模型可能生成正确答案，却缺乏对推理过程的深入理解和稳健的自我评估能力，导致无法可靠识别自身推理缺陷。
论文旨在解决如何在强化学习过程中同时提升模型的问题解决能力和自我验证能力，以构建更稳健、更自知的推理者。

## Method

*   **核心思想:** 提出 RISE（Reinforcing Reasoning with Self-Verification），一个在线强化学习框架，通过可验证奖励信号同时训练模型的问题解决和自我验证能力，确保两种能力在单一 RL 过程中同步提升。
*   **具体实现:**
    *   **问题解决生成:** 模型基于输入问题生成多个解决方案（包含推理过程和最终答案），通过基于规则的结果验证器（Outcome Verifier, OV）计算奖励，评估答案和格式的正确性。
    *   **在线解决方案验证:** 利用模型当前策略生成的解决方案和原始问题，构建验证任务，模型需对自己的解决方案进行批判并评分；评分结果由 OV 提供真实标签作为监督，确保验证任务与生成任务紧密相关。
    *   **强化学习整合:** 使用 Proximal Policy Optimization (PPO) 算法，将问题解决和自我验证的轨迹（trajectories）结合在一个统一的 RL 目标中，通过联合优化更新模型参数；采用 Generalized Advantage Estimation (GAE) 估计优势值，确保训练稳定性。
*   **关键创新:** 在线验证机制确保验证任务基于模型当前策略生成的解决方案，而非预先固定的数据，提供实时反馈；同时，共享的价值函数（critic）跨任务学习，进一步增强两种能力的协同性。

## Experiment

*   **有效性:** RISE 模型在推理准确率和自我验证准确率上均显著优于 Zero-RL 基线（仅包含问题解决监督），例如 RISE-1.5B 的验证准确率从 26.8% 提升至 74.5%，推理准确率平均提升约 0.4%-1.2%；在困难基准（如 AIME24）上验证能力提升尤为明显。
*   **模型规模扩展:** 随着模型规模从 1.5B 到 7B 增加，推理性能持续提升（例如 RISE-7B 平均推理准确率达 42.9%），而验证性能保持高水平（均超过 69%），表明框架在不同规模下均有效。
*   **测试时扩展:** RISE 模型通过自我验证加权投票在测试时进一步提升准确率，例如 RISE-7B 在 k=4 采样预算下比标准多数投票提升 1.9%，验证了自我验证能力对推理性能的贡献。
*   **实验设置合理性:** 实验基于 Qwen2.5 系列模型（1.5B、3B、7B），覆盖多个数学推理基准（如 MATH500、AIME 2024），对比了基线模型（SFT、Zero-RL）和外部验证器（如 GPT-4o）；消融实验验证了在线验证的重要性，离线验证导致验证准确率显著下降；此外，验证计算资源比例的扩展实验显示验证性能随资源增加持续提升，推理性能保持稳定。

## Further Thoughts

RISE 的在线验证机制启发了我思考如何在其他领域（如代码生成或物理推理）中应用类似的可验证奖励机制，通过自我验证增强模型的泛化能力；此外，论文中验证计算资源（verification compute）扩展性的分析让我考虑是否可以通过动态调整验证数据比例来优化训练效率，或者结合外部工具（如检索增强生成 RAG）进一步提升验证的准确性和可靠性。