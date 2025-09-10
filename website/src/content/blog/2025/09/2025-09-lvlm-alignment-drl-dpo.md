---
title: "Aligning Large Vision-Language Models by Deep Reinforcement Learning and Direct Preference Optimization"
pubDatetime: 2025-09-08T14:47:57+00:00
slug: "2025-09-lvlm-alignment-drl-dpo"
type: "arxiv"
id: "2509.06759"
score: 0.8116607153238521
author: "grok-3-latest"
authors: ["Thanh Thi Nguyen", "Campbell Wilson", "Janis Dalins"]
tags: ["LVLM", "Multimodal Alignment", "Deep Reinforcement Learning", "Direct Preference Optimization", "Human Feedback"]
institution: ["AiLECS Lab, Monash University, Australia", "ICMEC Australia, Sydney"]
description: "本文系统综述了深度强化学习（DRL）和直接偏好优化（DPO）在对齐大型视觉-语言模型（LVLMs）中的应用，分析了各自优劣并指出了未来研究方向。"
---

> **Summary:** 本文系统综述了深度强化学习（DRL）和直接偏好优化（DPO）在对齐大型视觉-语言模型（LVLMs）中的应用，分析了各自优劣并指出了未来研究方向。 

> **Keywords:** LVLM, Multimodal Alignment, Deep Reinforcement Learning, Direct Preference Optimization, Human Feedback

**Authors:** Thanh Thi Nguyen, Campbell Wilson, Janis Dalins

**Institution(s):** AiLECS Lab, Monash University, Australia, ICMEC Australia, Sydney


## Problem Background

大型视觉-语言模型（LVLMs）在跨视觉和文本模态的内容理解与生成方面表现出色，但预训练模型在与人类价值观对齐及执行特定任务时仍面临挑战。
论文聚焦于如何通过细调（fine-tuning）方法使 LVLMs 更好地适应人类偏好，提升任务性能，并实现自适应的多模态交互，解决对齐过程中的数据依赖、计算成本和泛化性等问题。

## Method

*   **深度强化学习（DRL）：**
    *   **基本框架：** 将模型细调问题建模为马尔可夫决策过程（MDP），其中状态包括图像和文本输入，动作是生成下一个 token，奖励函数通过人类反馈、AI 反馈或规则设计获得。
    *   **具体算法：** 常用 Proximal Policy Optimization (PPO) 及其变体如 Group Relative Policy Optimization (GRPO)。PPO 通过裁剪目标函数确保训练稳定性，GRPO 则通过组内优势计算优化策略。
    *   **特点：** DRL 允许灵活定义复杂奖励目标，支持多信号优化，适用于复杂对齐任务，但训练过程不稳定且计算成本高。
*   **直接偏好优化（DPO）：**
    *   **基本框架：** 直接利用偏好数据（正向和负向响应对）优化模型策略，通过分类损失函数调整输出分布以匹配人类偏好，避免显式奖励模型的构建。
    *   **具体实现：** DPO 将偏好学习转化为分类问题，基于多模态输入（如图像和文本提示）及偏好对，最大化正向响应的对数似然比。
    *   **特点：** DPO 训练更稳定，资源需求较低，适用于大规模偏好数据调优，但对数据质量依赖较高，灵活性不如 DRL。
*   **混合方法：** 结合 DRL 和 DPO 的优势，如在 Llama 4 中应用，以平衡灵活性和稳定性。

## Experiment

*   **DRL 效果：** 在复杂对齐任务中表现强劲，尤其结合人类反馈时能实现细粒度行为控制（如减少幻觉、提升推理能力），但训练不稳定，对奖励设计和超参数敏感，计算成本高。
*   **DPO 效果：** 在多个基准测试中性能与 DRL 相当甚至更优，尤其在减少幻觉和视觉推理任务中表现突出，单阶段训练方法更高效，但效果依赖偏好数据质量。
*   **实验设置：** 实验覆盖多种基础模型（如 LLaVA、InternVL）、任务（如视觉问答、减少幻觉）和数据集（如 Preference-10K、MM-RLHF），设置较为全面，但缺乏具体数值结果，更多为定性分析。
*   **合理性与局限：** 奖励来源（人类、AI、规则）和偏好数据质量对结果影响显著，实验设计合理但泛化性测试和数据一致性仍有改进空间。

## Further Thoughts

DRL 和 DPO 的混合应用为多模态模型对齐提供了新思路，未来可以探索更智能的奖励建模方法（如基于偏好学习的动态奖励）或高效数据采集策略（如主动学习）；此外，LVLMs 的多模态特性可能启发其他领域（如视频-语言模型）的对齐技术，视觉与文本信号的联合优化或将成为研究热点。