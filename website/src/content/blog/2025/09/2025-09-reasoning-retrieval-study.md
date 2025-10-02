---
title: "Reasoning or Retrieval? A Study of Answer Attribution on Large Reasoning Models"
pubDatetime: 2025-09-29T01:13:33+00:00
slug: "2025-09-reasoning-retrieval-study"
type: "arxiv"
id: "2509.24156"
score: 0.8495879420480237
author: "grok-3-latest"
authors: ["Yuhui Wang", "Changjiang Li", "Guangke Chen", "Jiacheng Liang", "Ting Wang"]
tags: ["LLM", "Reasoning", "Retrieval", "Fine-Tuning", "Reinforcement Learning"]
institution: ["Stony Brook University"]
description: "本文通过联合扰动实验验证了大型推理模型答案生成受推理和检索双重机制影响，并提出 FARL 框架，通过遗忘机制增强推理主导性，显著提升模型性能和泛化能力。"
---

> **Summary:** 本文通过联合扰动实验验证了大型推理模型答案生成受推理和检索双重机制影响，并提出 FARL 框架，通过遗忘机制增强推理主导性，显著提升模型性能和泛化能力。 

> **Keywords:** LLM, Reasoning, Retrieval, Fine-Tuning, Reinforcement Learning

**Authors:** Yuhui Wang, Changjiang Li, Guangke Chen, Jiacheng Liang, Ting Wang

**Institution(s):** Stony Brook University


## Problem Background

大型推理模型（LRMs）通过链式推理（Chain-of-Thought, CoT）在复杂问题解决中表现出色，但其最终答案常与推理轨迹不一致。
作者假设这种不一致源于两种竞争机制：CoT 推理（Reasoning）和内部记忆检索（Retrieval），并指出当前推理微调范式存在局限，模型可能通过检索机制‘走捷径’，绕过真正推理，削弱推理能力发展，影响可解释性和泛化能力。

## Method

*   **联合扰动框架：** 为了验证推理和检索机制的竞争影响，作者设计了联合扰动实验：
    *   **推理扰动（Reasoning Perturbation）：** 在 CoT 生成过程中注入误导性线索（如提示一个错误答案），观察最终答案是否受影响。
    *   **检索扰动（Retrieval Perturbation）：** 通过监督微调（SFT）‘毒化’模型记忆，强制模型记住特定提示与错误答案的关联，观察答案是否直接从记忆中检索。
    *   **联合扰动：** 同时进行推理和检索扰动，设置一致或不一致的目标答案，分析两种机制的相对主导性。
*   **FARL 框架（Forgetting-Augmented Reinforcement Learning）：** 针对检索机制‘hack’强化学习（RL）奖励信号的问题，提出一种结合记忆遗忘与 RL 的微调方法：
    *   使用 Negative Preference Optimization (NPO) 作为遗忘方法，抑制模型对检索捷径的依赖。
    *   在 RL 训练（如 Group Relative Policy Optimization, GRPO）中引入遗忘步骤，净化奖励信号，强制模型依赖推理能力。
    *   通过迭代更新，确保模型逐步遗忘特定记忆答案，增强推理主导性。

## Experiment

*   **联合扰动实验效果：** 实验表明推理和检索机制同时影响答案生成，扰动成功率（R-PSR 和 T-PSR）均非零，且联合扰动效果更强，验证了两者协同作用；推理主导性在数学逻辑领域、较大模型和 RL 训练模型中更显著，而蒸馏模型更倾向检索主导，常出现‘事后解释’现象。
*   **FARL 效果：** 相比标准 RL 和 SFT，FARL 显著降低扰动成功率（R-PSR 下降 47.8%，T-PSR 下降 38.5%），表明推理主导性增强；准确率在域内提升 22.8%，域外提升 5.8%，推理图质量（如 Small World Index）提升 84.0%，效果明显优于基线。
*   **实验设置：** 覆盖多个数据集（MMLU, ARC-Easy, ARC-Challenge, GPQA）和模型（R1-Llama-8B, R1-Qwen 系列, Qwen3 系列等），考虑了模型规模、问题领域和训练方法，设计全面合理；但 FARL 导致推理轨迹变长，可能增加计算成本。

## Further Thoughts

FARL 的遗忘机制启发了我，是否可以通过‘动态遗忘’策略，根据任务类型或模型状态调整遗忘强度，以更灵活地平衡推理和检索能力？此外，‘事后解释’现象提示模型可解释性可能被高估，未来可以探索通过约束 CoT 生成过程（如强制逻辑一致性）来提升答案与推理轨迹的一致性。