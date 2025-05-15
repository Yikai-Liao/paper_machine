---
title: "Explainable Reinforcement Learning Agents Using World Models"
pubDatetime: 2025-05-12T21:18:31+00:00
slug: "2025-05-reverse-world-model"
type: "arxiv"
id: "2505.08073"
score: 0.4079774320832991
author: "grok-3-latest"
authors: ["Madhuri Singh", "Amal Alabdulkarim", "Gennie Mansi", "Mark O. Riedl"]
tags: ["Reinforcement Learning", "World Model", "Explainability", "Counterfactual Reasoning", "User Interaction"]
institution: ["Georgia Institute of Technology"]
description: "本文提出通过反向世界模型（Reverse World Model）生成反事实状态，为非 AI 专家提供可操作的强化学习解释，显著提升用户理解、信任和满意度，同时降低认知负荷。"
---

> **Summary:** 本文提出通过反向世界模型（Reverse World Model）生成反事实状态，为非 AI 专家提供可操作的强化学习解释，显著提升用户理解、信任和满意度，同时降低认知负荷。 

> **Keywords:** Reinforcement Learning, World Model, Explainability, Counterfactual Reasoning, User Interaction

**Authors:** Madhuri Singh, Amal Alabdulkarim, Gennie Mansi, Mark O. Riedl

**Institution(s):** Georgia Institute of Technology


## Problem Background

强化学习（RL），尤其是深度强化学习（DRL），由于其策略模型基于神经网络，决策过程对用户而言往往是不透明的黑箱，尤其是在序列决策的时序背景下。
非 AI 专家用户无法直接修改或重新训练代理的策略，当代理行为偏离用户期望时，用户会感到困惑并丧失信任。
论文旨在解决如何为非 AI 专家提供可操作的解释（actionable explanations），帮助他们理解代理为何做出特定决策，并通过调整环境来间接影响代理行为。

## Method

*   **核心思想:** 利用世界模型（World Model, WM）生成可解释的反事实轨迹（counterfactual trajectories），帮助用户理解代理决策背后的环境因素，并提供可操作的洞察。
*   **具体实现:**
    *   **前向世界模型（Forward World Model, FWM）:** 在基于模型的强化学习中，FWM 通过与环境交互学习状态转移动态，预测给定当前状态和动作后的下一状态，用于模拟代理行为。
    *   **反向世界模型（Reverse World Model, RWM）:** 创新性地引入 RWM，预测为了让代理选择用户期望的动作，当前状态应该是什么样的（即反事实状态）。RWM 通过反转训练数据的时序顺序，与 FWM 共同训练，使用相同的经验回放缓冲区，确保预测与代理策略一致。
    *   **解释生成过程:** 当代理未执行用户期望的动作时，系统通过 RWM 生成反事实状态（即‘世界应该是什么样’），并将其以图像或状态描述形式呈现给用户，帮助用户理解代理决策依赖的环境条件，并可能通过改变环境来影响未来行为。
*   **关键优势:** 不需要修改或重新训练代理策略，解释直接基于代理对环境的理解，特别适合非技术用户，且通过环境操控提供间接控制手段。

## Experiment

*   **有效性:** 通过人类参与者研究验证，在 Crafter 虚拟环境中，实验组（提供 RWM 生成的解释）在识别代理失败原因的准确率上显著高于对照组（p < 0.00001），用户满意度（p ≈ 0.0036）、信任度（p ≈ 0.034）显著提升，认知负荷显著降低（p ≈ 9.76e-05）。
*   **实验设置:** 设计了四种失败场景（非必要对象移除、必要对象移动、必要对象被阻挡、必要对象移除），参与者背景多样（年龄 19-73 岁，性别均衡），使用 Fisher’s Exact Test 和 t-test 确保统计显著性。
*   **合理性与局限:** 实验全面覆盖多种失败场景，但代理策略被故意过拟合以在简单环境中失败，可能限制方法在复杂真实场景中的泛化性；任务完成时间无显著差异，表明解释未增加额外负担。

## Further Thoughts

反向世界模型（RWM）的概念不仅限于强化学习，是否可以应用于其他时序模型（如时间序列预测）中，生成‘过去应该是什么样’的解释，帮助用户理解系统行为？
此外，论文通过环境操控影响代理行为的思路是否可以扩展到其他 AI 领域（如推荐系统），让用户通过调整输入条件间接控制输出？
最后，如何通过个性化解释（如根据用户背景调整内容）进一步增强用户心理模型的更新效果？