---
title: "Self-correcting Reward Shaping via Language Models for Reinforcement Learning Agents in Games"
pubDatetime: 2025-06-30T08:45:04+00:00
slug: "2025-06-self-correcting-reward-shaping"
type: "arxiv"
id: "2506.23626"
score: 0.7418866955922266
author: "grok-3-latest"
authors: ["António Afonso", "Iolanda Leite", "Alessandro Sestini", "Florian Fuchs", "Konrad Tollmar", "Linus Gisslén"]
tags: ["Reinforcement Learning", "Language Model", "Reward Shaping", "Behavioral Tuning", "Game AI"]
institution: ["SEED - Electronic Arts (EA), Stockholm, Sweden", "KTH Royal Institute of Technology, Stockholm, Sweden"]
description: "本文提出了一种基于语言模型的自校正奖励塑造方法，通过迭代闭环优化奖励函数权重，显著提升游戏中RL代理性能，接近人类专家水平，同时降低对专家依赖。"
---

> **Summary:** 本文提出了一种基于语言模型的自校正奖励塑造方法，通过迭代闭环优化奖励函数权重，显著提升游戏中RL代理性能，接近人类专家水平，同时降低对专家依赖。 

> **Keywords:** Reinforcement Learning, Language Model, Reward Shaping, Behavioral Tuning, Game AI

**Authors:** António Afonso, Iolanda Leite, Alessandro Sestini, Florian Fuchs, Konrad Tollmar, Linus Gisslén

**Institution(s):** SEED - Electronic Arts (EA), Stockholm, Sweden, KTH Royal Institute of Technology, Stockholm, Sweden


## Problem Background

强化学习（Reinforcement Learning, RL）在游戏开发中展现出巨大潜力，但生产环境中面临两大挑战：设计有效的奖励函数需要RL专家的参与，以及当游戏内容或机制变化时，之前调优的奖励权重可能失效，需要反复手动调整。
本文的出发点是通过语言模型（Language Model, LM）自动化奖励函数的调优过程，减少对专家依赖，并适应动态变化的游戏环境，解决如何根据用户定义的高级行为目标（如‘尽可能快地驾驶’），迭代优化RL代理行为的问题。

## Method

*   **核心思想:** 利用语言模型（LM）通过一个闭环迭代过程，自动化调整奖励函数的权重，使RL代理的行为逐步逼近用户定义的高级目标，而无需手动奖励工程。
*   **奖励函数设计:** 定义一个模块化的线性奖励函数，形式为 r_t = Σ w_k * f_k(s_t, a_t)，其中 f_k 是衡量特定行为特征的子函数（如速度、赛道偏离等），w_k 是可调权重，控制各子函数的重要性。
*   **迭代调优流程:** 
    *   **初始权重提议:** LM接收环境描述、用户目标（如‘快速驾驶但不偏离赛道’）以及历史数据（初始为空），生成初始权重向量 w。
    *   **RL训练与评估:** 使用当前权重训练RL代理至收敛，运行评估回合（50个回合，5个随机种子），收集性能统计数据（如成功率、平均速度）。
    *   **反馈与自校正:** 将性能数据反馈给LM，结合历史权重和结果，生成新的权重向量 w，进行下一轮迭代。
    *   **迭代次数:** 共进行5次迭代，基于经验观察性能提升趋于平缓。
*   **关键创新:** 将奖励塑造转化为权重优化问题，LM通过文本提示和性能反馈自校正，无需直接访问环境代码或视觉信息，降低了调优门槛。

## Experiment

*   **任务与设置:** 实验在一个赛车任务中进行，目标是尽可能快地完成赛道圈数，同时避免偏离赛道；评估指标包括成功率、偏离赛道比例、超时比例、平均速度和完成步数；对比LM引导的调优与人类专家手动调优，迭代5次，每次评估50个回合，重复5个随机种子。
*   **有效性:** LM引导的代理性能显著提升，成功率从初始12.4%提升至第1次迭代的73.6%，最终在第5次迭代达到80.4%；平均速度从121.3 km/h增至135.2 km/h，完成步数减少至855步。
*   **对比人类专家:** 人类专家在第4次迭代达到峰值成功率93.6%，平均速度136.2 km/h，完成步数850步，略优于LM；但第5次迭代性能下降至26.0%，显示手动调优的不稳定性，而LM调优更稳定。
*   **合理性与局限:** 实验设置合理，统计数据可靠，赛车任务涵盖速度与控制的权衡，具有代表性；但实验仅限于单一环境，泛化性需进一步验证。

## Further Thoughts

论文中利用语言模型迭代自校正的能力，将用户抽象目标转化为具体奖励权重的思路非常启发性，不仅限于游戏领域，也可扩展至机器人控制或自动驾驶等RL任务；此外，LM仅依赖文本和统计数据即可接近人类专家水平，提示我们可以在资源受限场景中探索轻量级反馈机制；未来引入视觉-语言模型处理环境图像或视频的设想也值得关注，或许可以通过设计更智能的提示或结合领域知识，进一步提升LM在复杂任务中的决策能力。