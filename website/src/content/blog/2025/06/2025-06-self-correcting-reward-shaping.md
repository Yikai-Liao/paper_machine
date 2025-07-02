---
title: "Self-correcting Reward Shaping via Language Models for Reinforcement Learning Agents in Games"
pubDatetime: 2025-06-30T08:45:04+00:00
slug: "2025-06-self-correcting-reward-shaping"
type: "arxiv"
id: "2506.23626"
score: 0.7418866955922266
author: "grok-3-latest"
authors: ["António Afonso", "Iolanda Leite", "Alessandro Sestini", "Florian Fuchs", "Konrad Tollmar", "Linus Gisslén"]
tags: ["LLM", "Reinforcement Learning", "Reward Shaping", "Game AI", "Behavioral Alignment"]
institution: ["SEED - Electronic Arts (EA), Stockholm, Sweden", "KTH Royal Institute of Technology, Stockholm, Sweden"]
description: "本文提出了一种基于语言模型的自我校正循环方法，自动化调优强化学习代理的奖励函数权重，在赛车任务中显著提升性能并接近人类专家水平，降低了专家干预需求。"
---

> **Summary:** 本文提出了一种基于语言模型的自我校正循环方法，自动化调优强化学习代理的奖励函数权重，在赛车任务中显著提升性能并接近人类专家水平，降低了专家干预需求。 

> **Keywords:** LLM, Reinforcement Learning, Reward Shaping, Game AI, Behavioral Alignment

**Authors:** António Afonso, Iolanda Leite, Alessandro Sestini, Florian Fuchs, Konrad Tollmar, Linus Gisslén

**Institution(s):** SEED - Electronic Arts (EA), Stockholm, Sweden, KTH Royal Institute of Technology, Stockholm, Sweden


## Problem Background

强化学习（RL）在游戏开发中展现出巨大潜力，但生产环境中面临两大挑战：设计有效的奖励函数通常需要RL专家参与，而游戏内容或机制变更时，之前调优的奖励权重可能失效，需要反复手动调整。
本文的出发点是通过自动化方法，利用语言模型（LM）根据用户定义的高级行为目标（如‘尽可能快地驾驶’），动态调整RL代理的奖励函数权重，降低对专家依赖并提升适应性。

## Method

*   **核心思想**：提出一种基于语言模型的自我校正循环（self-correcting loop），通过迭代反馈自动调整RL代理奖励函数的权重，使其行为与用户意图对齐，而无需手动奖励工程。
*   **奖励函数设计**：将奖励函数分解为多个子奖励组件的线性组合，每个组件对应特定行为特征（如速度奖励‘speedDriveReward’、偏离赛道惩罚‘offRoadPenalty’），通过权重（w_k）控制各组件的重要性。
*   **LM引导权重调整**：在每个迭代中，LM接收用户定义的行为目标（如‘快速驾驶但不偏离赛道’）、环境描述以及之前迭代的性能统计数据（如成功率、平均速度），基于这些信息推理并提出新的奖励权重向量。
*   **RL训练与评估**：使用LM提出的权重训练RL代理，训练完成后进行评估，收集关键性能指标（如成功率、偏离率）。
*   **反馈与迭代优化**：将评估结果以文本形式反馈给LM，LM根据历史权重和性能数据进一步调整权重，形成闭环优化，运行固定次数（如5次迭代）后输出最终优化的RL策略和奖励函数。
*   **关键优势**：该方法将高层次用户意图转化为低层次奖励权重，自动化程度高，适应性强，特别适合游戏开发中快速迭代或非专家用户使用。

## Experiment

*   **有效性**：在赛车任务中，LM引导的代理性能显著提升，成功率从初始的12.4%在第一次迭代后跃升至73.6%，最终在第5次迭代达到80.4%；平均速度从121.3 km/h提升至135.2 km/h，偏离赛道率从83.2%降至14.8%，表明LM能快速改进代理行为并持续优化。
*   **与人类专家对比**：人类专家调优的代理在第4次迭代达到峰值成功率93.6%，平均速度136.2 km/h，略优于LM最终结果，但第5次迭代性能下降至26.0%，显示手动调优的不稳定性；LM调优则表现出更稳定的进步，行为分布更一致。
*   **实验设置合理性**：实验在赛车任务中进行，目标明确（快速完成赛道圈数），评估指标全面（成功率、偏离率、速度、步数），每次迭代后运行50个评估回合，跨5个随机种子，确保统计可靠性；迭代次数设为5次，基于经验观察性能提升趋于平缓。
*   **局限性**：实验仅限于单一赛车任务，缺乏多环境验证，泛化性待检验；LM仅依赖文本统计数据，缺乏视觉或轨迹反馈，可能错过细微行为特征。

## Further Thoughts

论文中利用语言模型将用户高层次意图映射到奖励权重并通过迭代反馈自我校正的思路非常启发性，提示我们语言模型不仅可用于静态配置生成，还能在动态闭环中适应复杂任务；这一方法可能扩展至机器人控制或NLP任务中的行为塑造。此外，作者提到未来引入视觉语言模型（VLM）处理环境图像或视频反馈，这启发我们可以探索多模态数据（如结合行为轨迹和数值统计）进一步提升LM在RL中的作用，创造更智能的自动化调优系统。