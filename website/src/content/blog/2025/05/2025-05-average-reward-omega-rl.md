---
title: "Average Reward Reinforcement Learning for Omega-Regular and Mean-Payoff Objectives"
pubDatetime: 2025-05-21T16:06:51+00:00
slug: "2025-05-average-reward-omega-rl"
type: "arxiv"
id: "2505.15693"
score: 0.7508631854735102
author: "grok-3-latest"
authors: ["Milad Kazemi", "Mateo Perez", "Fabio Somenzi", "Sadegh Soudjani", "Ashutosh Trivedi", "Alvaro Velasquez"]
tags: ["Reinforcement Learning", "Average Reward", "Omega-Regular", "Continuing Tasks", "Multi-Objective Optimization"]
institution: ["King's College London", "University of Colorado Boulder", "Max Planck Institute for Software Systems"]
description: "本文提出一种无模型平均奖励强化学习框架，将 omega-regular 规范转化为平均奖励目标，在持续任务中实现策略合成与多目标优化。"
---

> **Summary:** 本文提出一种无模型平均奖励强化学习框架，将 omega-regular 规范转化为平均奖励目标，在持续任务中实现策略合成与多目标优化。 

> **Keywords:** Reinforcement Learning, Average Reward, Omega-Regular, Continuing Tasks, Multi-Objective Optimization

**Authors:** Milad Kazemi, Mateo Perez, Fabio Somenzi, Sadegh Soudjani, Ashutosh Trivedi, Alvaro Velasquez

**Institution(s):** King's College London, University of Colorado Boulder, Max Planck Institute for Software Systems


## Problem Background

强化学习（Reinforcement Learning, RL）在持续任务（continuing tasks）中面临挑战，传统折扣奖励（discounted reward）机制因关注短期收益而难以捕捉长期行为目标，且依赖周期性重置（episodic resetting）与持续任务的无限交互特性不符。
论文旨在解决如何将 omega-regular 规范（一种描述无限行为轨迹的形式化语言）转化为平均奖励（average reward）目标，以在未知通信型马尔可夫决策过程（communicating MDPs）中，通过无模型强化学习合成满足规范的策略，同时探索在满足规范的前提下优化额外平均奖励目标（lexicographic multi-objective optimization）。

## Method

*   **核心思想:** 提出一种无模型强化学习框架，将 omega-regular 规范（特别是绝对活性属性，absolute liveness specifications）转化为平均奖励目标，通过奖励机器（Reward Machine）构建和重置机制，确保在持续任务中学习到满足规范的策略。
*   **奖励机器构建:** 基于 omega-regular 规范的 Büchi 自动机，构造奖励机器，在每个状态添加重置（reset）动作，确保产品 MDP（product MDP）保持通信性（communicating property）；奖励函数设计为在接受转换（accepting transitions）时给予正奖励，在重置时给予负奖励，引导策略避免无限重置。
*   **平均奖励强化学习:** 使用差分 Q 学习（Differential Q-learning）算法，在产品 MDP 上优化平均奖励目标，重置机制确保算法收敛，同时避免对环境进行周期性重置。
*   **词典序多目标优化:** 在满足 omega-regular 规范的前提下，优化额外的平均奖励目标，通过引入概率奖励机器（Probabilistic Reward Machine），在两个目标间进行权衡，确保策略既满足规范，又尽可能优化外部奖励。
*   **关键创新:** 通过重置机制解决产品 MDP 非通信性问题，将非马尔可夫目标（non-Markovian objectives）转化为马尔可夫目标（Markovian objectives），从而适用标准无模型强化学习算法。

## Experiment

*   **有效性:** 在持续任务中，相比基于折扣奖励的方法（如 [32] 和 [11] 的方法），论文提出的平均奖励方法表现更优，实验表明其在多种超参数组合下均能找到满足规范的策略（概率为 1），而折扣奖励方法往往无法收敛。
*   **优越性:** 即使允许周期性重置，本文方法在训练时间上与先前方法相当，且无需重置即可学习到最优策略，显示出更高的鲁棒性。
*   **多目标优化效果:** 在多目标案例中，方法成功以概率 1 满足绝对活性规范，同时获得的平均奖励接近最优值，表明方法在平衡规范满足与奖励优化方面有效。
*   **实验设置合理性:** 实验覆盖了多种通信型 MDP 和绝对活性规范，超参数通过采样和手动调整结合，确保结果的可靠性；提供了详细的训练步数和时间数据，设置较为全面。

## Further Thoughts

论文提出的平均奖励机制比折扣奖励更适合持续任务的观点，启发我们在工业控制、自主监控等长期运行场景中更多采用平均奖励设计；形式化规范与强化学习的结合思路，可扩展到其他形式化逻辑，探索更广泛应用；重置机制的设计不仅适用于 omega-regular 规范，也可能启发其他需要产品构造的强化学习任务中引入类似‘逃逸机制’来处理复杂状态空间。