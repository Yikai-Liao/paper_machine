---
title: "Reinforcement learning meets bioprocess control through behaviour cloning: Real-world deployment in an industrial photobioreactor"
pubDatetime: 2025-09-08T16:21:11+00:00
slug: "2025-09-rl-bioprocess-control"
type: "arxiv"
id: "2509.06853"
score: 0.6567350772462135
author: "grok-3-latest"
authors: ["Juan D. Gil", "Ehecatl Antonio Del Rio Chanona", "José L. Guzmán", "Manuel Berenguel"]
tags: ["Reinforcement Learning", "Behavior Cloning", "Bioprocess Control", "pH Regulation", "Dynamic Adaptation"]
institution: ["Universidad de Almería", "Imperial College London"]
description: "本文提出并验证了一种基于强化学习和行为克隆的 pH 控制策略，首次在工业规模开放式光生物反应器中实现 RL 控制的实际部署，显著提升了控制精度和效率。"
---

> **Summary:** 本文提出并验证了一种基于强化学习和行为克隆的 pH 控制策略，首次在工业规模开放式光生物反应器中实现 RL 控制的实际部署，显著提升了控制精度和效率。 

> **Keywords:** Reinforcement Learning, Behavior Cloning, Bioprocess Control, pH Regulation, Dynamic Adaptation

**Authors:** Juan D. Gil, Ehecatl Antonio Del Rio Chanona, José L. Guzmán, Manuel Berenguel

**Institution(s):** Universidad de Almería, Imperial College London


## Problem Background

开放式光生物反应器（PBRs）因暴露于不断变化的环境条件（如光照、温度、溶解氧等）而具有高度非线性和多干扰特性，传统控制方法（如 PID 控制器）难以维持稳定的 pH 值，而 pH 直接影响微藻生长和生产效率。
本文旨在解决这一复杂生物过程中的 pH 调控问题，以支持工业规模微藻生产的稳定性和效率。

## Method

*   **核心思想:** 提出一种结合强化学习（Reinforcement Learning, RL）和行为克隆（Behavior Cloning, BC）的控制策略，通过离线学习专家行为和在线微调适应动态变化，实现开放式 PBR 中的 pH 调控。
*   **具体实现:** 
    *   **离线训练阶段:** 基于 Deep Deterministic Policy Gradient (DDPG) 算法，利用 PID 控制器生成的历史数据训练 RL 代理，通过行为克隆模仿专家系统的控制策略，避免直接与真实系统交互的高风险。
    *   **在线微调阶段:** 每日通过有限次数的训练更新代理策略，适应新的运行条件和干扰，逐步用新收集的数据替换历史数据，确保策略的动态适应性。
    *   **观察空间设计:** 将问题建模为部分可观测马尔可夫决策过程（POMDP），观察空间包括直接测量的过程变量（如温度、光照、溶解氧）、时间信息（如昼夜周期）以及控制变量（如误差和误差积分），以推断隐藏状态并增强干扰抑制能力。
    *   **奖励函数设计:** 采用对数形式的奖励函数，平滑大误差的影响并放大接近零的误差对梯度的作用，提升训练稳定性。
    *   **动作空间:** 控制 CO2 注入速率，考虑物理约束并引入防饱和机制以避免积分饱和问题。
*   **关键点:** 该方法无需精确的系统模型，通过数据驱动的方式学习控制策略，同时结合离线和在线阶段平衡了安全性和适应性。

## Experiment

*   **有效性:** 模拟研究表明，提出的 RL-FT（带微调的 RL）方法在积分绝对误差（IAE）上比 PID 控制器提高了 8%，比标准离线 RL 提高了 5%；控制努力（CCE）分别降低了 54% 和 7%，显示出显著的控制精度和操作成本优势。
*   **真实系统验证:** 在工业规模 PBR 上进行了为期 8 天的实验，验证了方法的鲁棒性，最大 pH 偏差为 0.11，微调后进一步改善了干扰抑制能力，即使面对传感器校准和通信故障等挑战仍保持稳定。
*   **实验设置合理性:** 实验涵盖了不同季节和操作条件（如周末无收获与工作日有收获），数据采集和微调周期（每日 24 小时更新）设计合理，充分考虑了实际工业环境中的动态变化和多干扰特性。
*   **开销:** 主要计算开销在于离线训练和每日微调的神经网络更新，但通过限制微调迭代次数避免了过拟合和不稳定性。

## Further Thoughts

离线-在线混合训练策略为 RL 在高风险工业环境中的应用提供了实用框架，避免直接在线探索的风险；观察空间的设计思路（融合多源信息）为处理部分可观测环境提供了参考，可推广至其他复杂系统；奖励函数的对数设计在平滑大误差同时放大小误差对梯度的影响，这一技巧可能在其他 RL 控制任务中具有普适性。