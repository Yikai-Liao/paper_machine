---
title: "Cooperative Autonomous Driving in Diverse Behavioral Traffic: A Heterogeneous Graph Reinforcement Learning Approach"
pubDatetime: 2025-09-30T04:12:57+00:00
slug: "2025-09-heterogeneous-graph-av"
type: "arxiv"
id: "2509.25751"
score: 0.7342636022398382
author: "grok-3-latest"
authors: ["Qi Liu", "Xueyuan Li", "Zirui Li", "Juhui Gim"]
tags: ["Autonomous Driving", "Graph Neural Network", "Reinforcement Learning", "Heterogeneous Traffic", "Expert System"]
institution: ["Beijing Institute of Technology", "Changwon National University"]
description: "本文提出了一种异构图强化学习结合专家系统的框架，通过异构图表示和 HGNN-EM 模型显著提升了自动驾驶车辆在异构交通环境中的决策性能。"
---

> **Summary:** 本文提出了一种异构图强化学习结合专家系统的框架，通过异构图表示和 HGNN-EM 模型显著提升了自动驾驶车辆在异构交通环境中的决策性能。 

> **Keywords:** Autonomous Driving, Graph Neural Network, Reinforcement Learning, Heterogeneous Traffic, Expert System

**Authors:** Qi Liu, Xueyuan Li, Zirui Li, Juhui Gim

**Institution(s):** Beijing Institute of Technology, Changwon National University


## Problem Background

自动驾驶车辆（AVs）在异构交通环境中面临复杂挑战，由于人类驾驶车辆（HVs）具有多种驾驶风格（如激进、普通、保守），其交互动态性和复杂性使得 AV 的决策变得困难。
现有方法在建模异构交通参与者间的复杂交互以及处理异构特征时存在局限，导致决策性能不佳或不安全。
论文旨在通过准确建模车辆交互并高效编码异构信息，提升 AV 在安全性和效率上的表现。

## Method

*   **异构图表示（Heterogeneous Graph Representation）：** 将交通场景建模为异构无向图，节点代表车辆（包括 AV 和不同风格的 HVs），边代表车辆间交互。
    *   节点特征根据车辆类型提取不同驾驶特征，例如激进型 HVs 使用最大加速度，保守型 HVs 使用最大减速度，普通型 HVs 使用平均加速度。
    *   边特征根据驾驶风格设计不同交互模型，例如激进型 HVs 使用二维碰撞时间（2D-TTC）评估风险，普通型 HVs 使用加速度差异，保守型 HVs 使用相对距离。
*   **异构图神经网络结合专家模型（HGNN-EM）：** 包含四个关键组件：
    *   **预编码器（Pre-Encoders）：** 通过节点编码器和边编码器将异构特征映射到统一嵌入空间，节点编码器处理不同类型车辆特征，边编码器学习车辆间交互的边值。
    *   **关系图注意力网络（R-GAT）：** 扩展传统图注意力网络，处理多种边类型，计算不同关系类型的注意力值并聚合异构特征，生成驾驶策略。
    *   **专家模型（Expert Model）：** 通过监督学习从专家驾驶数据中学习，提供领域知识指导，输入为异构图状态空间，输出为专家策略。
    *   **策略融合模型（Policy Fusion Model）：** 动态融合 GRL 策略和专家策略，通过融合权重生成器根据交通特征调整权重，确保决策兼顾学习能力和专家经验。
*   **策略优化：** 使用双重深度 Q 学习（Double Deep Q-Learning, DDQN）算法训练决策模型，结合经验回放和目标网络更新策略，提升训练稳定性和收敛性。

## Experiment

*   **有效性：** 在四路交叉路口场景中，与基线模型（GCQ、HGNN、STGCN）相比，提出的方法在安全性（零碰撞）、效率（更高平均速度、更短行驶时间）、稳定性和收敛速度（约 120 个 episode 收敛）上均表现优异。
*   **实验设置合理性：** 实验基于 SUMO 仿真器，包含六辆不同风格 HVs，参数通过 IDM 和 MOBIL 模型调整，评价指标涵盖奖励、碰撞次数、平均速度、行驶时间和计算效率，全面评估性能；专家模型训练数据充足（5.5×10^5 样本），测试准确率达 99.23%。
*   **局限性：** 实验场景较为单一，仅限于四路交叉路口，未涉及更复杂场景；计算效率虽优于 STGCN，但与 GCQ 和 HGNN 相当，未显著降低计算成本。

## Further Thoughts

异构图表示的思路启发了我，是否可以将这种方法扩展到其他多实体交互场景，如多智能体协作或社交网络分析？
此外，动态策略融合机制提示，是否可以通过自适应权重调整，在不同风险场景中平衡学习策略与专家策略的依赖度？
另一个值得探索的方向是，现实中 HVs 驾驶风格可能需要预测，是否可以集成风格预测模块，进一步提升模型在真实环境中的适用性？