---
title: "VariAntNet: Learning Decentralized Control of Multi-Agent Systems"
pubDatetime: 2025-09-02T12:48:15+00:00
slug: "2025-09-variantnet-swarm-control"
type: "arxiv"
id: "2509.02271"
score: 0.4247643241877714
author: "grok-3-latest"
authors: ["Yigal Koifman", "Erez Koifman", "Eran Iceland", "Ariel Barel", "Alfred M. Bruckstein"]
tags: ["Multi-Agent Systems", "Decentralized Control", "Neural Networks", "Swarm Robotics", "Geometric Processing"]
institution: ["Technion Israel Institute of Technology"]
description: "VariAntNet 提出了一种基于深度学习的去中心化控制模型，通过几何预处理和可见性图损失函数优化多智能体聚集任务，显著提升收敛速度并在可接受范围内维持群体凝聚力。"
---

> **Summary:** VariAntNet 提出了一种基于深度学习的去中心化控制模型，通过几何预处理和可见性图损失函数优化多智能体聚集任务，显著提升收敛速度并在可接受范围内维持群体凝聚力。 

> **Keywords:** Multi-Agent Systems, Decentralized Control, Neural Networks, Swarm Robotics, Geometric Processing

**Authors:** Yigal Koifman, Erez Koifman, Eran Iceland, Ariel Barel, Alfred M. Bruckstein

**Institution(s):** Technion Israel Institute of Technology


## Problem Background

在灾难响应（如消防）等应用中，简单机器人（Ant Robots）需要在复杂环境中自主协作，面临有限感知范围（仅方向感知）、无共享坐标系、无显式通信等挑战。
关键问题是如何在去中心化条件下，通过本地观测维持群体凝聚力并完成协作任务（如聚集），而传统解析方法收敛速度慢，无法满足时间敏感场景的需求。

## Method

*   **核心思想:** 提出 VariAntNet，一种基于深度学习的去中心化控制模型，采用集中训练、去中心化执行（CTDE）框架，通过几何预处理和神经网络架构处理无序、变大小的本地观测，同时优化群体凝聚力和任务完成度。
*   **具体实现:**
    *   **预处理阶段:** 对智能体的本地观测进行旋转变换，确保旋转等变性（Rotational Equivariance），即无论智能体如何旋转，其决策输出保持一致，解决无共享坐标系的问题。
    *   **神经网络架构:** 受 PointNet 启发，网络包含两个模块：第一个模块提取每个邻居的本地几何特征；第二个模块通过最大池化（Max Pooling）聚合特征，输出移动方向和步长。网络支持变大小输入和无序观测，参数量少，计算效率高（CPU 上超 1000 FPS）。
    *   **损失函数设计:** 包含任务损失（Task Loss，衡量聚集任务完成度，基于群体质心最大距离）和凝聚力损失（Cohesiveness Loss，基于可见性图拉普拉斯矩阵的特征值，优化群体连接性）。通过调整损失权重，平衡速度与凝聚力。
*   **关键优势:** 不依赖通信或全局信息，仅通过本地观测实现协作；相比解析方法更适应复杂几何构型，收敛速度更快。

## Experiment

*   **有效性:** VariAntNet 在多智能体聚集任务中显著优于解析方法（Bellaiche et al.），尤其在收敛速度上。例如，在 30 个智能体、VR=0.875 的条件下，VariAntNet 加权变体平均收敛步数为 587 步，而解析方法需 1450 步，提升约 2.5 倍。
*   **权衡分析:** 解析方法保证 100% 连接性，而 VariAntNet 在高难度构型（VR=1）下存在断开风险（30 个智能体时断开比例达 17%），但在 VR≤0.875 时断开比例可控（<10%），适合时间敏感场景。
*   **实验设置合理性:** 实验覆盖 10、20、30 个智能体规模和多种难度级别（通过可见性比率 VR 定义），数据集生成确保初始构型连接性，评估了 1000 个随机环境。消融研究验证了加权损失和聚合函数对性能的影响，设置全面且严谨。

## Further Thoughts

VariAntNet 的几何预处理和旋转等变性设计为处理无序动态数据提供了新思路，可否推广至点云处理或序列无关任务？
基于可见性图拉普拉斯矩阵的凝聚力损失函数是否适用于其他图优化问题，如社交网络分析？
‘集中训练、去中心化执行’框架在多智能体强化学习中潜力巨大，能否通过引入历史轨迹提升决策鲁棒性？
接受部分智能体损失以换取速度的权衡思想，是否能启发其他资源分配策略，如分布式任务调度？