---
title: "Multiple Descents in Deep Learning as a Sequence of Order-Chaos Transitions"
pubDatetime: 2025-05-26T14:18:22+00:00
slug: "2025-05-multiple-descents-transitions"
type: "arxiv"
id: "2505.20030"
score: 0.5902724190798592
author: "grok-3-latest"
authors: ["Wei Wenbo", "Nicholas Chong Jia Le", "Lai Choy Heng", "Feng Ling"]
tags: ["Deep Learning", "Training Dynamics", "Overfitting", "Phase Transition", "Stability Analysis"]
institution: ["National University of Singapore, Department of Physics", "Institute of High Performance Computing, A*STAR"]
description: "本文通过渐近稳定性分析揭示了LSTM训练中的‘多重下降’现象与秩序-混沌相变的关联，指出全局最优epoch位于首次相变点，为深度学习训练动态提供了新理论视角。"
---

> **Summary:** 本文通过渐近稳定性分析揭示了LSTM训练中的‘多重下降’现象与秩序-混沌相变的关联，指出全局最优epoch位于首次相变点，为深度学习训练动态提供了新理论视角。 

> **Keywords:** Deep Learning, Training Dynamics, Overfitting, Phase Transition, Stability Analysis

**Authors:** Wei Wenbo, Nicholas Chong Jia Le, Lai Choy Heng, Feng Ling

**Institution(s):** National University of Singapore, Department of Physics, Institute of High Performance Computing, A*STAR


## Problem Background

深度学习模型的训练动态对性能、泛化能力和鲁棒性至关重要，而传统研究（如双重下降现象）无法完全解释过拟合阶段测试损失的复杂波动模式。
本文聚焦于一种新现象——‘多重下降’（Multiple Descents），即在长短期记忆网络（LSTM）过拟合阶段，测试损失呈现多次上升和急剧下降的周期性波动，旨在揭示其背后的机制并探索与模型动态稳定性（秩序与混沌相变）的关系，为优化训练时机提供理论依据。

## Method

*   **核心思想：** 通过渐近稳定性分析（Asymptotic Stability Analysis）研究LSTM训练过程中的秩序与混沌状态，揭示测试损失的多重下降与相变之间的关联。
*   **具体实现：**
    *   在LSTM输出单元引入微小扰动，计算扰动后输出的渐近轨迹差异（Asymptotic Distance），以此判断模型处于秩序（收敛）还是混沌（发散）状态。
    *   在每个训练epoch后，基于测试数据集的500个样本计算平均渐近距离的对数值（ln值），用于量化模型的稳定性状态。
    *   结合降维可视化技术（如输出向量的降维和），进一步确认秩序/混沌状态，并与测试损失波动进行关联分析。
    *   将LSTM训练动态与非线性动力系统中的tanh映射（tanh map）类比，解释其分叉到混沌的相图结构，强调首次秩序-混沌转变点的特殊性。
*   **实验设计：** 使用Large Movie Review Dataset进行情感分析，过训练LSTM模型至10000个epoch以诱发过拟合，观察多重下降现象。
*   **创新点：** 将深度学习训练动态与动力系统理论结合，提出多重下降与秩序-混沌相变的直接对应关系，为训练时机优化提供新视角。

## Experiment

*   **有效性：** 实验清晰展示了多重下降现象，在过拟合阶段（约500个epoch后），测试损失呈现8个明显的上升-下降周期，每个周期末尾损失急剧下降，且下降点与秩序-混沌相变点一致。
*   **最优epoch：** 全局最优epoch（测试损失最低）出现在首次秩序到混沌的转变点（epoch 114，准确率88.34%），此时‘混沌边缘’宽度最宽，模型权重配置探索能力最强。
*   **合理性：** 通过多次随机种子实验验证了结果一致性，尽管最优epoch位置略有变化，但始终与首次相变点重合；此外，过拟合阶段损失与混沌程度的正相关性进一步支持结论。
*   **局限性：** 由于计算成本，渐近距离计算仅基于500个测试样本，可能影响统计显著性；过拟合阶段的部分相变不够明显，可能是采样分辨率（每epoch一次）不足所致。
*   **结论：** 实验设置较为全面，数据可视化直观支持了多重下降与相变的关系，但样本量和采样频率的限制可能导致细节丢失。

## Further Thoughts

论文揭示的最优epoch与首次秩序-混沌转变点的关系，启发可以在训练中引入动态监测机制，实时评估模型稳定性状态，在‘混沌边缘’处停止训练以提升泛化能力；
此外，LSTM训练动态与tanh映射的相似性表明，非线性动力系统理论（如分叉理论）可能为理解其他深度学习模型的训练行为提供新工具；
未来可尝试在不同架构（如Transformer）或任务上验证多重下降现象的普适性，并结合数据集特性进一步分析相变机制。