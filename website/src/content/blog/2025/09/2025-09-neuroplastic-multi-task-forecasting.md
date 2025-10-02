---
title: "Neuroplasticity-inspired dynamic ANNs for multi-task demand forecasting"
pubDatetime: 2025-09-29T09:08:08+00:00
slug: "2025-09-neuroplastic-multi-task-forecasting"
type: "arxiv"
id: "2509.24495"
score: 0.5824393197291234
author: "grok-3-latest"
authors: ["Mateusz Zarski", "Sławomir Nowaczyk"]
tags: ["Dynamic Neural Networks", "Multi-Task Learning", "Demand Forecasting", "Neuroplasticity", "Task Similarity"]
institution: ["Institute of Theoretical and Applied Informatics, Polish Academy of Sciences", "School of Information Technology, Halmstad University"]
description: "本文提出受神经可塑性启发的 Neuroplastic Multi-Task Network (NMT-Net)，通过动态计算图调整和任务相似性识别，显著提升了多任务需求预测的性能和一致性。"
---

> **Summary:** 本文提出受神经可塑性启发的 Neuroplastic Multi-Task Network (NMT-Net)，通过动态计算图调整和任务相似性识别，显著提升了多任务需求预测的性能和一致性。 

> **Keywords:** Dynamic Neural Networks, Multi-Task Learning, Demand Forecasting, Neuroplasticity, Task Similarity

**Authors:** Mateusz Zarski, Sławomir Nowaczyk

**Institution(s):** Institute of Theoretical and Applied Informatics, Polish Academy of Sciences, School of Information Technology, Halmstad University


## Problem Background

在多任务需求预测中，传统方法面临高维数据异质性与计算成本的矛盾：合并数据到一个模型会导致性能下降，而独立处理每个任务则计算成本过高且忽略共享信息；动态人工神经网络（D-ANNs）目前主要关注推理时动态或效率，而缺乏结构适应性，难以平衡多任务学习中的共享知识与任务特异性。

## Method

* **核心思想**：提出 Neuroplastic Multi-Task Network (NMT-Net)，受生物神经可塑性启发，通过动态调整计算图来适应新任务，实现多任务学习中共享信息与任务特异性的平衡。
* **具体机制**：
  * **相似任务识别**：基于新任务与已有任务特征向量的根均方误差（RMSE），识别最相似的已有任务，以便复用相关知识。
  * **临时头部训练**：为新任务训练两个临时网络头部，一个从预训练通用权重开始，仅在新任务数据上训练；另一个从最相似任务的头部权重开始，在联合数据（新任务与相似任务数据）上训练，以探索最佳性能并避免灾难性遗忘。
  * **头部性能评估与选择**：在新任务的评估数据集上比较两个临时头部的损失，选择性能更优的头部保存到模型中，释放较差头部以节省资源；这种动态头部管理减少了模型头部数量，提升了效率。
* **特点**：方法不依赖超参数调整，强调结构适应性，支持多任务和持续学习场景，同时通过任务分组优化资源分配。

## Experiment

* **有效性**：在三个需求预测数据集（DF, SIDF, PDF）上，NMT-Net 在 DF 和 SIDF 上取得了最低均值 RMSE，在 PDF 上接近最佳（TAG 方法略优）；在最小和最大 RMSE 指标上也表现优异，尤其最大 RMSE 显著低于其他方法，表明最差情况下的稳定性更强。
* **一致性**：NMT-Net 的标准差（σ）远低于其他方法，通常低一个数量级，表明其性能在多次运行中非常稳定。
* **实验设置**：实验设置全面，涵盖多个数据集和多种基线方法（ARIMA, Decision Trees, Random Forests, MTL, MTWR, TAG, MTL-cluster），每个实验至少运行五次以计算标准差；数据集划分和输入向量设计合理，符合需求预测实际需求。
* **局限性**：实验未探讨不同 ANN 架构（如 RNN 或 Transformer）对性能的影响，也未深入分析训练阶段长度或学习率调度等超参数的优化潜力；PDF 数据集上部分传统方法在最小 RMSE 上优于 NMT-Net，提示对某些数据分布的适应性有待提升。

## Further Thoughts

NMT-Net 受神经可塑性启发的动态结构调整思路令人印象深刻，未来是否可以扩展到更深层次网络结构（如中间层），实现更全面的‘可塑性’？任务相似性识别是否可引入模型权重或梯度信息作为度量，以更直接反映学习相关性？此外，这种动态头部选择机制是否能结合元学习或在线学习，快速适应新任务，甚至预测任务相似性？作者提到的强化学习应用前景也值得探索，动态适应性在序列决策任务中可能有更大潜力。