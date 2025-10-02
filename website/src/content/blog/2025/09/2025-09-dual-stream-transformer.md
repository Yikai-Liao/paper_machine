---
title: "DSAT-HD: Dual-Stream Adaptive Transformer with Hybrid Decomposition for Multivariate Time Series Forecasting"
pubDatetime: 2025-09-29T13:50:56+00:00
slug: "2025-09-dual-stream-transformer"
type: "arxiv"
id: "2509.24800"
score: 0.5315089160454763
author: "grok-3-latest"
authors: ["Zixu Wang", "Hongbin Dong", "Xiaoping Zhang"]
tags: ["Time Series Forecasting", "Transformer", "Decomposition", "Multi-scale Modeling", "Adaptive Routing"]
institution: ["Harbin Engineering University", "Traditional Chinese Medicine Data Center, China Academy of Chinese Medical Sciences"]
description: "本文提出 DSAT-HD 框架，通过混合分解、多尺度自适应路径和双流残差学习，显著提升了多变量时间序列预测的精度和泛化能力。"
---

> **Summary:** 本文提出 DSAT-HD 框架，通过混合分解、多尺度自适应路径和双流残差学习，显著提升了多变量时间序列预测的精度和泛化能力。 

> **Keywords:** Time Series Forecasting, Transformer, Decomposition, Multi-scale Modeling, Adaptive Routing

**Authors:** Zixu Wang, Hongbin Dong, Xiaoping Zhang

**Institution(s):** Harbin Engineering University, Traditional Chinese Medicine Data Center, China Academy of Chinese Medical Sciences


## Problem Background

多变量时间序列预测在天气、交通、电力等领域至关重要，但现有基于 Transformer 的方法在处理非平稳时间序列时面临挑战，包括难以有效分离和建模季节性与趋势性成分的动态交互、捕捉多尺度特征的能力有限，以及长序列处理的计算效率瓶颈。

## Method

* **核心思想**：提出 DSAT-HD 框架，通过混合分解、多尺度自适应路径和双流残差学习，端到端地处理时间序列的非平稳性和多尺度特性，同时提升计算效率。
* **混合分解模块**：结合指数移动平均（EMA）在时域捕捉趋势、傅里叶分解在频域提取全局周期特征，以及多尺度移动平均通过不同大小卷积核捕捉多尺度特征；通过噪声 Top-k 门控机制动态平衡季节性和趋势性成分。
* **多尺度自适应路径**：利用稀疏分配器（Sparse Dispatcher）将输入特征动态路由到四个具有不同 patch 大小的 Transformer 层，每层专注于特定时间尺度模式；结合局部 CNN 注意力与全局 Transformer 交互，通过稀疏组合器（Sparse Combiner）融合跨尺度特征。
* **双流残差学习框架**：设计两个并行分支，CNN 流通过分层卷积结构处理季节性成分，MLP 流通过全连接层捕捉趋势变化；引入平衡损失函数（Balance Loss）协调专家协作，减少协作方差。
* **效率优化**：通过稀疏机制减少计算复杂度，适应长序列预测任务。

## Experiment

* **有效性**：DSAT-HD 在九个公开数据集（如 ETTh1、ETTh2、Weather、Electricity）上，针对不同预测窗口（96、192、336、720），在 MSE 和 MAE 指标上显著优于基线模型（如 PatchTST、Autoformer、FEDformer），尤其在长预测窗口表现突出。
* **全面性与合理性**：实验覆盖多种领域和时间粒度的数据集，测试了不同输入长度（48、192、336）的影响，显示模型能有效利用长序列信息；消融实验验证了混合分解、多尺度路径和双流框架的必要性，移除任一模块均导致性能下降。
* **局限性**：尽管性能提升明显，论文未提供详细计算开销数据，仅提及稀疏机制提升效率；在某些高噪声数据集（如 Traffic）上性能提升不显著，可能与数据特性有关。

## Further Thoughts

DSAT-HD 的混合分解机制（时域+频域）可扩展至其他信号处理任务，如音频或图像特征分离；多尺度自适应路径中的动态路由机制启发在自然语言处理中根据语义复杂度分配注意力资源；双流框架的专家分支设计可进一步引入更多类型模型（如 RNN 或 GNN），以适应更复杂数据模式。