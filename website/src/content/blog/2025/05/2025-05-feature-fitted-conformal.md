---
title: "Feature Fitted Online Conformal Prediction for Deep Time Series Forecasting Model"
pubDatetime: 2025-05-13T01:33:53+00:00
slug: "2025-05-feature-fitted-conformal"
type: "arxiv"
id: "2505.08158"
score: 0.4748568161433636
author: "grok-3-latest"
authors: ["Xiannan Huang", "Shuhan Qiu"]
tags: ["Time Series Forecasting", "Conformal Prediction", "Feature Extraction", "Uncertainty Quantification", "Online Learning"]
institution: ["Tongji University"]
description: "本文提出了一种轻量级共形预测方法FFDCI，利用深度模型特征和动态调整机制，为时间序列预测提供有效且理论保证的置信区间。"
---

> **Summary:** 本文提出了一种轻量级共形预测方法FFDCI，利用深度模型特征和动态调整机制，为时间序列预测提供有效且理论保证的置信区间。 

> **Keywords:** Time Series Forecasting, Conformal Prediction, Feature Extraction, Uncertainty Quantification, Online Learning

**Authors:** Xiannan Huang, Shuhan Qiu

**Institution(s):** Tongji University


## Problem Background

时间序列预测在天气、交通、医疗等领域至关重要，但由于随机因素影响，单纯的点预测无法充分应对不确定性，因此需要置信区间来量化预测不确定性。
现有基于深度学习模型的置信区间方法存在关键局限：需要昂贵的模型重训练、未能充分利用深度模型的特征表示能力、缺乏理论上的覆盖率保证。
本文旨在提出一种轻量级方法，在不重训练模型的前提下，提供有效的置信区间，同时适应分布偏移并具备理论保证。

## Method

*   **核心思想:** 提出一种名为Feature Fitted Dynamic Confidence Interval (FFDCI) 的轻量级共形预测框架，利用预训练深度模型提取的特征来预测误差分位数，并通过动态调整机制构建置信区间，以应对时间序列中的分布偏移。
*   **具体实现步骤:**
    *   **特征提取:** 从预训练的点预测深度模型中提取特征，这些特征被视为输入数据的高维表示，包含预测相关的信息。
    *   **分位数预测模型训练:** 在验证集上，使用提取的特征作为输入，训练一个分位数预测模型（例如基于MLP），目标是预测误差的特定分位数（如90%置信区间的上下界），采用pinball loss作为损失函数。
    *   **初始置信区间构建:** 在部署阶段，结合点预测模型的输出和分位数预测模型预测的误差分位数，构建初始置信区间。
    *   **动态调整机制:** 引入在线学习策略，通过一个调整项（基于历史覆盖情况的梯度更新）动态调整置信区间长度，确保覆盖率逐步收敛到目标水平（如90%），即使面对分布偏移也能保持有效性。
*   **关键优势:** 不需要重训练原始深度模型，仅需训练一个轻量级分位数预测模型，计算开销低；同时利用深度模型的强大表示能力提升置信区间效率；理论上证明了覆盖率渐近收敛性。

## Experiment

*   **有效性:** 实验在12个时间序列数据集上进行，FFDCI在大多数数据集上实现了接近目标的覆盖率（90%），且置信区间长度较短，例如在solar数据集上比最佳基线方法缩短了20.3%。
*   **优越性:** 与多个基线方法（如ECI, TQA-E, LPCI, CF-SST）相比，FFDCI在最差维度和最差时间步的覆盖率上表现更优，表明其对多维度和多步预测的适应性更强；同时，区间长度普遍较短，效率更高。
*   **实验设置合理性:** 实验覆盖了多种深度预测模型（iTransformer, Leddam, SOFTS）和多个数据集，评估指标包括覆盖率、区间长度、最差维度和时间步覆盖率；此外，消融实验验证了动态更新和特征拟合组件的重要性；对学习率和模型参数的敏感性分析进一步增强了结果的可信度。
*   **开销:** 主要额外开销在于训练轻量级分位数预测模型和在线调整置信区间，相比重训练深度模型的成本较低。

## Further Thoughts

利用深度模型提取的特征来预测误差分位数的思想非常具有启发性，未来可以探索在其他任务（如异常检测或风险评估）中应用特征表示来量化不确定性；此外，动态调整置信区间的机制提示我们可以在其他在线学习场景中设计自适应策略，尤其是在数据分布随时间变化的领域；最后，MACE指标与特征质量的关联性启发我们进一步研究特征选择或特征工程对置信区间效率的影响。