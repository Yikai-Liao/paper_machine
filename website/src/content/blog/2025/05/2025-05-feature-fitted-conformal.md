---
title: "Feature Fitted Online Conformal Prediction for Deep Time Series Forecasting Model"
pubDatetime: 2025-05-13T01:33:53+00:00
slug: "2025-05-feature-fitted-conformal"
type: "arxiv"
id: "2505.08158"
score: 0.4748568161433636
author: "grok-3-latest"
authors: ["Xiannan Huang", "Shuhan Qiu"]
tags: ["Time Series Forecasting", "Conformal Prediction", "Uncertainty Quantification", "Feature Extraction", "Dynamic Adjustment"]
institution: ["Tongji University"]
description: "本文提出 Feature Fitted Dynamic Confidence Interval (FFDCI)，通过深度学习特征拟合误差分位数和动态调整机制，为时间序列预测提供有效且高效的在线置信区间，无需重训练模型。"
---

> **Summary:** 本文提出 Feature Fitted Dynamic Confidence Interval (FFDCI)，通过深度学习特征拟合误差分位数和动态调整机制，为时间序列预测提供有效且高效的在线置信区间，无需重训练模型。 

> **Keywords:** Time Series Forecasting, Conformal Prediction, Uncertainty Quantification, Feature Extraction, Dynamic Adjustment

**Authors:** Xiannan Huang, Shuhan Qiu

**Institution(s):** Tongji University


## Problem Background

时间序列预测在天气、交通、医疗等领域至关重要，深度学习模型在点预测方面表现出色，但实际应用中需要量化预测不确定性（通过置信区间），尤其是在分布偏移可能导致传统方法失效的情况下。
现有方法存在局限：需要昂贵的模型重训练、未能充分利用深度学习模型的特征表示能力、缺乏理论覆盖保证。
本文旨在为深度学习时间序列预测模型提供有效的在线置信区间，既保证覆盖率（validity），又尽量缩短区间长度（efficiency），且无需重训练模型。

## Method

*   **核心思想:** 提出 Feature Fitted Dynamic Confidence Interval (FFDCI)，一种基于保形预测（Conformal Prediction）的轻量级框架，利用预训练深度学习模型提取的特征来预测误差分位数，并通过动态调整机制应对分布偏移。
*   **具体实现:**
    *   **特征提取与分位数预测:** 从预训练点预测模型中提取特征（如历史数据的表示），在验证集上训练一个分位数预测模型（quantile prediction model，通常为 MLP），以预测误差的特定分位数（如 90% 置信水平对应的上下界）。训练使用 pinball loss 优化，确保分位数估计准确。
    *   **动态置信区间构建:** 在部署阶段，根据点预测模型的输出和分位数预测模型估计的误差分位数，初步构建置信区间；同时引入动态调整项（adjustment term），通过在线梯度下降更新（基于过去覆盖情况），以适应分布偏移，确保长期覆盖率收敛到目标水平。
    *   **理论支持:** 证明了方法在温和假设下的渐近覆盖率收敛性，并分析了覆盖误差度量（Mean Absolute Coverage Error, MACE）的上界与特征质量相关。
*   **关键优势:** 无需重训练原始深度学习模型，仅需训练一个轻量级分位数预测模型，计算开销低；利用深度学习特征实现样本自适应置信区间；动态调整机制有效应对时间序列中的分布偏移。

## Experiment

*   **有效性:** 在12个时间序列数据集上，FFDCI 在大多数情况下实现了目标覆盖率（90%），如 weather 数据集覆盖率为89.3%，solar 数据集为89.8%，且置信区间长度较基线方法显著缩短，例如 solar 数据集上减少了20.3%，weather 数据集上减少了7.4%。
*   **优越性:** 相比基线方法（如 ECI, TQA-E, LPCI, CF-SST），FFDCI 在最差维度和最差预测步骤的覆盖率上表现更优，显示出鲁棒性；消融实验表明动态更新确保覆盖率，特征拟合显著缩短区间长度。
*   **实验设置:** 实验基于三种深度学习点预测模型（iTransformer, Leddam, SOFTS），涵盖多变量多步预测场景，数据集和基线选择合理，评价指标包括覆盖率（Coverage）、区间长度（Length）及最差维度/步骤覆盖率，设置全面。
*   **局限性:** 在 ETTh2 数据集上覆盖率略低（87.3%），对学习率 *γ* 敏感，需进一步优化参数调整策略；计算开销主要来自分位数预测模型的前向推理和动态更新，相对轻量但仍有优化空间。

## Further Thoughts

FFDCI 利用深度学习特征预测误差分位数的思路启发性很强，传统保形预测通常不考虑样本特定信息，而特征驱动的自适应置信区间可推广到其他任务（如分类、回归）的不确定性量化中；此外，动态调整机制在处理分布偏移方面的潜力值得探索，未来可结合更复杂的在线学习策略（如专家聚合或自适应学习率）提升性能；另一个方向是是否可以引入领域知识（如时间序列周期性）或多模型特征融合，进一步优化特征提取和分位数预测的准确性。