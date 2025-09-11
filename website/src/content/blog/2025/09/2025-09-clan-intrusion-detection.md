---
title: "Contrastive Self-Supervised Network Intrusion Detection using Augmented Negative Pairs"
pubDatetime: 2025-09-08T11:04:10+00:00
slug: "2025-09-clan-intrusion-detection"
type: "arxiv"
id: "2509.06550"
score: 0.5421855148819007
author: "grok-3-latest"
authors: ["Jack Wilkie", "Hanan Hindy", "Christos Tachtatzis", "Robert Atkinson"]
tags: ["Self-Supervised Learning", "Contrastive Learning", "Network Intrusion Detection", "Anomaly Detection", "Data Augmentation"]
institution: ["University of Strathclyde", "Ain Shams University"]
description: "本文提出 CLAN 框架，通过将增强样本视为负样本学习良性流量的整体分布，显著提升了网络入侵检测的性能和推理效率。"
---

> **Summary:** 本文提出 CLAN 框架，通过将增强样本视为负样本学习良性流量的整体分布，显著提升了网络入侵检测的性能和推理效率。 

> **Keywords:** Self-Supervised Learning, Contrastive Learning, Network Intrusion Detection, Anomaly Detection, Data Augmentation

**Authors:** Jack Wilkie, Hanan Hindy, Christos Tachtatzis, Robert Atkinson

**Institution(s):** University of Strathclyde, Ain Shams University


## Problem Background

网络入侵检测系统（NIDS）在网络安全中至关重要，但传统监督学习方法依赖大量标注数据，而这些数据在现实中难以获取，尤其是在新网络环境或面对零日攻击时；异常检测方法虽可仅基于良性流量训练，但高误报率限制了实用性；自监督学习（SSL）虽有潜力，但现有方法在建模良性流量整体分布和区分恶意流量方面仍显不足。
本文旨在改进自监督学习在 NIDS 中的应用，解决标注数据稀缺和高误报率问题，同时提升推理效率以适应大规模网络环境。

## Method

*   **核心思想:** 提出 Contrastive Learning using Augmented Negative Pairs (CLAN) 框架，与传统对比学习不同，CLAN 将数据增强后的样本视为负样本（代表潜在恶意分布），而将其他原始良性样本视为正样本，旨在学习良性流量的整体分布而非个体样本分布。
*   **实现细节:**
    *   使用神经网络（MLP 架构）将输入特征映射到潜在空间，假设良性流量在潜在空间中服从高斯分布，具有统一的协方差。
    *   设计 CLAN 损失函数，通过最小化正样本（良性样本）之间的距离，同时最大化正样本与增强负样本之间的距离，优化潜在空间表征；增强负样本通过对原始样本特征进行均匀重采样生成。
    *   引入铰链正则化项（hinge regularization）以控制正负样本距离的边界，并可选使用余弦距离替代欧几里得距离以优化 von Mises-Fisher 分布假设。
    *   在推理阶段，预计算良性流量分布的中心点（centroid），通过测试样本与中心的距离概率判断其是否属于良性分布，计算复杂度为 O(1)。
*   **关键优势:** 不依赖恶意样本标注，训练时仅需良性流量；推理效率高，适合大规模网络监控；学习整体分布而非个体分布，提升了表征的泛化能力。

## Experiment

*   **数据集与设置:** 实验基于 Lycos2017 数据集，包含多种网络攻击类别，数据不平衡严重；实验包括二分类（异常检测）和有限数据微调后的多分类任务，采用分层采样划分训练和测试集，并通过交叉验证优化超参数。
*   **二分类性能:** CLAN 在 AUROC 指标上显著优于现有自监督学习方法（如 CLDNN, SSCL-IDS）和异常检测方法（如 Autoencoder, Deep SVDD），均值 AUROC 达 0.958591，领先第二名约 0.031，表明其在区分良性与恶意流量方面的优越性。
*   **多分类性能:** 在有限标注数据微调后，CLAN 的宏平均 F1 分数在大多数训练集规模下优于基线方法，表明其预训练表征对下游任务有较强适应性。
*   **推理效率:** CLAN 推理复杂度为 O(1)，远低于传统 SSL 方法的 O(N_train)，在大规模网络环境中具有显著优势。
*   **合理性与局限:** 实验设置全面，涵盖多种基线和任务场景，数据划分考虑类别不平衡，合理性较高；但未充分讨论训练数据中混杂恶意样本时的鲁棒性，可能影响实际应用效果。

## Further Thoughts

CLAN 将增强样本视为负样本以学习整体类分布的思路非常新颖，启发我在其他领域（如图像或文本分类）中思考是否可应用类似反向增强策略，尤其在数据分布单一或类别不平衡场景下；此外，CLAN 假设训练数据均为良性流量，但现实中可能存在噪声，是否可通过对抗训练等机制提升鲁棒性？其高效推理设计（O(1) 复杂度）也提示我们可以在其他高吞吐量任务中探索中心点表征方法。