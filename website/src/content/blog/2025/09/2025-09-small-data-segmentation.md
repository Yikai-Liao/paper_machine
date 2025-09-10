---
title: "Challenges in Deep Learning-Based Small Organ Segmentation: A Benchmarking Perspective for Medical Research with Limited Datasets"
pubDatetime: 2025-09-07T01:54:20+00:00
slug: "2025-09-small-data-segmentation"
type: "arxiv"
id: "2509.05892"
score: 0.6589665365400406
author: "grok-3-latest"
authors: ["Phongsakon Mark Konrad", "Andrei-Alexandru Popa", "Yaser Sabzehmeidani", "Liang Zhong", "Elisa A. Liehn", "Serkan Ayvaz"]
tags: ["Medical Image Segmentation", "Small Dataset Evaluation", "Statistical Stability", "Benchmarking", "Explainable AI"]
institution: ["Centre for Industrial Software, University of Southern Denmark", "Centre for Industrial Mechanics, University of Southern Denmark", "Duke-NUS, Singapore", "National Heart Center Singapore"]
description: "本文通过在极小数据集上的系统性评估，揭示了传统基准测试在医学AI小数据场景中的不稳定性，倡导从追求单一最佳模型转向报告统计稳定性和基于实用性的模型选择。"
---

> **Summary:** 本文通过在极小数据集上的系统性评估，揭示了传统基准测试在医学AI小数据场景中的不稳定性，倡导从追求单一最佳模型转向报告统计稳定性和基于实用性的模型选择。 

> **Keywords:** Medical Image Segmentation, Small Dataset Evaluation, Statistical Stability, Benchmarking, Explainable AI

**Authors:** Phongsakon Mark Konrad, Andrei-Alexandru Popa, Yaser Sabzehmeidani, Liang Zhong, Elisa A. Liehn, Serkan Ayvaz

**Institution(s):** Centre for Industrial Software, University of Southern Denmark, Centre for Industrial Mechanics, University of Southern Denmark, Duke-NUS, Singapore, National Heart Center Singapore


## Problem Background

在心血管病理学研究中，标注好的组织病理图像数据极为稀缺（本文数据集仅9张图像），这对深度学习模型的训练和评估构成了巨大挑战。
论文旨在解决小数据集场景下模型选择和评估的可靠性问题，质疑传统基准测试方法是否因数据分割的随机性和统计噪声导致模型排名不稳定，从而可能误导临床决策。

## Method

*   **核心框架:** 设计了一个系统性的评估框架，旨在公平比较多种深度学习模型在小数据集上的表现，并揭示评估过程中的统计不稳定性。
*   **数据处理:** 针对不同模型需求调整输入分辨率（CNNs 使用 256x256，基础模型使用 1024x1024），并通过在线数据增强（包括几何变换如旋转、翻转和颜色抖动）扩充有限数据集以提高泛化能力。
*   **模型选择:** 涵盖三种架构范式：传统卷积神经网络（U-Net, DeepLabV3+），基于Transformer的模型（SegFormer），以及大型基础模型（SAM, MedSAM, MedSAM+UNet 混合模型），确保评估的多样性和代表性。
*   **超参数优化:** 采用贝叶斯优化方法，对每个模型进行广泛的超参数搜索（总计1000次运行），覆盖学习率、批大小、优化器、损失函数等参数，确保各模型达到近乎最优配置，避免因配置不当导致的性能偏差。
*   **评估策略:** 使用两种交叉验证策略（Leave-One-Out Cross-Validation, LOOCV 和 3-Fold CV）评估模型性能，探索不同评估协议对结果稳定性的影响；主要评估指标为宏平均 Dice 相似系数（DSC）和交并比（IoU）。
*   **统计分析:** 运用非参数方法（如 Friedman 检验和 Nemenyi 后验检验）比较模型排名，结合 Bootstrap 置信区间量化性能不确定性，并通过效应量（Cohen’s d）评估差异的实际意义。
*   **可解释性分析:** 设计五层解释框架（XAI），包括错误分析、不确定性估计、形态分析、类别关注和梯度显著性，用于诊断模型预测的稳定性来源，提供临床信任基础。

## Experiment

*   **性能表现:** 在 LOOCV 下，SegFormer 以宏平均 Dice 分数 0.821 表现最佳；在 3-Fold CV 下，DeepLabV3+ 和 SegFormer 均表现突出；然而，排名随评估协议变化，表明‘最佳模型’并非固定属性。
*   **稳定性分析:** Bootstrap 置信区间显示顶级模型性能高度重叠，Friedman 和 Nemenyi 检验表明在 3-Fold CV 下无统计显著差异（p=0.221），证明性能差异更多是统计噪声而非算法优越性。
*   **折间波动:** LOOCV 排名波动极大，表明高方差评估对单样本敏感；3-Fold CV 排名相对稳定但可能引入偏差。
*   **能力悖论:** 尽管量化指标不稳定，顶级模型的分割掩码视觉上高度相似且接近真实值，表明指标噪声主要来自边界微小分歧，临床相关性可能有限。
*   **实验设置:** 实验设计全面，涵盖多种模型、评估协议和统计分析，数据增强和超参数优化减少了配置偏差；但数据集极小（N=9）为故意极端设置，可能不完全代表稍大数据量场景。

## Further Thoughts

论文揭示了小数据场景中传统基准测试的不稳定性，启发我重新思考评估目标，不应盲目追求排行榜，而是关注统计稳定性和实际部署价值；此外，量化指标与视觉结果的矛盾提示需探索更贴近临床需求的评估方法，如基于关键区域错误容忍度的指标；最后，研究数据量与稳定性的‘相变’边界是一个重要方向，可通过在不同规模数据集上重复实验，为数据稀缺领域提供指导。