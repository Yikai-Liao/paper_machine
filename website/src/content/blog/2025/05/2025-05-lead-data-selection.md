---
title: "LEAD: Iterative Data Selection for Efficient LLM Instruction Tuning"
pubDatetime: 2025-05-12T10:57:51+00:00
slug: "2025-05-lead-data-selection"
type: "arxiv"
id: "2505.07437"
score: 0.5277643749646701
author: "grok-3-latest"
authors: ["Yizhang Zhu", "Xiaotian Lin", "Yanlin Qi", "Themis Palpanas", "Chengliang Chai", "Nan Tang", "Yuyu Luo"]
tags: ["LLM", "Instruction Tuning", "Data Selection", "Uncertainty Estimation", "Efficiency"]
institution: ["Hong Kong University of Science and Technology (Guangzhou)", "Université Paris Cité", "Beijing Institute of Technology"]
description: "LEAD 提出了一种高效的迭代数据选择框架，通过无推理的实例级动态不确定性（IDU）效用估计和粗到精选择策略，使用仅 2.5% 数据显著提升 LLM 指令调优性能并缩短训练时间 5-10 倍。"
---

> **Summary:** LEAD 提出了一种高效的迭代数据选择框架，通过无推理的实例级动态不确定性（IDU）效用估计和粗到精选择策略，使用仅 2.5% 数据显著提升 LLM 指令调优性能并缩短训练时间 5-10 倍。 

> **Keywords:** LLM, Instruction Tuning, Data Selection, Uncertainty Estimation, Efficiency

**Authors:** Yizhang Zhu, Xiaotian Lin, Yanlin Qi, Themis Palpanas, Chengliang Chai, Nan Tang, Yuyu Luo

**Institution(s):** Hong Kong University of Science and Technology (Guangzhou), Université Paris Cité, Beijing Institute of Technology


## Problem Background

大型语言模型（LLM）的指令调优（Instruction Tuning）是提升模型性能和对齐能力的关键方法，但传统迭代式模型感知数据选择方法依赖于每轮训练后对整个数据集进行模型推理以评估样本效用，导致显著的计算开销。
论文旨在解决这一效率瓶颈，提出一个核心问题：能否在不进行额外推理的情况下，利用标准训练过程中的已有信号实现高效的迭代数据选择，同时保持模型性能提升和数据选择的适应性？

## Method

*   **核心思想：** LEAD 是一个无推理（Inference-Free）的迭代数据选择框架，通过利用训练过程中的自然信号（如损失和梯度）来估计样本效用，避免传统方法中昂贵的额外推理步骤，同时动态适应模型的训练状态。
*   **具体实现：**
    *   **实例级动态不确定性（IDU）：** 设计了一种新型效用函数，结合三部分信号：(1) 当前训练损失，反映样本对当前模型的难度；(2) 基于梯度的损失变化近似，利用历史梯度信息预测参数更新后的损失变化，解决时间错配问题；(3) 历史损失的指数平滑，减少随机噪声和训练不稳定性对效用估计的影响。IDU 完全基于训练过程中的已有计算，无需额外推理。
    *   **粗到精选择策略：** 首先离线将数据集基于指令难度和任务相似性进行双层聚类，形成粗粒度和细粒度簇；在线训练时，通过多臂老虎机（MAB）机制动态选择高回报的粗粒度簇（基于历史效用增益），然后在选定簇内利用 IDU 进行细粒度样本选择，确保高效处理大规模数据集。
    *   **理论优化：** 通过拉格朗日优化和互补松弛条件推导出 IDU 的最优平滑参数和梯度近似权重，确保效用估计的理论严谨性和实际效果。
*   **关键特点：** LEAD 无需修改模型架构或训练流程，计算开销仅限于标准训练信号的处理和 MAB 的轻量级更新，实现了效率与效果的平衡。

## Experiment

*   **性能提升：** 在四个基准数据集（MMLU、TYDIQA、GSM8K、HumanEval）上，LEAD 使用仅 2.5% 的训练数据，平均性能提升 6.1%-10.8%，在某些任务（如 TYDIQA）上提升高达 29.15%（Mistral-7B），超越全数据集训练和多种基线方法（如 IFD、PPL、SelectIT）。
*   **效率优势：** 相比传统方法，LEAD 将整体训练时间缩短 5-10 倍，推理时间从 98 小时（IFD 方法）降至 10.3 小时，仅需初始推理，后续通过 IDU 实现无推理效用估计。
*   **实验设置合理性：** 数据池包含 60 万样本，任务覆盖代码生成、数学推理、多任务知识和跨语言问答，测试了三种代表性模型（LLaMA3.1-8B、Mistral-7B、Qwen2-7B），对比了多种基线方法；消融实验验证了 IDU、MAB 和任务聚类（TC）各组件的贡献，IDU 对性能影响最大（去除后平均下降 3.27%）。
*   **潜在局限：** 实验未深入探讨 IDU 在不同训练阶段的稳定性表现，以及 MAB 在极不平衡簇分布下的适应性，可能需进一步验证。

## Further Thoughts

IDU 的动态效用估计理念（结合当前信号、历史趋势和预测变化）是否可推广至其他机器学习任务，如图像分类或强化学习的数据选择，形成通用高效训练范式？此外，MAB 机制在动态选择簇上的成功应用，是否可扩展至模型架构选择或超参数调优，动态调整学习率或层数？最后，论文揭示的‘对齐适配数据’有限性，是否能结合主动学习动态生成新数据，进一步探索数据质量与数量的最佳平衡点？