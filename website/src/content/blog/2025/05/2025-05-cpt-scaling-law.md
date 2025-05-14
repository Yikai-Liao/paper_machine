---
title: "Learning Dynamics in Continual Pre-Training for Large Language Models"
pubDatetime: 2025-05-12T17:47:32+00:00
slug: "2025-05-cpt-scaling-law"
type: "arxiv"
id: "2505.07796"
score: 0.773506214328321
author: "grok-3-latest"
authors: ["Xingjin Wang", "Howe Tissue", "Lu Wang", "Linjing Li", "Daniel Dajun Zeng"]
tags: ["LLM", "Continual Pre-Training", "Scaling Law", "Distribution Shift", "Learning Rate Annealing"]
institution: ["School of Artificial Intelligence, University of Chinese Academy of Sciences", "State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, Chinese Academy of Sciences", "Ritzz-AI"]
description: "本文提出一个 CPT 缩放法则，通过解耦分布偏移和学习率退火的影响，量化持续预训练过程中损失变化规律，并预测任意训练步骤下的性能表现，为超参数优化提供指导。"
---

> **Summary:** 本文提出一个 CPT 缩放法则，通过解耦分布偏移和学习率退火的影响，量化持续预训练过程中损失变化规律，并预测任意训练步骤下的性能表现，为超参数优化提供指导。 

> **Keywords:** LLM, Continual Pre-Training, Scaling Law, Distribution Shift, Learning Rate Annealing

**Authors:** Xingjin Wang, Howe Tissue, Lu Wang, Linjing Li, Daniel Dajun Zeng

**Institution(s):** School of Artificial Intelligence, University of Chinese Academy of Sciences, State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, Chinese Academy of Sciences, Ritzz-AI


## Problem Background

持续预训练（Continual Pre-Training, CPT）是大型语言模型（LLMs）适应特定下游领域任务的重要方法，但现有研究缺乏对 CPT 过程中学习动态（Learning Dynamics）的量化分析，尤其是在通用领域和下游领域性能随训练步骤变化的规律性描述。
论文旨在解决这一问题，通过验证损失（Validation Loss）追踪性能变化，提出一个 CPT 缩放法则（CPT Scaling Law），以预测任意训练步骤下的损失值，并优化 CPT 过程中的超参数设置。

## Method

*   **核心思想:** 提出一个 CPT 缩放法则，通过解耦分布偏移（Distribution Shift）和学习率退火（Learning Rate Annealing）的影响，量化 CPT 过程中的损失变化规律。
*   **基础损失建模:** 基于学习率退火的影响，定义基础损失函数，包含前向面积（Forward Area, S1）和退火面积（Annealing Area, S2）两个变量，分别表示学习率累积和退火效应的影响，公式为 L_base(t) = L0 + A·(S1_pt + S1_cpt)^(-α) - C·(S2_pt + S2_cpt)。
*   **分布偏移项:** 通过实验观察，分布偏移项与训练步骤或前向面积呈幂律关系（Power-Law Form），形式为 ΔL(t) = B·(1 - (1 + E·S1_cpt)^(-β))，且与转移起点无关，反映了从通用数据集到领域特定数据集的分布差异。
*   **最终转移曲线:** 将基础损失和分布偏移项结合，形成完整 CPT 损失曲线公式 L(t) = L_base(t) + ΔL(t)，能够适应不同学习率调度（Learning Rate Schedules, LRS，如常数、Warmup-Stable-Decay、Cosine）。
*   **扩展性:** 进一步将模型大小（Model Size）和数据回放比例（Replay Ratio）纳入公式，通过幂律形式和指数形式分别建模其对分布偏移和退火项的影响，增强法则的适用性。
*   **应用性:** 提供超参数优化指导，如损失潜力（Loss Potential）、峰值学习率（Peak Learning Rate）、回放比例等，基于预测损失曲线调整 CPT 策略。

## Experiment

*   **有效性:** 实验表明，CPT 缩放法则能够准确拟合和预测不同学习率调度下的损失曲线，在通用领域（FineWeb）和下游领域（Knowledge-Pile, Pile-of-Law）验证集上均表现良好。
*   **显著性:** 对于不同模型大小（106M 到 1.7B 参数）、回放比例（0% 到 100%）和学习率调度，法则均适用，预测误差较小；特别是在预测其他未见学习率调度和外部领域（Out-of-Domain, OOD）数据集损失时，表现出较强的泛化能力。
*   **合理性:** 实验设置全面，涵盖多种超参数变化（如批大小、序列长度），并对开源模型（LLaMA3.2-1B）进行了测试，验证了法则在未知 PT 信息情况下的可行性；但模型大小范围较小，未达到主流 LLMs 规模（如 7B 或 70B），可能限制结论普适性。
*   **局限性:** OOD 数据集损失预测依赖线性组合假设，可能在高度领域特定的 CPT 数据集上失效；此外，实验缺乏对更大规模模型的验证。

## Further Thoughts

论文提出的‘损失潜力’（Loss Potential）概念非常有启发性，是否可以通过动态选择预训练模型的退火状态（即不同损失潜力）来优化 CPT 效果？未来可以探索自适应策略，根据下游任务需求选择合适的 PT 模型状态；此外，分布偏移项与模型大小和转移起点无关的特性提示，是否可以通过直接度量数据集分布距离（如 KL 散度）来预测分布偏移大小，从而减少实验成本？