---
title: "Exploring Scaling Laws for EHR Foundation Models"
pubDatetime: 2025-05-29T01:05:11+00:00
slug: "2025-05-ehr-scaling-laws"
type: "arxiv"
id: "2505.22964"
score: 0.5288247644507841
author: "grok-3-latest"
authors: ["Sheng Zhang", "Qin Liu", "Naoto Usuyama", "Cliff Wong", "Tristan Naumann", "Hoifung Poon"]
tags: ["EHR Foundation Model", "Scaling Laws", "Transformer", "Clinical Prediction", "Sequential Modeling"]
institution: ["Microsoft Research", "University of Southern California"]
description: "本文首次系统探索了电子健康记录（EHR）基础模型的缩放规律，通过Transformer架构和MIMIC-IV数据验证了可预测的性能提升模式，为资源高效训练和临床预测任务提供了指导。"
---

> **Summary:** 本文首次系统探索了电子健康记录（EHR）基础模型的缩放规律，通过Transformer架构和MIMIC-IV数据验证了可预测的性能提升模式，为资源高效训练和临床预测任务提供了指导。 

> **Keywords:** EHR Foundation Model, Scaling Laws, Transformer, Clinical Prediction, Sequential Modeling

**Authors:** Sheng Zhang, Qin Liu, Naoto Usuyama, Cliff Wong, Tristan Naumann, Hoifung Poon

**Institution(s):** Microsoft Research, University of Southern California


## Problem Background

大型语言模型（LLMs）通过增加模型规模、数据集和计算资源，性能可以预测性地提升，这种缩放规律（Scaling Laws）已被广泛研究。然而，电子健康记录（EHR）作为一种结构化、时序性强且全球范围内丰富的医疗数据，其缩放规律尚未被系统性探索。EHR数据与自然语言文本有相似之处（如时序性），但也存在显著差异（如结构化编码、隐私限制）。本研究旨在探索EHR基础模型是否遵循类似的缩放规律，以解决资源高效训练和临床预测任务中的关键问题。

## Method

* **核心思想：** 使用Transformer架构从头训练EHR基础模型，系统性研究模型规模、训练数据量和计算预算对性能的影响，探索EHR领域的缩放规律。
* **数据处理：** 基于MIMIC-IV数据库，构建患者时间线（Patient Timeline），将临床事件（如诊断、用药）转化为离散令牌（Tokens），并采用自回归建模方式预测下一个令牌。改进ETHOS方法，确保训练批次不跨患者时间线，防止信息泄露。
* **模型架构：** 采用Llama架构的解码器专用Transformer（Decoder-Only Transformer），不使用预训练权重，从头初始化，确保实验控制变量。模型规模从1M到近1B参数不等，通过调整深度、宽度和注意力配置生成变体。
* **实验设计：** 以浮点运算（FLOPs）为计算预算指标，构建IsoFLOP曲线，分析固定计算预算下模型规模与验证损失的关系。推导计算预算、最优模型参数和训练令牌数之间的幂律关系。
* **训练与评估：** 使用标准优化器和学习率调度（如余弦衰减、预热），通过验证损失评估模型性能，并在下游临床任务（如ICU死亡率预测）上进行零样本评估。

## Experiment

* **有效性：** 实验表明EHR基础模型遵循类似LLM的缩放规律，IsoFLOP曲线呈抛物线形状，验证了固定计算预算下存在最优模型规模。幂律关系显示最优模型参数和训练令牌数随计算预算增加分别按0.58和0.44的指数增长。
* **性能提升：** 在下游任务（如ICU死亡率和30天再入院预测）中，验证损失与任务性能呈强线性相关，模型规模从1M到28M参数时性能持续提升（如ICU死亡率ROC AUC从0.68提升至0.75）。
* **局限性：** 超过28M参数后，由于MIMIC-IV数据量不足（约2.67亿令牌），性能趋于饱和，显示出过拟合迹象，表明数据量需与模型规模匹配。
* **实验设置：** 实验覆盖多种模型规模和计算预算，设置较为全面，但受限于单一数据集，未能探索更大规模EHR数据的效果。

## Further Thoughts

论文揭示了Transformer架构对EHR等非自然语言时序数据的普适性，启发我们探索其他领域（如金融、物联网）数据的缩放规律。此外，验证损失作为下游任务性能的强代理指标，提示在资源有限时可优先优化通用指标，而非针对具体任务微调。数据量不足导致性能饱和的问题，启发未来研究跨机构数据协作或合成数据生成技术，以扩展EHR数据集规模。