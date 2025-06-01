---
title: "Unsupervised Word-level Quality Estimation for Machine Translation Through the Lens of Annotators (Dis)agreement"
pubDatetime: 2025-05-29T07:20:36+00:00
slug: "2025-05-unsupervised-wqe-annotation"
type: "arxiv"
id: "2505.23183"
score: 0.3622499090772838
author: "grok-3-latest"
authors: ["Gabriele Sarti", "Vilém Zouhar", "Malvina Nissim", "Arianna Bisazza"]
tags: ["Machine Translation", "Quality Estimation", "Uncertainty Quantification", "Human Variation", "Model Interpretability"]
institution: ["University of Groningen", "ETH Zurich"]
description: "本文通过评估无监督词级质量估计指标，揭示了翻译模型内部信号在错误检测中的潜力，并强调人类标注变异性对评估的影响，为机器翻译质量估计提供了新视角和实用建议。"
---

> **Summary:** 本文通过评估无监督词级质量估计指标，揭示了翻译模型内部信号在错误检测中的潜力，并强调人类标注变异性对评估的影响，为机器翻译质量估计提供了新视角和实用建议。 

> **Keywords:** Machine Translation, Quality Estimation, Uncertainty Quantification, Human Variation, Model Interpretability

**Authors:** Gabriele Sarti, Vilém Zouhar, Malvina Nissim, Arianna Bisazza

**Institution(s):** University of Groningen, ETH Zurich


## Problem Background

机器翻译（MT）的词级质量估计（WQE）旨在自动识别翻译输出中的细粒度错误片段，以辅助后期编辑等任务，但当前方法依赖昂贵的大型语言模型推理或大量人工标注数据，资源受限环境下难以应用；此外，人类标注变异性对评估指标性能的影响尚未被充分研究。

## Method

* **核心思想**：利用无监督方法从翻译模型的内部信号（如预测分布和中间激活）中提取指标，识别翻译错误，避免对人工标注数据的依赖，同时对比监督方法并改进其校准能力。
* **预测分布指标**：包括 Surprisal（预测 token 的负对数概率）和 Entropy（输出分布熵），用于量化预测不确定性；通过 Monte Carlo Dropout（MCD）计算 Surprisal 的均值（MCD AVG）和方差（MCD VAR），以估计认知不确定性。
* **词汇投影指标**：采用 LogitLens 技术，从模型每一层中间激活提取概率分布，计算层级 Surprisal（LL-Surprisal）、层间 KL 散度（LL KL-Div）和预测深度（LL Pred. Depth），分析模型对错误预测的敏感性。
* **上下文混合指标**：通过注意力权重熵（Attn. Entropy AVG/MAX）和层间变换平滑性（BLOOD）衡量上下文信息局部性，作为错误检测的辅助信号。
* **监督基准改进**：以 XCOMET（XL 和 XXL 版本）为基准，提出 XCOMET CONF，通过将错误类型概率总和作为连续置信度分数，提升校准性能。
* **实现细节**：使用 Inseq 库提取指标，在生成时强制解码以确保与标注输出一致，评估覆盖多种开源模型和数据集。

## Experiment

* **有效性**：在 DivEMT、WMT24 和 QE4PE 数据集上，基于预测分布的无监督指标（如 Surprisal MCD VAR）在错误片段识别中表现最佳，与人类标注变异性有较好相关性，但整体性能低于监督方法。
* **监督方法改进**：默认 XCOMET 模型召回率较低（仅 26%-32%），改进的 XCOMET CONF 通过置信度加权显著提升了 AP 和 F1 分数，显示出校准的重要性。
* **人类变异性影响**：在 QE4PE 数据集的多组标注评估中，指标排名受标注者主观选择影响，但多组标注可缓解此问题；Surprisal MCD VAR 在高相关性区间内优于默认 XCOMET。
* **实验设置合理性**：实验覆盖 12 个翻译方向、多种语言和领域（社交、生物医学等），数据多样性较好；但 MCD 方法计算成本高，无法在 Aya23 模型上测试，且数据集受限于开源模型可用性，泛化性有待验证。

## Further Thoughts

利用模型内部信号（如预测分布熵和 Surprisal）进行无监督错误检测的思路可扩展至其他生成任务，如文本摘要或对话生成，通过不确定性信号评估内容质量；此外，人类标注变异性的研究启发我们，未来评估框架应更多采用多参考方法以捕捉主观判断差异，而 XCOMET CONF 的校准改进提示置信度加权或可优化其他监督模型输出。