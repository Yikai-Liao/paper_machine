---
title: "Latent Collective Preference Optimization: A General Framework for Robust LLM Alignment"
pubDatetime: 2025-09-29T01:17:49+00:00
slug: "2025-09-lcpo-robust-alignment"
type: "arxiv"
id: "2509.24159"
score: 0.7765352558134222
author: "grok-3-latest"
authors: ["Xiaoyang Cao", "Zelai Xu", "Mo Guang", "Kaiwen Long", "Michiel A. Bakker", "Yu Wang", "Chao Yu"]
tags: ["LLM", "Preference Optimization", "Noise Handling", "Alignment", "Expectation-Maximization"]
institution: ["Massachusetts Institute of Technology", "Tsinghua University", "Li Auto Inc.", "Zhongguancun Academy"]
description: "本文提出 Latent Collective Preference Optimization (LCPO) 框架，通过 EM 算法从噪声偏好数据中推断集体共识，显著提升多种 LLM 对齐方法的鲁棒性和性能。"
---

> **Summary:** 本文提出 Latent Collective Preference Optimization (LCPO) 框架，通过 EM 算法从噪声偏好数据中推断集体共识，显著提升多种 LLM 对齐方法的鲁棒性和性能。 

> **Keywords:** LLM, Preference Optimization, Noise Handling, Alignment, Expectation-Maximization

**Authors:** Xiaoyang Cao, Zelai Xu, Mo Guang, Kaiwen Long, Michiel A. Bakker, Yu Wang, Chao Yu

**Institution(s):** Massachusetts Institute of Technology, Tsinghua University, Li Auto Inc., Zhongguancun Academy


## Problem Background

大型语言模型 (LLM) 的对齐方法（如 RLHF 和 DPO）依赖于人类偏好数据，但这些方法假设数据是同质且无噪声的，而现实中人类偏好具有多元性，标注错误普遍存在（噪声率高达 20%-40%），导致模型性能下降；本文旨在解决这一问题，通过学习潜在的集体偏好共识来实现鲁棒对齐。

## Method

* **核心思想**：提出 Latent Collective Preference Optimization (LCPO)，一个基于期望-最大化 (EM) 算法的框架，通过将真实偏好视为潜变量，从噪声数据中推断集体共识，并动态调整数据点对训练损失的贡献。
* **具体步骤**：
  * **E-Step (期望步)**：根据当前模型参数和标注者可靠性，计算每个偏好标签正确的后验概率，作为置信度分数 (confidence score)，即软标签。
  * **M-Step (最大化步)**：利用置信度分数作为自适应权重，更新模型参数 (通过加权损失函数优化) 和标注者可靠性参数 (通过平均置信度或指数移动平均更新)。
* **通用性设计**：LCPO 是一个元框架，通过 Gibbs 分布将任意偏好损失函数 (如 DPO、IPO、SimPO、CPO) 转化为概率模型，使其能增强多种对齐方法的鲁棒性。
* **实现优化**：采用小批量训练和指数移动平均 (EMA) 更新标注者可靠性参数，以提高计算效率并适应大规模数据集。
* **关键优势**：不依赖硬标签，而是动态调整权重以减轻噪声影响，同时保持对现有方法的兼容性。

## Experiment

* **有效性**：在 Mistral-7B 和 Llama-3-8B 模型上，LCPO 增强的四种对齐方法 (DPO、IPO、SimPO、CPO) 在 AlpacaEval 2 和 Arena-Hard 基准测试中均显著优于原始方法，胜率提升最高达 7.0% (AlpacaEval 2) 和 5.4% (Arena-Hard)。
* **全面性**：实验覆盖了不同模型、算法和数据集，验证了 LCPO 的通用性和鲁棒性；此外，消融研究分析了初始可靠性参数和 EMA 动量参数的影响，表明 LCPO 在合理参数范围内表现稳定。
* **理论验证**：论文证明了在理想条件下 LCPO 能收敛到真实噪声水平，实验通过模拟噪声数据进一步验证了这一结论。
* **观察**：提升效果在更强大模型 (Llama-3-8B) 上更显著，且不同算法受益程度不同 (DPO 和 IPO 提升更大)，可能与其损失函数对噪声的敏感性有关。

## Further Thoughts

LCPO 的潜变量建模和自适应权重机制启发我们，不仅在 LLM 对齐中，也可以在其他机器学习任务 (如图像分类或推荐系统) 中应用类似方法处理噪声数据；此外，LCPO 作为元框架的通用性提示未来对齐方法可以设计为模块化结构，灵活集成不同损失函数或去噪策略；进一步思考，是否可以引入动态可靠性模型以适应标注者随时间或任务难度的变化？