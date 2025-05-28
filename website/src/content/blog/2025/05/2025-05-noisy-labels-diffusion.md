---
title: "Calibrating Pre-trained Language Classifiers on LLM-generated Noisy Labels via Iterative Refinement"
pubDatetime: 2025-05-26T08:31:55+00:00
slug: "2025-05-noisy-labels-diffusion"
type: "arxiv"
id: "2505.19675"
score: 0.4691171959406303
author: "grok-3-latest"
authors: ["Liqin Ye", "Agam Shah", "Chao Zhang", "Sudheer Chava"]
tags: ["LLM", "Noisy Labels", "Diffusion Model", "Embedding Space", "Classification"]
institution: ["Georgia Institute of Technology"]
description: "本文提出 SiDyP 框架，通过动态标签候选检索和单纯形扩散模型迭代精炼噪声标签，显著提升预训练语言分类器在 LLM 生成噪声标签上的性能。"
---

> **Summary:** 本文提出 SiDyP 框架，通过动态标签候选检索和单纯形扩散模型迭代精炼噪声标签，显著提升预训练语言分类器在 LLM 生成噪声标签上的性能。 

> **Keywords:** LLM, Noisy Labels, Diffusion Model, Embedding Space, Classification

**Authors:** Liqin Ye, Agam Shah, Chao Zhang, Sudheer Chava

**Institution(s):** Georgia Institute of Technology


## Problem Background

大型语言模型（LLMs）为自然语言处理（NLP）任务提供了自动生成标注数据的可能性，相比传统的手工标注方法更加高效且成本低廉。然而，LLM 生成的标签往往包含噪声（即错误标签），这会导致深度神经网络（DNNs）在训练时过拟合噪声，损害模型的泛化能力。尽管学习噪声标签（Learning from Noisy Labels）领域已有较多研究，但针对 LLM 生成噪声的研究较少，而这种噪声具有独特的上下文相关性，与合成噪声和现实世界噪声有显著区别。因此，本文致力于增强预训练语言分类器（PLCs）对 LLM 生成噪声标签的鲁棒性，解决模型在噪声数据上训练时性能下降的关键问题。

## Method

* **核心思想**：提出 SiDyP（Simplex Label Diffusion with Dynamic Prior）框架，通过迭代精炼噪声标签来校准预训练语言分类器的预测，增强其对 LLM 生成噪声的鲁棒性。
* **阶段一：预训练分类器微调**：
  * 使用 BERT 作为预训练语言分类器（PLC），在 LLM 生成的噪声标签数据集上进行微调。
  * 记录训练动态（Training Dynamics），即嵌入空间中的训练轨迹，用于后续区分干净样本和噪声样本。
* **阶段二：噪声标签去噪**：
  * **标签候选检索（Label Candidate Retrieval）**：基于嵌入空间中的邻域标签分布，利用 K-Nearest Neighbor (KNN) 算法为每个样本检索一组可能的真实标签候选，而非单一确定标签。通过概率阈值（certain threshold 和 dominant threshold）将样本分为确定性样本（关联单一高概率标签）和不确定性样本（关联多个候选标签）。
  * **候选标签精炼（Candidate Distillation）**：针对不确定性样本，设计动态蒸馏算法，在训练过程中根据生成模型的预测反馈更新候选标签的权重，并基于权重多重采样可能的真实标签用于当前轮训练，减少不确定性对模型训练的干扰。
  * **单纯形去噪扩散模型（Simplex Denoising Label Diffusion Model）**：采用单纯形扩散模型在概率空间中操作，将真实标签推断视为从噪声标签到真实标签的迭代去噪过程。模型通过逐步添加和去除高斯噪声，重建真实标签的后验概率分布，确保中间向量始终是有效的概率分布。
* **辅助机制**：
  * 利用训练动态信息增强生成模型的学习能力。
  * 引入多分支协同正则化（Co-Regularization），通过多个相同架构但初始化不同的模型分支计算共识概率，并用 KL 散度作为损失促使各分支预测一致，减少噪声影响。
* **关键创新**：针对 LLM 噪声的上下文相关性，SiDyP 结合动态标签候选和单纯形扩散模型，允许样本关联多个可能标签并迭代精炼，而非传统方法中假设单一真实标签。

## Experiment

* **有效性**：SiDyP 在多个 NLP 任务（NumClaim, TREC, SemEval, 20News）和多种 LLM（Llama-3-70b, Llama-3.1-70B, Llama-3.1-405B, GPT-4o, Mixtral-8x22B）生成的噪声标签数据集上显著优于基线方法（Co-Teaching, JoCoR, NPC, DyGen），在零样本和少样本设置下分别平均提升分类准确率 7.21% 和 7.30%。尤其在噪声比例较高的 SemEval 数据集上，性能提升最为显著（平均 3.7%）。
* **鲁棒性**：SiDyP 对不同类型和家族的 LLM 噪声表现出一致的鲁棒性，相比第二好的基线平均提升 4.47%，相比直接微调的 PLC 提升 8.02%，相比 LLM 原始标注准确率提升 11.73%。
* **对比其他噪声类型**：在合成噪声（Symmetric Noise, Asymmetric Noise, Instance-Dependent Noise）和现实世界噪声上的实验表明，SiDyP 同样优于基线，但对 LLM 噪声的提升幅度更大（5.21% vs 3.26%），验证了 LLM 噪声的复杂性及 SiDyP 的针对性优势。
* **实验设置合理性**：实验覆盖多种任务、多种 LLM、多种噪声类型，并通过 5 个随机种子确保结果稳定性。消融实验验证了各组件的有效性：动态标签候选检索比固定先验提升 9.5% 正确标签比例，候选蒸馏纠正 16.95% 不确定样本，单纯形扩散模型比其他生成模型提升 2.17% 至 8.58%。唯一局限是部分数据集（如 20News）因上下文长度限制仅测试零样本设置，但整体设计全面合理。

## Further Thoughts

论文揭示了 LLM 生成噪声的上下文相关性，提示去噪方法需结合语义嵌入空间而非仅依赖统计模式；动态标签候选检索和迭代精炼的‘软标签’思想可推广至其他噪声学习或半监督学习场景；单纯形扩散模型在概率空间中的应用启发探索其他分布建模工具（如流模型）或多模态联合去噪；此外，LLM 噪声比例和上下文限制的影响提示未来可优化提示策略或数据选择机制以减少噪声。