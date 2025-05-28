---
title: "REACT: Representation Extraction And Controllable Tuning to Overcome Overfitting in LLM Knowledge Editing"
pubDatetime: 2025-05-25T01:57:06+00:00
slug: "2025-05-react-knowledge-editing"
type: "arxiv"
id: "2505.18933"
score: 0.6897004518819689
author: "grok-3-latest"
authors: ["Haitian Zhong", "Yuhuan Liu", "Ziyang Xu", "Guofan Liu", "Qiang Liu", "Shu Wu", "Zhe Zhao", "Liang Wang", "Tieniu Tan"]
tags: ["LLM", "Knowledge Editing", "Overfitting", "Representation Learning", "Hidden States"]
institution: ["Institute of Automation, Chinese Academy of Sciences", "Lanzhou University", "The Chinese University of Hong Kong", "Tencent"]
description: "本文提出 REACT 框架，通过提取事实表示和可控扰动隐藏状态，显著减少大型语言模型知识编辑中的过拟合问题，同时保持基本编辑性能。"
---

> **Summary:** 本文提出 REACT 框架，通过提取事实表示和可控扰动隐藏状态，显著减少大型语言模型知识编辑中的过拟合问题，同时保持基本编辑性能。 

> **Keywords:** LLM, Knowledge Editing, Overfitting, Representation Learning, Hidden States

**Authors:** Haitian Zhong, Yuhuan Liu, Ziyang Xu, Guofan Liu, Qiang Liu, Shu Wu, Zhe Zhao, Liang Wang, Tieniu Tan

**Institution(s):** Institute of Automation, Chinese Academy of Sciences, Lanzhou University, The Chinese University of Hong Kong, Tencent


## Problem Background

大型语言模型（LLMs）在知识编辑过程中常遭遇过拟合问题，即更新事实后模型过度偏向编辑目标，即使在上下文不相关时也优先输出更新后的内容，导致复杂推理或无关查询中的表现下降。
论文旨在解决这一关键问题，确保模型在精准更新事实的同时，维持对非目标知识的完整性和上下文适应能力。

## Method

* **核心思想：** 提出 REACT（Representation Extraction And Controllable Tuning），一个双阶段框架，通过提取潜在事实表示并对隐藏状态进行可控扰动，实现精准知识编辑并减少过拟合。
* **第一阶段 - 提取潜在知识表示：** 使用定制的正负样本提示（Stimuli）提取模型在编辑前后的潜在事实表示；通过主成分分析（PCA）降维，并结合可学习的线性变换，计算每个编辑实例的‘信念偏移’向量（Belief Shift Vector），该向量表征事实更新在表示空间中的方向性变化。
* **第二阶段 - 可控表示扰动：** 利用预训练分类器作为门控机制，判断当前上下文是否需要编辑（概率阈值0.5）；仅在必要时，通过信念偏移向量和可学习标量对模型隐藏状态施加扰动，控制扰动的方向和幅度，确保编辑仅影响相关上下文。
* **优化目标：** 设计编辑损失（Editing Loss）和局部性损失（Localization Loss）联合优化，确保模型在更新事实的同时最小化对无关内容的干扰。
* **关键创新：** 不直接修改模型权重，而是通过动态扰动隐藏状态实现编辑；分类器和标量提供精细控制，避免过拟合。

## Experiment

* **有效性：** 在 COUNTERFACT 和 MQuAKE 数据集上，REACT 在可靠性（Reliability）、泛化性（Generality）、局部性（Locality）和可移植性（Portability）指标上表现平衡，平均得分比第二好基线高出至少 20 个百分点。
* **过拟合减少：** 在 EVOKE 数据集上，REACT 的直接概率（DP）显著低于基线，表明其有效避免了过度偏向编辑目标；编辑过拟合分数（EOS）和答案修改分数（AMS）较高，输出质量得以保持；正确答案概率（CAP）适中，反映谨慎编辑策略。
* **实验设置：** 使用 Llama3.1-8B 和 Qwen2.5-7B 模型，与多种基线（Fine-Tuning, MEND, MEMIT 等）对比，覆盖多个基准数据集（COUNTERFACT, MQuAKE, EVOKE）；评估指标全面，消融实验验证了方法参数（如刺激向量数量 N=512 和 PCA 使用）的合理性。
* **计算开销：** 主要开销来自预训练分类器和 REACT 框架的训练（分别为 12 和 40 GPU 小时），但推理时仅需对隐藏状态进行动态扰动，相对高效。

## Further Thoughts

REACT 的隐藏状态扰动思路启发了我，是否可以通过更复杂的表示学习方法捕捉事实更新的多维度变化，或者引入外部决策模块（如分类器）动态控制模型行为，扩展到安全对齐或个性化输出等领域；此外，结合多模态数据（如文本+图像）增强表示丰富性可能是一个有趣的方向。