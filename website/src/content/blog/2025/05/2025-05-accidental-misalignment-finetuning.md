---
title: "Accidental Misalignment: Fine-Tuning Language Models Induces Unexpected Vulnerability"
pubDatetime: 2025-05-22T15:30:00+00:00
slug: "2025-05-accidental-misalignment-finetuning"
type: "arxiv"
id: "2505.16789"
score: 0.6157712040592102
author: "grok-3-latest"
authors: ["Punya Syon Pandey", "Samuel Simko", "Kellin Pelrine", "Zhijing Jin"]
tags: ["LLM", "Fine-Tuning", "Adversarial Attack", "Dataset Design", "Model Alignment"]
institution: ["University of Toronto", "ETH Zurich", "Max Planck Institute for Intelligent Systems", "McGill University", "FAR AI", "Vector Institute", "MILA"]
description: "本文通过实证分析揭示了微调数据集特性（如毒性、Token数量）与大型语言模型对抗性脆弱性的关联，并通过特征干预实验验证了因果关系，为安全数据集设计提供了重要见解。"
---

> **Summary:** 本文通过实证分析揭示了微调数据集特性（如毒性、Token数量）与大型语言模型对抗性脆弱性的关联，并通过特征干预实验验证了因果关系，为安全数据集设计提供了重要见解。 

> **Keywords:** LLM, Fine-Tuning, Adversarial Attack, Dataset Design, Model Alignment

**Authors:** Punya Syon Pandey, Samuel Simko, Kellin Pelrine, Zhijing Jin

**Institution(s):** University of Toronto, ETH Zurich, Max Planck Institute for Intelligent Systems, McGill University, FAR AI, Vector Institute, MILA


## Problem Background

大型语言模型（LLMs）在公开应用中越来越受欢迎，但其对抗性攻击的脆弱性成为主要安全隐患。
微调（fine-tuning）作为提升模型在特定任务上性能的常见手段，可能意外引入不对齐（misalignment），削弱模型的安全机制（如抵抗‘越狱’攻击的能力）。
现有研究较少关注微调数据集的具体特性如何影响模型脆弱性，论文旨在填补这一空白，探索数据集特征与对抗性攻击成功率（ASR）之间的关系。

## Method

*   **研究目标与框架:** 论文通过系统性实证方法，研究微调数据集特性对模型对抗性脆弱性的影响，核心问题是‘哪些数据集特征会增加微调后模型的对抗性脆弱性’。
*   **数据集选择:** 选择了六种多样化的问答格式数据集，包括良性数据集（如Alpaca）、有害数据集（如LLM-LAT、Gray-Swan Circuit Breaking）以及领域特定数据集（如法律、cybersecurity、电气工程），以对比不同数据特性对模型的影响。
*   **微调设置:** 使用LLaMA 3.1 8B Instruct模型，通过低秩适应（LoRA）方法进行微调，采用AdamW优化器、5e-5学习率、批大小2等超参数配置，并以交叉熵损失作为早停指标，确保模型拟合一致性。
*   **对抗性评估:** 采用HarmBench框架中的三种攻击技术：贪婪坐标梯度（GCG）、AutoPrompt和PEZ，覆盖多种越狱攻击方式，评估攻击成功率（ASR）；同时将攻击提示分为五个有害行为类别（犯罪、毒品/有害化学品、版权、cybercrime、操纵）以分析细粒度脆弱性。
*   **特征分析与干预:** 提取数据集的多维度特征，包括语义相似性（余弦相似性、欧几里得距离、KL散度）、情感分数（Sentiment Score）、毒性分数（Toxicity Score）、可读性（Flesch-Kincaid分数）和词汇多样性（Type-Token Ratio, TTR）等；通过Spearman相关性分析探索特征与ASR的关系；进一步通过特征干预实验（移除高毒性或高Token数量等特征的数据子集）验证因果关联。
*   **通用性能评估:** 使用Massive Multitask Language Understanding（MMLU）基准测试模型通用能力，确保脆弱性并非由灾难性遗忘引起。

## Experiment

*   **对抗性脆弱性结果:** 实验表明，领域特定数据集（如法律、cybersecurity）微调后的模型在对抗性攻击下的成功率（ASR）显著高于原始模型（从18.8%提升至21.7%），而有害数据集的ASR提升更为明显（高达42.5%-61.7%），表明微调确实可能引入脆弱性。
*   **特征相关性分析:** Spearman相关性分析显示，响应中的Token数量（r=0.714, p<0.001）、提示和响应的毒性（r=0.708, p<0.001; r=0.701, p<0.001）与ASR呈强正相关，而提示情感（r=-0.664, p<0.01）与ASR呈负相关，提示数据集特征可能直接影响模型安全性。
*   **特征干预效果:** 通过移除高毒性、高Token数量等特征的数据子集，ASR有所下降（如在Cybersecurity数据集上下降5.56%，在CB Harmful数据集上最大下降14.89%），初步验证了特征与脆弱性的因果关联，但效果有限。
*   **通用性能稳定性:** MMLU基准测试显示，微调后模型通用能力基本保持稳定（性能变化在-2.1%到+0.4%之间），排除了灾难性遗忘作为脆弱性来源的可能性。
*   **实验设置合理性与局限:** 实验设置较为全面，涵盖多种数据集、攻击技术和特征维度，且通过MMLU验证了模型通用性；但主要基于单一模型（LLaMA 3.1 8B），跨模型验证有限（仅测试了Qwen 2.5 7B和Falcon 7B，且仅用PEZ攻击），结果普适性有待进一步验证；此外，特征干预采用单变量消融，可能忽略了特征间的交互影响。

## Further Thoughts

论文揭示了微调数据集特性与模型脆弱性的关联，启发我思考是否可以通过‘数据预筛选’或‘对抗性数据增强’来设计更安全的微调数据集，例如在微调前对数据进行毒性检测和情感分析，优先剔除高毒性或负面情感样本；此外，是否可以借鉴对抗性训练思路，在微调数据中引入安全对齐相关的提示或对抗性样本，以增强模型对越狱攻击的抵抗力；特征干预实验效果有限，是否可以通过多特征联合干预或动态数据采样策略进一步提升模型鲁棒性？