---
title: "Mirror Mirror on the Wall, Have I Forgotten it All? A New Framework for Evaluating Machine Unlearning"
pubDatetime: 2025-05-13T00:23:17+00:00
slug: "2025-05-computational-unlearning"
type: "arxiv"
id: "2505.08138"
score: 0.40810950894123443
author: "grok-3-latest"
authors: ["Brennon Brimhall", "Philip Mathew", "Neil Fendley", "Yinzhi Cao", "Matthew Green"]
tags: ["Machine Unlearning", "Differential Privacy", "Indistinguishability", "Model Evaluation", "Privacy Protection"]
institution: ["Johns Hopkins University", "Johns Hopkins University Applied Physics Laboratory"]
description: "本文提出计算遗忘框架，通过不可区分性评估机器遗忘方法，揭示现有方法在强攻击模型下的局限性，并为隐私保护和模型安全提供了新的理论和实证视角。"
---

> **Summary:** 本文提出计算遗忘框架，通过不可区分性评估机器遗忘方法，揭示现有方法在强攻击模型下的局限性，并为隐私保护和模型安全提供了新的理论和实证视角。 

> **Keywords:** Machine Unlearning, Differential Privacy, Indistinguishability, Model Evaluation, Privacy Protection

**Authors:** Brennon Brimhall, Philip Mathew, Neil Fendley, Yinzhi Cao, Matthew Green

**Institution(s):** Johns Hopkins University, Johns Hopkins University Applied Physics Laboratory


## Problem Background

随着隐私法规（如欧盟 GDPR 的‘被遗忘权’）和数据安全需求的增加，机器遗忘（Machine Unlearning）成为一个重要研究领域，旨在从已训练的机器学习模型中移除特定数据（遗忘集）的影响，使模型表现得仿佛从未见过这些数据，同时避免从头重新训练的高昂成本。
现有遗忘方法存在缺陷，攻击者可以通过区分遗忘后模型与对照模型（从未见过遗忘数据的模型）来推断遗忘数据信息，造成隐私泄露和安全风险。

## Method

*   **核心框架：计算遗忘（Computational Unlearning）**：提出一个基于不可区分性（Indistinguishability）的评估框架，通过一个安全博弈（Security Game）定义遗忘有效性——如果攻击者无法以高于随机猜测的概率区分遗忘后模型（Unlearned Model）和对照模型（Mirror Model），则认为遗忘方法达到计算遗忘。
*   **区分算法设计**：设计两种评分方法来测试现有遗忘方法的有效性：
    *   **MIAScore**：基于成员推理攻击（Membership Inference Attack），利用遗忘后模型在遗忘集上的异常表现来区分模型，认为遗忘模型的 MIA 分数应与对照模型相似，而非过度最小化。
    *   **KLDScore**：基于 Kullback-Leibler 散度（KL Divergence），计算原始模型与候选模型在遗忘集附近数据上的推理输出差异，发现遗忘模型通常与原始模型的散度较小，从而可区分。
*   **理论分析**：从理论上探讨计算遗忘的局限性，证明对于熵性学习算法（如随机梯度下降），不存在确定性计算遗忘方法，必须引入随机性；同时分析基于差分隐私（Differential Privacy）的遗忘方法会导致实用性崩溃（Utility Collapse）。
*   **适用范围**：框架适用于白盒（攻击者可访问模型参数）和黑盒（攻击者仅能通过 API 查询）两种场景，强调遗忘方法需在强攻击模型下保持不可区分性。

## Experiment

*   **测试对象与设置**：在 CIFAR-10 数据集上，使用 ResNet-18 模型测试了多种遗忘方法，包括启发式方法（Bad Teacher、Amnesiac、Selective Synaptic Dampening (SSD)）和近似方法（Certified Deep Unlearning, CDU）。实验设置包括不同遗忘集大小（10到1000）和 CDU 的噪声参数 σ 变化，运行 128 次试验以确保统计显著性。
*   **区分效果**：所有测试方法均未能达到计算遗忘，攻击者使用 MIAScore 和 KLDScore 的区分成功率显著高于 50%，在类级遗忘（Classwise Unlearning）时甚至达到 100%。遗忘集越大，区分成功率越高。
*   **参数影响**：对于 CDU，噪声参数 σ 增加时 KLDScore 也增加，但仍无法避免区分，表明现有方法在参数调整下仍易被识别。
*   **合理性与局限**：实验设置在图像分类任务上较为标准，但未涉及更复杂的模型（如大型语言模型）或跨领域任务，可能存在推广性问题；此外，区分成功率的高低与遗忘集分布相关，提示遗忘难度可能因任务而异。

## Further Thoughts

计算遗忘框架的不可区分性理念可以扩展到生成模型（如大型语言模型 LLMs）的对齐（Alignment）任务中，通过遗忘特定内容来减少模型输出中的有害信息，但如何精确定义‘遗忘内容’（如语义相似性而非简单关键词）是一个挑战；此外，是否可以通过结合联邦学习或数据匿名化技术，在严格的隐私保护和模型实用性之间找到平衡，值得进一步探索。