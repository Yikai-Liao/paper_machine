---
title: "Adversarial Cooperative Rationalization: The Risk of Spurious Correlations in Even Clean Datasets"
pubDatetime: 2025-05-04T14:00:04+00:00
slug: "2025-05-adversarial-rationalization"
type: "arxiv"
id: "2505.02118"
score: 0.4764684373415036
author: "grok-3-latest"
authors: ["Wei Liu", "Zhongyu Niu", "Lang Gao", "Zhiying Deng", "Jun Wang", "Haozhao Wang", "Ruixuan Li"]
tags: ["LLM", "Rationalization", "Spurious Correlations", "Adversarial Attack", "Interpretability"]
institution: ["Huazhong University of Science and Technology", "Central China Normal University", "iWudao Tech"]
description: "本文提出 A2I 方法，通过对抗性攻击检测并纠正自解释合理化框架中模型引入的虚假相关性，显著提升 Rationale 质量。"
---

> **Summary:** 本文提出 A2I 方法，通过对抗性攻击检测并纠正自解释合理化框架中模型引入的虚假相关性，显著提升 Rationale 质量。 

> **Keywords:** LLM, Rationalization, Spurious Correlations, Adversarial Attack, Interpretability

**Authors:** Wei Liu, Zhongyu Niu, Lang Gao, Zhiying Deng, Jun Wang, Haozhao Wang, Ruixuan Li

**Institution(s):** Huazhong University of Science and Technology, Central China Normal University, iWudao Tech


## Problem Background

自解释框架中的合理化（Rationalization）方法，如 Rationalizing Neural Predictions (RNP)，通过生成器和预测器的合作博弈从输入中提取关键信息（Rationale）进行预测。然而，即使在干净的数据集上，生成器的采样过程可能引入虚假相关性（Spurious Correlations），导致预测器依赖与标签语义无关的特征，损害模型的可解释性和可靠性。

## Method

*   **核心思想:** 提出一种名为 Attack to Inspection and Instruction (A2I) 的方法，通过对抗性攻击检测并纠正生成器引入的虚假相关性，提升 Rationale 提取质量。
*   **具体实现:** 
    *   **攻击作为检查（Inspection）:** 引入一个攻击生成器（Attacker），其目标是从输入中提取一个攻击 Rationale（Z_A），使得预测器对 Z_A 的预测结果与真实标签相反。若攻击成功率高，说明预测器学习了虚假相关性。
    *   **攻击作为指导（Instruction）:** 通过调整预测器的训练目标，强制其对 Z_A 进行随机分类（即不偏向任何标签），避免学习虚假相关性。这一过程通过损失函数实现，确保预测器不受虚假模式影响。
    *   **训练流程:** 交替训练生成器-预测器对和攻击器，确保生成器接收到正确的反馈，逐步优化 Rationale 选择。
*   **关键特点:** 方法模型无关，可应用于不同架构（如 GRUs、BERT、GCN），且不依赖数据集固有因果关系，而是聚焦模型自身引入的偏差。

## Experiment

*   **有效性:** A2I 方法在六个文本分类数据集和两个图分类数据集上显著提升了 Rationale 质量（以 F1 分数衡量），例如在 Beer-Appearance 数据集上，相比标准 RNP，F1 分数提升高达 9.0%；与先进方法 FR 结合后，性能进一步提高，显示出普适性。
*   **攻击成功率（ASR）:** 未加入指导损失时，ASR 高达 95%，证明预测器确实学习了虚假相关性；加入指导后，ASR 降至约 50%（接近随机分类），表明预测器不再依赖虚假模式。
*   **对比分析:** 与大型语言模型（如 llama-3.1-8b-instruct）相比，A2I 在多个场景下表现相当甚至更优，尤其在低稀疏度设置下。
*   **实验设置合理性:** 实验覆盖多种数据类型（文本和图）、不同稀疏度（10%、20%、30%）和多种架构（GRUs、BERT、GCN），设置全面且具代表性；但未深入探讨大规模数据集上的计算开销。

## Further Thoughts

论文揭示的模型引入虚假相关性问题启发我思考：是否其他自解释方法（如注意力机制）也存在类似偏差？对抗性攻击是否可作为通用工具检测各类模型偏差？此外，合理化作为数据清洗手段，是否可用于提取高质量子集，高效微调大型语言模型，降低计算成本并提升可控性？