---
title: "Bridging Generative and Discriminative Learning: Few-Shot Relation Extraction via Two-Stage Knowledge-Guided Pre-training"
pubDatetime: 2025-05-18T05:17:36+00:00
slug: "2025-05-few-shot-re-tkre"
type: "arxiv"
id: "2505.12236"
score: 0.39453295431725427
author: "grok-3-latest"
authors: ["Quanjiang Guo", "Jinchuan Zhang", "Sijie Wang", "Ling Tian", "Zhao Kang", "Bin Yan", "Weidong Xiao"]
tags: ["LLM", "Synthetic Data", "Relation Extraction", "Pre-Training", "Contrastive Learning"]
institution: ["University of Electronic Science and Technology of China", "Nanyang Technological University", "Information Engineering University", "National University of Defense Technology"]
description: "本文提出 TKRE 框架，通过两阶段知识引导预训练结合 LLMs 生成的解释驱动知识和合成数据，显著提升了少样本关系抽取性能，成功桥接了生成式与判别式学习范式。"
---

> **Summary:** 本文提出 TKRE 框架，通过两阶段知识引导预训练结合 LLMs 生成的解释驱动知识和合成数据，显著提升了少样本关系抽取性能，成功桥接了生成式与判别式学习范式。 

> **Keywords:** LLM, Synthetic Data, Relation Extraction, Pre-Training, Contrastive Learning

**Authors:** Quanjiang Guo, Jinchuan Zhang, Sijie Wang, Ling Tian, Zhao Kang, Bin Yan, Weidong Xiao

**Institution(s):** University of Electronic Science and Technology of China, Nanyang Technological University, Information Engineering University, National University of Defense Technology


## Problem Background

少样本关系抽取（Few-Shot Relation Extraction, FSRE）由于标注数据稀缺和模型泛化能力有限而面临巨大挑战，尤其是在现实场景（如罕见疾病关系挖掘或金融事件检测）中，传统模型依赖大规模标注数据，而现有方法难以适应跨领域或细粒度关系；此外，大型语言模型（LLMs）虽通过上下文学习（In-Context Learning）展现潜力，但其通用训练目标导致在任务特定关系抽取中的性能不佳，生成式与判别式学习范式之间存在脱节。

## Method

* **核心思想**：提出 TKRE（Two-Stage Knowledge-Guided Pre-training for Relation Extraction）框架，通过整合大型语言模型（LLMs）的生成能力与传统关系抽取模型的判别能力，解决数据稀缺和泛化能力不足的问题。
* **数据准备**：
  * **解释驱动的知识生成（Explanation-Driven Knowledge Generation）**：设计特定指令引导 LLMs 生成关系逻辑解释（如‘主体 X 与对象 Y 具有关系 R 因为...’），为下游模型提供可解释的推理路径，增强对关系语义的理解。
  * **模式约束的合成数据生成（Schema-Constrained Data Generation）**：通过预定义规则和模式约束（如实体类型兼容性和关系特定句法模式），利用 LLMs 生成符合任务结构的训练样本，缓解数据稀缺并减少分布偏移。
* **两阶段预训练**：
  * **第一阶段 - 掩码跨度语言建模（Masked Span Language Modeling, MSLM）**：扩展传统掩码语言建模，通过选择性掩码关系指示跨度并优化其重建，增强模型对上下文依赖和关系语义的捕捉能力；具体策略包括根据语言角色（如关系跨度、实体跨度）设置不同掩码概率（80%、50%、20%），并通过交叉熵损失训练模型预测掩码跨度。
  * **第二阶段 - 跨度级对比学习（Span-Level Contrastive Learning, SCL）**：在对比学习框架内，将与目标关系语义一致的正跨度与不一致但上下文合理的负跨度进行对比，通过优化对比损失（基于余弦相似度和温度参数）提升模型对实体交互的判别能力，强化关系结构的理解。
* **任务导向微调**：在预训练后，使用少量标注数据和合成数据对模型进行微调，通过分类层预测实体间关系，适应具体的 FSRE 任务；损失函数为交叉熵损失，确保模型在小样本数据上的性能优化。
* **关键点**：TKRE 通过知识引导和结构化数据生成弥补数据不足，同时利用两阶段预训练提升模型的泛化性和关系推理能力，形成生成式与判别式范式的闭环协同。

## Experiment

* **有效性**：TKRE 在四个基准数据集（SemEval, TACRED, TACREV, Re-TACRED）上的实验表明，其在 8-shot、16-shot 和 32-shot 少样本设置下均显著优于传统方法、纯 LLM 方法和混合方法；例如，基于 GenPT 架构的 TKRE 平均 F1 分数提升了 5.0%，基于 TYP Marker 提升了 7.8%，尤其在数据极少（8-shot）时表现突出。
* **优越性**：相比纯 LLM 方法（如 GPT-4），TKRE 避免了生成式模型的黑箱问题和结构化目标缺失问题；相比传统方法，TKRE 通过合成数据和知识引导显著提升了性能，尤其在细粒度关系区分上更具优势。
* **实验设置合理性**：实验覆盖多个数据集和样本量设置，验证了方法的鲁棒性；消融研究进一步确认了各组件（如解释驱动知识生成、模式约束数据生成、MSLM、SCL）的贡献，例如移除解释驱动知识生成后性能显著下降，尤其在小样本设置下。
* **局限性与分析**：实验显示合成数据量过多可能引入噪声，导致性能下降，提示未来需优化数据生成质量；此外，TKRE 在 Re-TACRED 数据集上表现更优，可能是由于其更精确的标签设计对模型学习有利。

## Further Thoughts

TKRE 框架通过 LLMs 生成解释驱动知识和合成数据，为低资源任务提供了生成式与判别式范式结合的新思路，这种方法是否可以扩展到其他 NLP 任务（如命名实体识别或事件抽取），甚至零样本场景，通过更复杂的逻辑推理减少对标注数据的依赖？此外，合成数据质量对性能的影响提示我们，是否可以引入强化学习（RL）或自动质量评估机制，动态优化生成数据，减少噪声干扰？