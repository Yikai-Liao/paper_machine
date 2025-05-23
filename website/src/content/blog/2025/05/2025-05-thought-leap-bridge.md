---
title: "Mind the Gap: Bridging Thought Leap for Improved Chain-of-Thought Tuning"
pubDatetime: 2025-05-20T17:59:31+00:00
slug: "2025-05-thought-leap-bridge"
type: "arxiv"
id: "2505.14684"
score: 0.7774084695623201
author: "grok-3-latest"
authors: ["Haolei Xu", "Yuchen Yan", "Yongliang Shen", "Wenqi Zhang", "Guiyang Hou", "Shengpei Jiang", "Kaitao Song", "Weiming Lu", "Jun Xiao", "Yueting Zhuang"]
tags: ["LLM", "Chain of Thought", "Reasoning", "Data Augmentation", "Mathematical Reasoning"]
institution: ["Zhejiang University", "The Chinese University of Hong Kong", "Microsoft Research Asia"]
description: "本文提出CoT Thought Leap Bridge Task，通过构建ScaleQM+数据集和CoT-Bridge模型，自动检测并填补链式推理中的思维跳跃，显著提升数学推理性能并展现跨领域泛化能力。"
---

> **Summary:** 本文提出CoT Thought Leap Bridge Task，通过构建ScaleQM+数据集和CoT-Bridge模型，自动检测并填补链式推理中的思维跳跃，显著提升数学推理性能并展现跨领域泛化能力。 

> **Keywords:** LLM, Chain of Thought, Reasoning, Data Augmentation, Mathematical Reasoning

**Authors:** Haolei Xu, Yuchen Yan, Yongliang Shen, Wenqi Zhang, Guiyang Hou, Shengpei Jiang, Kaitao Song, Weiming Lu, Jun Xiao, Yueting Zhuang

**Institution(s):** Zhejiang University, The Chinese University of Hong Kong, Microsoft Research Asia


## Problem Background

大型语言模型（LLMs）在数学任务中通过链式推理（Chain of Thought, CoT）取得了显著进展，但现有CoT数据集常因专家省略中间步骤而出现‘思维跳跃’（Thought Leap），导致推理链条不完整，影响模型的学习和泛化能力。
论文通过初步实验证明，这种跳跃可使模型准确率下降高达27.83%，因此提出解决这一问题以提升推理完整性和模型性能。

## Method

* **任务定义与形式化**：首次系统化定义了‘思维跳跃’现象，提出CoT Thought Leap Bridge Task，将任务分为检测推理链中的跳跃位置和生成缺失中间步骤两个阶段，使用完整性函数判断相邻步骤间的逻辑连贯性。
* **数据集构建**：基于ScaleQuestMath数据集，构建专门的训练数据集ScaleQM+，通过策略性移除完整推理链中的中间步骤（考虑链长和位置，移除1-3步，保留20%完整链）模拟思维跳跃，并以原始完整链作为参考答案，确保训练数据的多样性和复杂性。
* **模型开发与训练**：开发CoT-Bridge模型，基于Qwen2.5-Math-7B进行微调，学习从不完整推理链预测跳跃位置并生成缺失步骤；同时设计基线模型CoT-Bridge-Random，仅在给定跳跃位置时生成步骤，用于对比分析跳跃检测的重要性。
* **数据增强应用**：将训练好的CoT-Bridge应用于现有数学推理数据集（如MetaMathQA和NuminaMath-CoT），生成增强版本（如MetaMath-Bridge和NuminaMath-Bridge），通过识别跳跃并插入生成步骤提升数据质量，支持下游模型训练。
* **核心创新**：方法不依赖模型架构改进，而是通过自动化填补推理空白提升数据结构完整性，是一种通用的数据增强策略。

## Experiment

* **实验设置全面**：在多个数学推理基准数据集（GSM8K, MATH500, GaoKao2023EN, MathOdyssey, OlympiadBenchEN, AMC23）上测试，使用Meta-Llama3.1-8B和Qwen2.5-Math-1.5B两种模型，涵盖基本和竞赛级任务，确保结果的广泛代表性。
* **性能提升显著**：CoT-Bridge在NuminaMath数据集上使Meta-Llama3.1-8B平均准确率提升5.87%（竞赛级任务如AMC23提升高达15.63%），在MetaMathQA上提升3.36%；与直接监督微调（Direct SFT）相比，性能一致性提高。
* **对比分析合理**：与随机插入步骤的CoT-Bridge-Random相比，准确检测跳跃位置对性能提升至关重要（随机插入可能导致性能下降）；与零样本桥接（Zero-shot Bridging）相比，CoT-Bridge表现出更稳定和显著的改进。
* **泛化能力验证**：在域外逻辑推理任务（如FOLIO, LogicQA）上，增强数据集提升平均准确率2.99%，表明推理完整性改进具有跨领域适用性。
* **即插即用效果**：作为增强模块，CoT-Bridge在知识蒸馏场景中提升3.02%，在强化学习中提升3.1%，实验覆盖多种训练范式，验证了方法的灵活性和实用性。

## Further Thoughts

论文强调推理链结构完整性对模型学习的影响可能大于内容准确性，这启发我们在数据质量评估中重新审视结构因素的重要性，尤其是在法律、科学问答等需要多步推理的领域；
自动化填补推理空白的策略展示了无需大规模人工标注即可提升数据质量的潜力，为处理多领域大数据集提供了新思路；
此外，分析不同位置（开始、中间、结束）桥接内容的影响提示，未来可针对推理阶段设计定制化增强策略，以进一步优化模型性能。