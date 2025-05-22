---
title: "Mind the Gap: Bridging Thought Leap for Improved Chain-of-Thought Tuning"
pubDatetime: 2025-05-20T17:59:31+00:00
slug: "2025-05-thought-leap-bridge"
type: "arxiv"
id: "2505.14684"
score: 0.7853950014592904
author: "grok-3-latest"
authors: ["Haolei Xu", "Yuchen Yan", "Yongliang Shen", "Wenqi Zhang", "Guiyang Hou", "Shengpei Jiang", "Kaitao Song", "Weiming Lu", "Jun Xiao", "Yueting Zhuang"]
tags: ["LLM", "Chain of Thought", "Reasoning", "Data Augmentation", "Fine-Tuning"]
institution: ["Zhejiang University", "The Chinese University of Hong Kong", "Microsoft Research Asia"]
description: "本文提出CoT Thought Leap Bridge任务，通过自动检测并填补推理链中的思维跳跃，显著提升了大型语言模型在数学推理任务中的性能和泛化能力。"
---

> **Summary:** 本文提出CoT Thought Leap Bridge任务，通过自动检测并填补推理链中的思维跳跃，显著提升了大型语言模型在数学推理任务中的性能和泛化能力。 

> **Keywords:** LLM, Chain of Thought, Reasoning, Data Augmentation, Fine-Tuning

**Authors:** Haolei Xu, Yuchen Yan, Yongliang Shen, Wenqi Zhang, Guiyang Hou, Shengpei Jiang, Kaitao Song, Weiming Lu, Jun Xiao, Yueting Zhuang

**Institution(s):** Zhejiang University, The Chinese University of Hong Kong, Microsoft Research Asia


## Problem Background

大型语言模型（LLMs）在数学任务中通过链式推理（Chain of Thought, CoT）取得了显著进展，但现有CoT数据集中普遍存在‘思维跳跃’（Thought Leap）现象，即专家省略了中间推理步骤，导致推理链条不完整，阻碍了模型的学习和泛化能力。
论文通过初步实验证明，这种跳跃可导致性能下降高达27.83%，因此亟需解决推理结构的完整性问题。

## Method

* **任务定义与形式化**：提出‘CoT Thought Leap Bridge Task’，旨在自动检测推理链中的思维跳跃（即相邻步骤间完整性校验失败），并生成缺失的中间步骤以恢复逻辑连贯性。
* **数据集构建**：基于ScaleQuestMath数据集，构造专门的训练数据集ScaleQM+，通过从完整推理链中移除1-3个中间步骤（根据链条长度调整），模拟思维跳跃，并保留完整步骤作为参考答案，同时以0.2概率保留完整链以训练模型识别无需桥接的情况。
* **模型开发**：训练CoT-Bridge模型（基于Qwen2.5-Math-7B微调），学习从不完整推理链预测跳跃位置并生成缺失步骤；同时设计基线模型CoT-Bridge-Random，仅在给定跳跃位置时生成步骤，用于对比实验。
* **数据增强应用**：将CoT-Bridge应用于现有数学推理数据集（如MetaMathQA和NuminaMath-CoT），生成增强版本（如MetaMath-Bridge和NuminaMath-Bridge），通过识别跳跃并插入生成步骤，提升训练数据质量。
* **技术细节**：方法注重推理结构的完整性而非仅事实准确性，适配不同数据集的步骤分隔符（如‘
’或‘

’），并作为即插即用模块与其他优化技术（如知识蒸馏、强化学习）结合。

## Experiment

* **实验设置**：在多个数学推理基准（GSM8K, MATH500, GaoKao2023EN, MathOdyssey, OlympiadBenchEN, AMC23）和逻辑推理基准（FOLIO, LogicQA等）上，使用Meta-Llama3.1-8B和Qwen2.5-Math-1.5B进行监督微调（SFT），评估CoT-Bridge的效果。
* **性能提升**：CoT-Bridge显著提升模型性能，例如在Meta-Llama3.1-8B上，NuminaMath数据集平均准确率提升+5.87%（竞赛级AMC23提升高达+15.63%）；在Qwen2.5-Math-1.5B上，MetaMathQA平均准确率提升+3.36%（MATH500提升+7.00%）。
* **对比分析**：相较于直接SFT、零样本桥接和随机桥接（CoT-Bridge-Random），CoT-Bridge在精准定位跳跃和生成高质量步骤方面表现更优，证明了准确检测跳跃位置的重要性。
* **泛化性**：在域外逻辑推理任务上，CoT-Bridge提升了模型性能（Meta-Llama3.1-8B平均+2.99%），表明推理完整性改进具有广泛适用性。
* **即插即用性**：作为增强模块，CoT-Bridge在知识蒸馏和拒绝采样中提升数据质量（平均准确率分别+3.02%和+1.37%），并为强化学习提供更好起点（RL准确率从60.88%提升至63.98%）。
* **合理性与局限**：实验设计全面，覆盖多种模型、数据集和任务类型；噪声分析表明低质量桥接步骤影响有限，但方法未在大规模模型（如32B/72B）上验证，且训练数据局限于数学领域。

## Further Thoughts

论文强调推理结构完整性对模型学习的重要性，启发我们思考是否可以在其他多步骤推理领域（如法律或科学问答）中应用类似方法检测并修复结构缺陷；此外，自动化数据增强的思路是否能扩展到修复其他数据问题（如逻辑矛盾或冗余步骤）；最后，桥接内容在推理链不同位置（开始、中间、结束）的不同作用提示我们，或许可以通过针对性增强特定位置的推理步骤，进一步优化训练效果。