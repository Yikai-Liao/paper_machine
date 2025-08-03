---
title: "Self-Foveate: Enhancing Diversity and Difficulty of Synthesized Instructions from Unsupervised Text via Multi-Level Foveation"
pubDatetime: 2025-07-31T11:18:42+00:00
slug: "2025-07-self-foveate-instruction"
type: "arxiv"
id: "2507.23440"
score: 0.6395123446155101
author: "grok-3-latest"
authors: ["Mingzhe Li", "Xin Lu", "Yanyan Zhao"]
tags: ["LLM", "Instruction Tuning", "Synthetic Data", "Diversity", "Difficulty"]
institution: ["Harbin Institute of Technology, China"]
description: "本文提出 SELF-FOVEATE 方法，通过多层次注视机制从无监督文本中合成多样性和难度更高的指令，显著提升大型语言模型在下游任务中的性能。"
---

> **Summary:** 本文提出 SELF-FOVEATE 方法，通过多层次注视机制从无监督文本中合成多样性和难度更高的指令，显著提升大型语言模型在下游任务中的性能。 

> **Keywords:** LLM, Instruction Tuning, Synthetic Data, Diversity, Difficulty

**Authors:** Mingzhe Li, Xin Lu, Yanyan Zhao

**Institution(s):** Harbin Institute of Technology, China


## Problem Background

大型语言模型（LLMs）在指令调优（Instruction Tuning）中依赖高质量的指令数据，但当前从无监督文本中合成指令的方法存在多样性（Diversity）和难度（Difficulty）不足的问题，导致模型泛化能力和复杂任务表现受限。
作者旨在通过自动化方法挖掘无监督文本中的丰富信息，生成更具多样性和难度的指令数据，以提升模型在下游任务中的问题解决能力，同时减少对人工标注的依赖。

## Method

*   **核心思想:** 提出 SELF-FOVEATE 方法，通过 'Micro-Scatter-Macro' 多层次注视机制，从无监督文本中挖掘细粒度到全局的信息，结合多种合成范式生成多样性和难度更高的指令。
*   **具体实现:**
    *   **Micro-Foveate Level（微观注视层）:** 关注文本中的细粒度信息（如实体和属性，称为 '注视元素'），通过 '逆向合成'（Reverse Synthesis）生成指令，即先提取潜在答案，再反向生成问题，确保指令覆盖关键细节，提升多样性。
    *   **Scatter-Foveate Level（分散注视层）:** 提取文本中分散的关键信息，随机组合成 '注视组'（Foveate Groups），通过 '直接合成'（Direct Synthesis）生成指令，强调实体间深层关系，提升指令难度。
    *   **Macro-Foveate Level（宏观注视层）:** 从全局视角提取文本中的修辞手法或写作技巧（如隐喻、夸张，称为 '注视片段'），通过 '转录合成'（Transcription Synthesis）将陈述性内容转化为问答形式，增强指令对整体信息的理解深度。
    *   **Re-synthesis Module（再合成模块）:** 对生成的指令进行后处理，过滤异常指令，通过参考成功样例和调整超参数（如温度）进行多次迭代再合成，确保指令与源文本的一致性和质量。
*   **关键点:** 该方法无需人工标注，完全依赖 LLMs 的能力，通过多层次信息提取模拟人类阅读理解过程，兼顾指令的多样性和难度。

## Experiment

*   **有效性:** SELF-FOVEATE 在多样性（Diversity）和难度（Difficulty）上显著优于基线方法（如 Self-QA, Bonito），在 SQuAD, HotpotQA, FilmWiki 数据集上的 SelfBLEU 和 Embedding Diversity 指标均有提升，难度胜率高达 70%-100%。
*   **下游任务表现:** 在指令调优后，SELF-FOVEATE 显著提升了模型（如 Llama-3.1-8B, Qwen2.5-7B）在下游任务上的性能，Recall 和 LLM Accuracy 指标在所有数据集上均优于基线，例如在 SQuAD 上 Llama-3.1-8B 的 Recall 从 0.386 提升至 0.484，Accuracy 从 0.405 提升至 0.490。
*   **实验设置合理性:** 实验覆盖多个数据集和模型，确保结果泛化性；消融研究验证了各组件（Micro, Scatter, Macro）的必要性；指令数量对性能的影响分析显示 SELF-FOVEATE 随着数据规模增加性能提升更显著。
*   **不足与成本:** 尽管效果显著，但计算成本较高（约 110 GPU 小时），可能限制大规模应用。

## Further Thoughts

SELF-FOVEATE 的多层次注视机制（Micro-Scatter-Macro）启发了我思考如何将类似的分层信息提取方法应用于其他 NLP 任务，如文本摘要或对话生成，通过结合细粒度、分散关系和全局视角提升生成质量；此外，是否可以通过自适应调整注视层次权重（例如根据文本类型动态调整 Micro 和 Macro 比例），进一步优化指令合成效果？另一个方向是设计更具挑战性的合成数据，针对性提升模型在数学推理或多步问答等领域的表现。