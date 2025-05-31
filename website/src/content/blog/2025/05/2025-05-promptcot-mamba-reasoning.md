---
title: "Scaling Reasoning without Attention"
pubDatetime: 2025-05-28T14:52:15+00:00
slug: "2025-05-promptcot-mamba-reasoning"
type: "arxiv"
id: "2505.22425"
score: 0.8974387814203968
author: "grok-3-latest"
authors: ["Xueliang Zhao", "Wei Wu", "Lingpeng Kong"]
tags: ["LLM", "State Space Model", "Reasoning", "Curriculum Learning", "Efficiency"]
institution: ["The University of Hong Kong", "Ant Group"]
description: "本文提出 PromptCoT-Mamba，一种基于 Mamba-2 架构的无注意力语言模型，通过课程式微调和 PromptCoT 数据合成，在复杂推理任务中显著提升性能和效率，超越同规模 Transformer 模型。"
---

> **Summary:** 本文提出 PromptCoT-Mamba，一种基于 Mamba-2 架构的无注意力语言模型，通过课程式微调和 PromptCoT 数据合成，在复杂推理任务中显著提升性能和效率，超越同规模 Transformer 模型。 

> **Keywords:** LLM, State Space Model, Reasoning, Curriculum Learning, Efficiency

**Authors:** Xueliang Zhao, Wei Wu, Lingpeng Kong

**Institution(s):** The University of Hong Kong, Ant Group


## Problem Background

大型语言模型（LLMs）在复杂推理任务中取得了显著进展，但仍面临两大瓶颈：
1. **架构效率问题**：基于Transformer的模型依赖自注意力机制，导致在长上下文任务中内存和计算成本随序列长度线性增长，效率低下。
2. **结构化训练不足**：在高难度领域（如数学和代码生成）缺乏系统化的微调策略，限制了模型的推理能力。
本文旨在通过无注意力架构和创新训练方法，解决效率和推理能力提升的双重挑战。

## Method

* **核心思想**：提出 PromptCoT-Mamba，一个完全无注意力机制的语言模型，通过高效的状态空间架构和结构化训练策略，实现复杂推理任务的高性能和高效推理。
* **架构设计**：
  - 基于 Mamba-2 的状态空间双层（State Space Dual, SSD）架构，替代传统自注意力机制。
  - SSD 层通过递归更新隐藏状态，在自回归推理中实现固定内存和常数时间复杂度（每步计算复杂度为 O(NP)），避免了 Transformer 中键值缓存（KV Cache）随序列长度增长的问题。
  - 训练时，SSD 层通过并行化的结构化矩阵运算实现高效计算，总复杂度为 O(TNP)，远低于 Transformer 的 O(T²N)。
* **训练策略**：
  - 采用两阶段课程式微调（Curriculum Fine-Tuning）框架：
    1. **初始化阶段**：基于开源数据集（如 OpenCodeReasoning 和 OpenThoughts2）进行基础推理能力训练，包含大量自动合成的示例和少量专家编写的竞赛问题。
    2. **高级阶段**：基于 PromptCoT 合成范式生成高复杂度的教学式问题，通过抽象概念选择和理据引导生成，培养专家级推理能力。
  - PromptCoT 方法通过联合优化理据生成和问题生成的似然性，确保生成的问题具有抽象性和推理深度。
* **关键创新**：结合架构效率（无注意力机制）和数据驱动监督（课程式训练），使模型在复杂任务中保持高性能，同时显著降低推理成本。

## Experiment

* **性能表现**：PromptCoT-Mamba-7B 在多个基准测试（如 AIME 24、AIME 25、Livecodebench）上显著优于同等规模的 Transformer 和混合 Mamba-Transformer 模型，甚至在部分任务上超越更大的 Gemma3-27B（如 AIME 24 上高出 2.6%，Livecodebench 上高出 3.0%）。
* **效率提升**：在推理效率上，相比 Transformer 基线 s1.1-7B，PromptCoT-Mamba-7B 在 24GB GPU 内存下实现了 3.66 倍吞吐量提升，在 72GB 内存下提升 1.69 倍，展现了在资源受限和长上下文场景下的优势。
* **实验设置合理性**：实验覆盖数学和代码生成两大领域，包含七个基准数据集，评估指标（如 pass@1 和 avg@k）与现有研究一致，确保了结果的可比性；消融研究验证了课程式训练和 PromptCoT 合成阶段的重要性，缺少该阶段会导致性能显著下降（如 AIME 24 从 35.2% 降至 11.7%）。
* **领域特化效果**：数学特化版本 PromptCoT-Mamba-Math-7B 在数学任务上进一步提升（如 AIME 24 提升至 42.9%），但代码任务性能下降，体现了领域特化的权衡。

## Further Thoughts

论文中的无注意力架构（Mamba-2）通过状态空间模型实现了高效推理，这启发我们思考是否可以将类似架构与其他高效机制（如线性注意力）结合，进一步优化长上下文任务的表现；此外，PromptCoT 范式通过教学式数据合成提升推理能力，这种方法可以扩展到其他高难度领域（如科学推理或法律分析），甚至可以探索多模态推理数据的合成；最后，领域特化与通用能力的权衡让我联想到，是否可以通过动态调整训练数据分布或模块化架构设计，在不同任务间实现更灵活的性能优化。