---
title: "Scaling Reasoning without Attention"
pubDatetime: 2025-05-28T14:52:15+00:00
slug: "2025-05-attention-free-reasoning"
type: "arxiv"
id: "2505.22425"
score: 0.8974387814203968
author: "grok-3-latest"
authors: ["Xueliang Zhao", "Wei Wu", "Lingpeng Kong"]
tags: ["LLM", "State Space Model", "Reasoning", "Curriculum Learning", "Efficiency"]
institution: ["The University of Hong Kong", "Ant Group"]
description: "本文提出 PromptCoT-Mamba，一种基于 Mamba-2 的无注意力语言模型，通过两阶段课程微调和 PromptCoT 数据合成，在复杂推理任务上超越同规模 Transformer 模型，并显著提升推理效率。"
---

> **Summary:** 本文提出 PromptCoT-Mamba，一种基于 Mamba-2 的无注意力语言模型，通过两阶段课程微调和 PromptCoT 数据合成，在复杂推理任务上超越同规模 Transformer 模型，并显著提升推理效率。 

> **Keywords:** LLM, State Space Model, Reasoning, Curriculum Learning, Efficiency

**Authors:** Xueliang Zhao, Wei Wu, Lingpeng Kong

**Institution(s):** The University of Hong Kong, Ant Group


## Problem Background

大型语言模型（LLMs）在复杂推理任务中面临两大挑战：基于 Transformer 架构的注意力机制导致内存和计算成本随上下文长度线性增长，限制了长上下文推理的效率；同时，缺乏针对高难度领域（如数学和代码推理）的结构化微调方法，影响模型性能。
论文旨在解决如何在不依赖注意力机制的情况下，构建高效且推理能力强大的模型，并通过结构化训练提升其在复杂任务上的表现。

## Method

*   **架构创新：基于 Mamba-2 的状态空间双层（SSD Layers）**：
    *   完全替代自注意力机制，通过递归状态更新和线性读出计算 token 分布，避免 Transformer 的键值缓存（KV Cache）问题，实现固定内存和常数时间推理。
    *   推理时，SSD 层根据当前 token 嵌入更新隐藏状态，通过低秩状态更新（复杂度 O(NP)）计算下一个 token 分布；训练时，采用并行结构化收缩操作（复杂度 O(TNP)），比 Transformer 的 O(T²N) 更高效。
*   **训练策略：两阶段课程微调框架**：
    *   **初始化阶段**：基于开源数据集（如 OpenCodeReasoning 和 OpenThoughts2），训练基础推理能力，使用大量自动合成和少量专家编写的示例，确保模型掌握基本推理技能。
    *   **高级阶段**：引入 PromptCoT 合成范式，通过抽象概念选择和推理引导生成高复杂度的教学式问题，优化模型在数学和代码推理等高难度任务上的表现，强调专家级抽象和演绎深度。
*   **核心思想**：结合 Mamba-2 的架构效率和 PromptCoT 的结构化数据监督，解决 Transformer 效率瓶颈，同时提升复杂推理能力。

## Experiment

*   **性能表现**：PromptCoT-Mamba-7B 在多个高难度基准测试中显著优于同规模 Transformer 和混合 Mamba-Transformer 模型，在 AIME 24（35.2%）、AIME 25（24.6%）和 Livecodebench（29.9%）上甚至超越更大模型 Gemma3-27B（分别高 2.6%、0.6%、3.0%）。
*   **效率提升**：在 24GB GPU 内存下，吞吐量比 Transformer 基线（s1.1-7B）高 3.66 倍；在 72GB 内存下高 1.69 倍，展现了无注意力架构在资源受限和长上下文场景下的优势。
*   **消融分析**：PromptCoT 合成阶段对性能至关重要，去除后 AIME 24 得分从 35.2% 降至 11.7%；数学特化版本（PromptCoT-Mamba-Math-7B）在数学任务上进一步提升（如 AIME 24 达 42.9%），但代码任务性能下降，反映领域特化与泛化的权衡。
*   **实验设置评价**：实验覆盖数学和代码领域的高难度基准（如 MATH-500、HumanEval+），评价指标（pass@1、avg@k）合理，训练和测试细节透明；但未深入探讨超长上下文（>65,536 token）表现，可能限制对极限推理能力的评估。

## Further Thoughts

论文揭示了无注意力架构（如 Mamba-2）在复杂推理任务中的潜力，启发我们思考是否可以通过优化状态空间模型的参数或层级结构，扩展其在多模态推理等更广泛任务中的应用；此外，PromptCoT 范式通过教学式数据合成提升推理能力，提示我们是否可以将类似方法应用于法律或科学推理，通过领域专家引导的数据设计进一步提升模型性能；最后，领域特化与泛化权衡的研究让我考虑是否可以通过动态数据分布调整或多任务学习，在保持特化优势的同时增强模型的泛化能力。