---
title: "Neural Incompatibility: The Unbridgeable Gap of Cross-Scale Parametric Knowledge Transfer in Large Language Models"
pubDatetime: 2025-05-20T14:42:03+00:00
slug: "2025-05-neural-incompatibility-pkt"
type: "arxiv"
id: "2505.14436"
score: 0.8008444799563887
author: "grok-3-latest"
authors: ["Yuqiao Tan", "Shizhu He", "Kang Liu", "Jun Zhao"]
tags: ["LLM", "Knowledge Transfer", "Parameter Alignment", "Neural Incompatibility"]
institution: ["The Key Laboratory of Cognition and Decision Intelligence for Complex Systems, Institute of Automation, Chinese Academy of Sciences", "School of Artificial Intelligence, University of Chinese Academy of Sciences"]
description: "本文首次系统定义并探索了跨规模大型语言模型间的参数知识转移（PKT），提出Pre-Align范式和LaTen方法，通过神经元定位与参数对齐减少对齐成本，同时揭示了神经不兼容性作为根本挑战。"
---

> **Summary:** 本文首次系统定义并探索了跨规模大型语言模型间的参数知识转移（PKT），提出Pre-Align范式和LaTen方法，通过神经元定位与参数对齐减少对齐成本，同时揭示了神经不兼容性作为根本挑战。 

> **Keywords:** LLM, Knowledge Transfer, Parameter Alignment, Neural Incompatibility

**Authors:** Yuqiao Tan, Shizhu He, Kang Liu, Jun Zhao

**Institution(s):** The Key Laboratory of Cognition and Decision Intelligence for Complex Systems, Institute of Automation, Chinese Academy of Sciences, School of Artificial Intelligence, University of Chinese Academy of Sciences


## Problem Background

大型语言模型（LLMs）通过参数编码了大量知识，参数透明且可操作，为直接通过参数进行知识转移（Parametric Knowledge Transfer, PKT）提供了可能性，尤其是在跨规模模型（如从大模型到小模型）之间。然而，不同规模模型的参数空间不匹配（如层数、维度差异）以及潜在的‘神经不兼容性’（Neural Incompatibility，即行为和结构上的根本差异），导致跨规模参数知识转移效果不佳，成为亟待解决的关键问题。

## Method

* **核心范式**：论文提出了两种参数知识转移（PKT）范式：Post-Align PKT（PostPKT）和Pre-Align PKT（PrePKT），以解决跨规模模型参数转移中的对齐问题。
* **PostPKT**：现有方法（如SEEKING）的范式，先从大模型提取任务相关参数（如通过LoRA初始化），注入小模型后，通过大规模微调进行参数对齐，对齐成本高。
* **PrePKT**：论文提出的新范式，旨在先对齐参数空间，再进行知识注入，以减少后续微调成本，直接提升小模型性能。
* **LaTen方法（Locate-Then-Align）**：为实现PrePKT，提出了一种具体解决方案：
  1. **Locate（定位）**：采用神经元级归因方法（Neuron-level Attribution），针对前馈网络（FFN）和多头自注意力（MHSA）模块，计算每个神经元的重要性分数，定位大模型中与任务相关的最重要神经元和层，提取相关参数。
  2. **Align（对齐）**：通过一个轻量级超网络（Hypernetwork，两层MLP），将提取的参数从大模型参数空间映射到小模型参数空间，仅用少量数据（<100个样本）训练对齐过程，减少计算成本。
* **关键创新**：LaTen避免了大规模微调，通过预对齐减少注入后对模型性能的破坏，同时保持任务相关知识的有效性。

## Experiment

* **有效性**：在Llama-2模型（7B和13B）上，基于MMLU、GSM8K、HumanEval和MBPP四个基准数据集的实验显示，PostPKT（如SEEKING）在大多数任务上优于随机初始化，但不如自模型参数初始化的PiSSA方法，性能不稳定；PrePKT的LaTen方法在所有基准上均有提升，例如在GSM8K上从16.07提升至20.47，接近大模型原始性能（20.55），平均提升1.86个百分点。
* **对比分析**：在受限数据条件下，LaTen相比PostPKT的SEEKING方法表现更优，尤其在GSM8K上提升5.69个百分点；在低资源场景下，LaTen优于语言蒸馏方法（如SeqKD、Supervised KD）。
* **实验设置合理性**：实验任务覆盖世界知识、数学推理和代码生成，领域多样；数据分离（提取、对齐、训练集）避免泄露；但模型规模限于7B和13B，未涉及更大模型，结论普适性有待验证。
* **不足**：LaTen对齐过程不稳定，难以找到最佳检查点；对更强的大模型（如WizardCoder）参数转移效果未见显著提升，验证了‘神经不兼容性’问题。

## Further Thoughts

论文提出的‘神经不兼容性’（Neural Incompatibility）概念揭示了跨规模LLM在行为和参数结构上的低相似性，这种差异可能源于预训练数据分布、架构设计或优化动态。未来可探索设计‘中间桥梁模型’或‘参数转换器’来缓解不兼容性；研究知识表示粒度的差异（大模型抽象表示 vs 小模型具体表示）对转移的影响；或在特定训练阶段（如预训练）进行参数空间规范化约束，以提高PKT效果。