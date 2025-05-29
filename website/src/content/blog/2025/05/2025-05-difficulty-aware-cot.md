---
title: "Concise Reasoning, Big Gains: Pruning Long Reasoning Trace with Difficulty-Aware Prompting"
pubDatetime: 2025-05-26T09:04:44+00:00
slug: "2025-05-difficulty-aware-cot"
type: "arxiv"
id: "2505.19716"
score: 0.7113065705379048
author: "grok-3-latest"
authors: ["Yifan Wu", "Jingze Shi", "Bingheng Wu", "Jiayi Zhang", "Xiaotian Lin", "Nan Tang", "Yuyu Luo"]
tags: ["LLM", "Distillation", "Reasoning", "Prompt Engineering", "Data Efficiency"]
institution: ["HKUST (Guangzhou)", "Independent Researcher", "DeepWisdom"]
description: "本文提出难度感知提示（DAP）方法，通过动态调整推理轨迹长度构建简洁高效的 LiteCoT 数据集，显著降低训练和推理成本，同时提升学生模型在多样化基准上的性能。"
---

> **Summary:** 本文提出难度感知提示（DAP）方法，通过动态调整推理轨迹长度构建简洁高效的 LiteCoT 数据集，显著降低训练和推理成本，同时提升学生模型在多样化基准上的性能。 

> **Keywords:** LLM, Distillation, Reasoning, Prompt Engineering, Data Efficiency

**Authors:** Yifan Wu, Jingze Shi, Bingheng Wu, Jiayi Zhang, Xiaotian Lin, Nan Tang, Yuyu Luo

**Institution(s):** HKUST (Guangzhou), Independent Researcher, DeepWisdom


## Problem Background

大型语言模型（LLM）在推理任务中通过链式思维（Chain-of-Thought, CoT）蒸馏将强大推理能力转移到较小模型，但现有方法生成的推理轨迹往往过长（高达 32K token），导致训练和推理成本高昂；此外，统一长度的推理轨迹缺乏对问题难度的适应性，无法让学生模型学习根据任务复杂度调整推理策略。
论文旨在解决这两个关键问题，构建简洁且难度自适应的 CoT 数据集，以降低成本并提升效率。

## Method

*   **核心思想:** 提出难度感知提示（Difficulty-Aware Prompting, DAP）方法，利用强大的教师模型动态评估问题难度，并将冗长的推理轨迹重写为简洁、难度匹配的 CoT，减少训练和推理成本，同时保持性能。
*   **具体步骤:**
    *   **长 CoT 生成:** 使用教师模型（如 DeepSeek-R1）为每个问题生成详细的长 CoT 作为初始推理轨迹。
    *   **难度评估:** 设计包含简单、中等、复杂三个难度级别的提示模板，指导教师模型评估问题难度，分类为相应级别。
    *   **CoT 精炼:** 根据评估的难度级别，教师模型遵循对应提示模板，将长 CoT 重写为结构化和简洁的短 CoT，减少冗余内容，适应问题复杂度。
    *   **数据集构建:** 通过 DAP 流程，构建 LiteCoT 数据集，包含 100K 个难度自适应的推理样本，平均每个样本仅 720 个 token，远低于传统 CoT 数据集的 5K-10K token。
    *   **模型训练:** 使用 LiteCoT 数据集对基于 Qwen2.5 架构的学生模型（命名为 Liter，参数规模为 1.5B、7B、32B）进行微调。
*   **创新点:** 不依赖额外辅助模型训练，仅通过提示工程实现 CoT 压缩和难度适应，显著降低计算开销，同时保留必要推理深度。

## Experiment

*   **有效性验证:** 实验分为三部分，全面评估 DAP 方法和 LiteCoT 数据集的效果。
    *   **Exp-1（短 CoT vs 长 CoT）:** 在 11 个基准数据集上，训练于 LiteCoT 的模型性能显著优于长 CoT 模型，同时推理时间大幅减少。例如，Qwen2.5-7B-Instruct 在短 CoT 下的平均准确率为 57.3%，高于长 CoT 的 53.3%。
    *   **Exp-2（与主流模型对比）:** Liter 模型（训练于 100K LiteCoT 样本）在多个基准上超越训练于 800K 长 CoT 样本的主流模型。例如，Liter-32B 在 AIME24 上达到 76.7% 准确率，远超 DeepSeek-R1 蒸馏模型的 48.0%，且 token 使用量大幅减少。
    *   **Exp-3（与其他 CoT 优化方法对比）:** DAP 方法在性能和速度-准确率权衡上优于其他 CoT 压缩技术（如 Chain-of-Draft、LLMLingua-2），在大多数基准上取得最高准确率，同时保持较低推理时间。
*   **实验设置合理性:** 实验涵盖多种模型规模（1.5B 到 32B）和 11 个多样化基准（如 MATH500、AIME24），数据量对比（100K vs 800K）体现了方法的高效性，验证了 DAP 在不同任务和模型上的泛化能力。
*   **结论:** DAP 和 LiteCoT 显著提升了训练和推理效率，性能提升明显，尤其在资源受限场景下表现出色。

## Further Thoughts

难度感知提示（DAP）的核心 idea 令人启发，通过提示工程实现推理轨迹的动态调整，不依赖额外模型训练即可降低成本，这种方法是否可以扩展到其他领域（如多模态推理或代码生成）？此外，是否可以通过自适应提示设计，让模型在推理时动态分配资源（如 token 预算），实现更高效的实时推理？LiteCoT 数据集的成功也表明‘少而精’的数据可能比‘多而杂’的数据更有价值，未来是否可以通过类似策略优化其他任务的数据筛选和蒸馏过程？