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
description: "本文提出难度感知提示（DAP）方法，通过动态调整推理轨迹长度构建简洁的 LiteCoT 数据集，显著提升推理模型性能和效率。"
---

> **Summary:** 本文提出难度感知提示（DAP）方法，通过动态调整推理轨迹长度构建简洁的 LiteCoT 数据集，显著提升推理模型性能和效率。 

> **Keywords:** LLM, Distillation, Reasoning, Prompt Engineering, Data Efficiency

**Authors:** Yifan Wu, Jingze Shi, Bingheng Wu, Jiayi Zhang, Xiaotian Lin, Nan Tang, Yuyu Luo

**Institution(s):** HKUST (Guangzhou), Independent Researcher, DeepWisdom


## Problem Background

大型语言模型（LLMs）通过链式思维（Chain-of-Thought, CoT）蒸馏方法可以将推理能力转移到较小模型，但现有方法生成的推理轨迹冗长（高达 32K token），导致训练和推理成本高昂，同时缺乏根据问题难度调整推理长度的适应性，使得学生模型无法学习灵活的推理策略。
论文旨在解决这些问题，构建一个简洁且难度适应的推理数据集，以提升训练效率和模型性能。

## Method

*   **核心思想:** 提出难度感知提示（Difficulty-Aware Prompting, DAP）方法，利用强大的教师模型（如 DeepSeek-R1）根据问题难度动态调整推理轨迹长度，生成简洁且完整的 CoT。
*   **具体步骤:**
    *   **长 CoT 生成:** 首先使用教师模型为每个问题生成详细的长 CoT，作为原始推理轨迹。
    *   **难度评估:** 设计针对不同难度（简单、中等、复杂）的提示模板，指导教师模型评估问题难度。
    *   **CoT 精炼:** 根据评估的难度，利用对应的提示模板重写长 CoT，生成简洁、难度适应的短 CoT。
    *   **数据集构建:** 通过 DAP 流程，构建 LiteCoT 数据集，包含 100K 个推理样本，平均每个样本仅 720 个 token，远低于传统 CoT 数据集的 5K-10K token。
    *   **模型训练:** 基于 LiteCoT 数据集，训练一组名为 Liter 的推理模型（参数规模为 1.5B、7B、32B），架构基于 Qwen2.5。
*   **创新点:** 通过提示工程而非额外训练辅助模型实现 CoT 压缩和难度适应，降低了计算成本，同时保持推理质量。

## Experiment

*   **有效性:** 实验表明，训练于 LiteCoT 的模型在 11 个基准数据集上的准确率普遍优于基于长 CoT 的模型，同时推理时间显著减少。例如，Liter-32B 在 AIME24 基准上达到 76.7% 的准确率，远超其他方法，且推理 token 仅约 5K。
*   **数据效率:** 使用仅 100K 个 LiteCoT 样本训练的模型，性能超越基于 800K 长 CoT 样本蒸馏的模型，展现了高质量数据的优势。
*   **对比分析:** 与其他 CoT 优化方法（如 Chain-of-Draft、LLMLingua-2）相比，DAP 方法在准确率和推理速度的平衡上表现更优。
*   **实验设置合理性:** 实验覆盖多种参数规模（1.5B 到 32B）和 11 个多样化基准（数学、通用问答、学术考试等），数据量对比（100K vs 800K）充分验证了方法的效率和泛化能力。

## Further Thoughts

论文通过提示工程实现难度感知的动态调整，启发我们可以在其他生成式任务中探索提示设计对输出控制的影响，例如根据用户需求调整文本详细程度；此外，难度感知的概念可扩展到情感分析或多模态任务中，优化资源分配；同时，少量高质量、结构化数据超越大量冗余数据的发现，提示我们在数据构建中应注重质量和适配性，而非单纯追求数量。