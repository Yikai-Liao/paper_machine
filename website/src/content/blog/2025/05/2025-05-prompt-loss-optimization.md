---
title: "PMPO: Probabilistic Metric Prompt Optimization for Small and Large Language Models"
pubDatetime: 2025-05-22T06:59:10+00:00
slug: "2025-05-prompt-loss-optimization"
type: "arxiv"
id: "2505.16307"
score: 0.7053155712926437
author: "grok-3-latest"
authors: ["Chenzhuo Zhao", "Ziqian Liu", "Xingda Wang", "Junting Lu", "Chaoyi Ruan"]
tags: ["LLM", "Prompt Optimization", "Cross-Entropy Loss", "Iterative Refinement", "Model Scalability"]
institution: ["Peking University", "Unaffiliated", "National University of Singapore"]
description: "PMPO 提出了一种基于交叉熵损失的提示优化框架，通过迭代精炼提升语言模型性能，兼顾效率和跨模型适用性。"
---

> **Summary:** PMPO 提出了一种基于交叉熵损失的提示优化框架，通过迭代精炼提升语言模型性能，兼顾效率和跨模型适用性。 

> **Keywords:** LLM, Prompt Optimization, Cross-Entropy Loss, Iterative Refinement, Model Scalability

**Authors:** Chenzhuo Zhao, Ziqian Liu, Xingda Wang, Junting Lu, Chaoyi Ruan

**Institution(s):** Peking University, Unaffiliated, National University of Singapore


## Problem Background

提示优化作为一种替代微调的方法，用于提升大型语言模型（LLM）的性能，但现有方法面临高计算成本、依赖大型模型自省能力以及缺乏任务和模型规模通用性的问题。
PMPO 旨在通过一种轻量级、统一的框架解决这些问题，利用模型内部的交叉熵损失作为直接评估信号，避免输出生成和人工标注的依赖。

## Method

*   **核心思想:** 将提示优化转化为损失最小化问题，直接利用模型的交叉熵损失作为评估信号，无需输出生成或外部评分，适用于大小模型和多样任务。
*   **具体步骤:**
    *   **掩码引导的重要性评估（Mask-Guided Importance Evaluation）:** 将提示分解为语义单元，逐一掩码并计算批量交叉熵损失变化（∆L），识别对性能贡献正向（∆L > 0）、负向（∆L < 0）或中性的片段，为后续优化提供目标。
    *   **基于损失的提示评估与变体生成（Prompt Evaluation and Variant Generation）:** 使用交叉熵损失（针对监督任务）或偏好损失（针对偏好任务）评估提示质量，针对高损失样本（即模型表现较差的输入-输出对）构造重写提示，生成多个变体，并通过模型自身生成改进版本。
    *   **迭代优化（Iterative Refinement）:** 在多轮迭代中，持续评估所有提示变体，选择损失最低的提示作为下一轮起点，直至达到最大迭代次数（实验中为 20 轮）。
*   **关键优势:** 仅依赖前向传播和似然计算，避免生成循环的高成本；支持批量评估，提升样本效率；通过统一损失框架兼容监督和偏好任务，且对小型模型友好，无需复杂推理能力。

## Experiment

*   **有效性:** PMPO 在多个基准数据集上表现出色，例如在 BBH 上平均准确率达 80.6%，超越 EvoPrompt（78.0%）和 OPRO（77.0%）；在 GSM8K 和 AQUA-RAT 上分别达到 94.0% 和 84.6% 准确率；在 AlpacaEval 2.0 上胜率提升超 19 个百分点（从 31.81% 到 51.52%）。
*   **提升显著性:** 相比依赖输出生成和自省的基线方法，PMPO 的损失评估机制显著提高效率，尤其在多步推理和指令对齐任务中效果突出，生成的提示更轻量且结构适应性强。
*   **实验设置合理性:** 实验覆盖多种模型规模（Qwen2.5-0.5B 到 32B）和任务类型（推理、数学、指令跟随），数据集划分合理（训练集上限 50 样本，测试集完整），对比基线全面（包括 CoT、OPRO 等手动和自动化方法）；跨模型泛化实验验证了提示迁移性，尽管小型模型上效果略降。
*   **局限性:** 闭源模型因无法访问完整损失信号而应用受限，极低资源场景下可能过拟合，实验未完全解决但通过消融研究和跨模型测试部分揭示影响。

## Further Thoughts

PMPO 的损失评估机制启发了对模型内部信号的更广泛利用，例如是否可以结合注意力权重或中间层表示进一步精细化提示片段评估？此外，PMPO 静态优化的框架是否可扩展为动态提示生成，即在推理时根据输入实时调整提示？跨模型泛化结果还提示了一种可能性：设计‘模型自适应提示优化’框架，根据目标模型规模和架构自动调整优化策略，以提升迁移效果。