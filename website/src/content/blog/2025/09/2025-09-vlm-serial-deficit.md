---
title: "Visual serial processing deficits explain divergences in human and VLM reasoning"
pubDatetime: 2025-09-29T17:51:20+00:00
slug: "2025-09-vlm-serial-deficit"
type: "arxiv"
id: "2509.25142"
score: 0.7346548345328666
author: "grok-3-latest"
authors: ["Nicholas Budny", "Kia Ghods", "Declan Campbell", "Raja Marjieh", "Amogh Joshi", "Sreejan Kumar", "Jonathan D. Cohen", "Taylor W. Webb", "Thomas L. Griffiths"]
tags: ["VLM", "Serial Processing", "Visual Reasoning", "Human Comparison", "Task Complexity"]
institution: ["Princeton Neuroscience Institute", "Department of Psychology, Princeton University", "Department of Psychology, Université de Montréal", "Mila - Quebec AI Institute", "Department of Computer Science, Princeton University"]
description: "本文提出并验证了视觉语言模型（VLMs）在视觉推理任务中的序列处理缺陷假设，通过跨领域实验揭示其与人类性能差距的根本原因，并为模型改进提供了理论指导。"
---

> **Summary:** 本文提出并验证了视觉语言模型（VLMs）在视觉推理任务中的序列处理缺陷假设，通过跨领域实验揭示其与人类性能差距的根本原因，并为模型改进提供了理论指导。 

> **Keywords:** VLM, Serial Processing, Visual Reasoning, Human Comparison, Task Complexity

**Authors:** Nicholas Budny, Kia Ghods, Declan Campbell, Raja Marjieh, Amogh Joshi, Sreejan Kumar, Jonathan D. Cohen, Taylor W. Webb, Thomas L. Griffiths

**Institution(s):** Princeton Neuroscience Institute, Department of Psychology, Princeton University, Department of Psychology, Université de Montréal, Mila - Quebec AI Institute, Department of Computer Science, Princeton University


## Problem Background

视觉语言模型（VLMs）尽管在标准基准测试中表现出色，但在看似简单的视觉推理任务（如计数、视觉搜索）上经常无法达到人类水平。
作者提出，这一差异源于VLMs在视觉基础上的序列处理（serial processing）能力的缺陷，而序列处理是人类视觉推理的核心机制，能够通过逐步分析复杂场景来维持准确性。

## Method

*   **核心假设与设计思路:** 作者假设VLMs缺乏视觉基础的序列处理能力，导致其在需要逐步分析的视觉推理任务中表现不佳。为验证这一假设，他们设计了三个不同领域的任务，系统性地操控序列处理负荷，并将人类反应时间（reaction time, RT）作为序列处理负荷的代理指标，观察VLM准确率与RT的相关性。
*   **具体任务设计:**
    *   **几何推理任务:** 使用Geoclidean领域特定语言（DSL）生成几何刺激，通过最小描述长度（MDL）调整几何概念的复杂性（即所需几何原语的数量），以增加序列处理需求。任务为‘oddball’检测，要求从6个图像中识别不符合概念的图像。
    *   **视觉枚举任务:** 设计视觉计数任务，操控物体数量（1-8）、空间排列（重叠 vs. 非重叠）和颜色（统一 vs. 独特），以改变物体个体化（individuation）的序列处理负荷。任务要求报告场景中物体总数。
    *   **心理旋转任务:** 基于经典认知科学范式，呈现两幅图像（相同或镜像），通过调整旋转角度（0°-360°）增加序列处理需求。任务要求判断两图像是否相同。
*   **模型与人类对比:** 测试了多个VLM模型（如GPT-4o、Claude Sonnet 3.7等）在上述任务中的表现，并与人类参与者的准确率和反应时间进行对比。
*   **增强方法测试:** 进一步测试了使用链式思维（Chain-of-Thought, CoT）、推理训练和工具使用（如图像裁剪和旋转）的增强型VLM，评估这些方法是否能缓解序列处理缺陷。

## Experiment

*   **有效性与显著性:** 实验结果一致表明，VLM准确率与人类反应时间呈显著负相关（例如几何任务中 r = -0.73, p < 0.001；心理旋转任务中 r = -0.88, p < 0.001）。当任务需要更多序列处理时（即人类RT增加），VLM与人类的性能差距显著扩大，支持了序列处理缺陷假设。
*   **具体表现:** 在几何推理任务中，随着MDL增加，人类RT上升但准确率稳定，而VLM准确率下降；在视觉枚举任务中，VLM在重叠物体和高数量条件下表现较差，而人类通过增加RT维持准确率；在心理旋转任务中，VLM在较大旋转角度下准确率急剧下降，而人类仅略有下降。
*   **增强方法的局限性:** 使用CoT和推理训练的VLM在特定条件下（如视觉枚举中颜色独特的非重叠物体）表现有所提升，但对重叠条件或复杂几何任务帮助有限；工具使用在心理旋转任务中显著提高准确率（接近人类水平），但对重叠物体计数仍无明显改善。
*   **实验设置的合理性:** 实验覆盖了三种不同视觉推理领域，任务设计系统性地操控了序列处理负荷，测试了多种VLM模型，结果具有统计显著性，设置较为全面。但任务范围仍有限，未涵盖所有视觉推理场景（如视觉搜索），可能影响结论的普适性。

## Further Thoughts

论文提出的序列处理缺陷假设为理解VLM局限性提供了一个统一框架，启发我们思考如何通过模仿人类视觉注意机制（如眼动和注视循环）来改进模型。特别是通过视觉基础的强化学习（Visually Grounded Reinforcement Learning）生成与图像区域挂钩的推理轨迹，可能是一种有效方法。此外，是否可以通过设计新的多模态架构，将视觉序列处理与语言推理更紧密结合，以弥补当前VLM在非语言化视觉任务上的不足？这一方向值得进一步探索。