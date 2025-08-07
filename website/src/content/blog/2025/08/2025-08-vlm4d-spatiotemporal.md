---
title: "VLM4D: Towards Spatiotemporal Awareness in Vision Language Models"
pubDatetime: 2025-08-04T06:06:06+00:00
slug: "2025-08-vlm4d-spatiotemporal"
type: "arxiv"
id: "2508.02095"
score: 0.43692044183516227
author: "grok-3-latest"
authors: ["Shijie Zhou", "Alexander Vilesov", "Xuehai He", "Ziyu Wan", "Shuwang Zhang", "Aditya Nagachandra", "Di Chang", "Dongdong Chen", "Xin Eric Wang", "Achuta Kadambi"]
tags: ["VLM", "Spatiotemporal Reasoning", "Benchmark", "Video Understanding", "Feature Reconstruction"]
institution: ["UCLA", "Microsoft", "UCSC", "USC"]
description: "VLM4D 提出首个评估视觉语言模型时空推理能力的基准，揭示现有模型不足，并通过监督微调和 4D 特征场重建探索改进方向，为动态环境下的视觉智能研究奠定基础。"
---

> **Summary:** VLM4D 提出首个评估视觉语言模型时空推理能力的基准，揭示现有模型不足，并通过监督微调和 4D 特征场重建探索改进方向，为动态环境下的视觉智能研究奠定基础。 

> **Keywords:** VLM, Spatiotemporal Reasoning, Benchmark, Video Understanding, Feature Reconstruction

**Authors:** Shijie Zhou, Alexander Vilesov, Xuehai He, Ziyu Wan, Shuwang Zhang, Aditya Nagachandra, Di Chang, Dongdong Chen, Xin Eric Wang, Achuta Kadambi

**Institution(s):** UCLA, Microsoft, UCSC, USC


## Problem Background

视觉语言模型（VLMs）在整合语言和视觉推理方面表现出色，但在理解动态时空交互方面存在根本性局限。
人类能够直观地追踪和推理物体运动、旋转和视角变化，这对动态现实世界的理解至关重要，而当前 VLMs 通常依赖于静态图像或简单的视频特征聚合，无法有效处理需要深层时空推理的任务。
论文旨在解决这一关键问题：如何评估和提升 VLMs 的时空推理能力，以接近人类水平，并在动态环境中实现更可靠的视觉智能。

## Method

*   **核心思想:** 提出 VLM4D 基准测试，首个专门设计用于评估 VLMs 时空推理能力的框架，通过构建多样化的视频数据集和问答对，系统性地测试模型在动态场景中的表现，并探索改进方法。
*   **数据集构建:** VLM4D 包含 1000 个视频（600 个真实视频，400 个合成视频）和超过 1800 个问答对，涵盖平移运动、旋转运动、视角感知和运动连续性等维度。
    *   真实视频来源于 Ego4D、DAVIS 和 YouTube-VOS 数据集，聚焦于动态动作片段，平均时长 3-8 秒。
    *   合成视频通过 Cosmos 模型生成，结合空间引导和人工验证确保质量，平均时长 5 秒。
    *   问答对主要由人工标注，辅以 LLM（如 GPT-4o）生成多选答案，并通过三轮交叉验证确保准确性和时空对齐。
*   **评估框架:** 对 23 个最先进的开源和闭源 VLMs 进行零样本评估，采用多选题（MCQ）形式，使用准确率作为主要指标。
    *   评估设置包括直接输出（Direct Output, DO）和思维链（Chain-of-Thought, CoT）两种推理模式。
    *   使用 LLM-as-Judge 方法（结合 GPT-o3 和 o4-mini）评估输出，确保对推理过程的全面判断。
*   **改进探索:** 提出两种潜在解决方案以增强时空理解能力。
    *   **时空监督微调（Spatial-Temporal SFT）:** 在真实和合成数据上对模型（如 Qwen 2VL）进行微调，聚焦时空丰富的动作和交互。
    *   **4D 特征场重建:** 基于 Feature4X 框架，将 2D 特征提升到 4D 特征场，提供结构化的时空场景表示，增强推理阶段的运动和空间理解。

## Experiment

*   **有效性:** 实验结果显示，当前 VLMs 在时空推理能力上与人类基准（98.8% 准确率）存在显著差距，最好的模型（Gemini-2.5-Pro）整体准确率为 62.0%，其次是 GPT-4o（57.5%），开源模型如 Qwen2.5-VL-72B 达到 53.0%，表明现有模型在 4D 理解上的不足。
*   **性能差异:** 模型在不同类别（如真实 vs 合成数据、自我中心 vs 外部中心视角）上的表现不一致，显示缺乏泛化的时空理解能力。
*   **改进效果:** 
    *   监督微调（SFT）在 Qwen 2VL 和 Qwen 2.5VL 模型上显著提升准确率（例如 Qwen 2.5VL-7B 从 43.4% 提升到 56.3%），表明针对性训练有效，但合成数据未显著优于真实数据，提示数据质量的重要性。
    *   4D 特征场重建在 InternVideo2-8B 模型上带来小幅提升（CoT 模式下从 36.0% 提升到 37.4%），显示 4D 表示的潜力，但方法需逐场景优化，计算成本高，泛化性受限。
*   **实验设置合理性:** 实验覆盖了多种模型（23 个）、数据类型（真实和合成）和推理模式（DO 和 CoT），评估框架全面，但改进方法的提升幅度有限，反映问题复杂性，需进一步研究。

## Further Thoughts

VLM4D 基准揭示了 VLMs 在时空推理上的核心缺陷，启发未来研究可从认知科学中借鉴人类时空认知机制，设计更接近人类推理的模型架构；4D 特征场重建的思路提示结合结构化场景表示可能是突破方向，尽管当前实现有局限，但其潜力值得探索；此外，现有视频指令微调数据集的时空标签不足问题，启发可以通过构建更精细的标注数据或利用生成模型提升合成数据质量，以弥补真实数据的局限性。