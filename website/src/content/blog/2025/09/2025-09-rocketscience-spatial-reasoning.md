---
title: "Understanding Space Is Rocket Science -- Only Top Reasoning Models Can Solve Spatial Understanding Tasks"
pubDatetime: 2025-09-02T10:32:58+00:00
slug: "2025-09-rocketscience-spatial-reasoning"
type: "arxiv"
id: "2509.02175"
score: 0.4891725862814311
author: "grok-3-latest"
authors: ["Nils Hoehing", "Ellen Rushe", "Mayug Maniparambil", "Noel E. O’Connor", "Anthony Ventresque"]
tags: ["VLM", "Spatial Reasoning", "Contrastive Benchmark", "Chain of Thought", "Object Localization"]
institution: ["University College Dublin", "Dublin City University", "Trinity College Dublin"]
description: "本文提出 RocketScience 基准测试，通过全新的现实世界对比性数据揭示视觉语言模型在空间理解上的不足，并证明空间推理是主要瓶颈，为模型改进提供明确方向。"
---

> **Summary:** 本文提出 RocketScience 基准测试，通过全新的现实世界对比性数据揭示视觉语言模型在空间理解上的不足，并证明空间推理是主要瓶颈，为模型改进提供明确方向。 

> **Keywords:** VLM, Spatial Reasoning, Contrastive Benchmark, Chain of Thought, Object Localization

**Authors:** Nils Hoehing, Ellen Rushe, Mayug Maniparambil, Noel E. O’Connor, Anthony Ventresque

**Institution(s):** University College Dublin, Dublin City University, Trinity College Dublin


## Problem Background

视觉语言模型（VLMs）在理解图像中物体间空间关系（如相对位置）方面存在显著不足，尽管这些任务对人类而言非常简单。
现有基准测试常因数据重复使用（可能导致训练数据污染）、非对比性设计（允许模型利用捷径）和合成图像（难以推广到现实场景）等问题，未能准确评估模型的空间理解能力。
因此，作者提出 RocketScience 基准测试，旨在通过全新的现实世界对比性数据，更严格地评估 VLMs 的空间推理能力，并揭示性能瓶颈。

## Method

*   **基准设计:** 提出了 RocketScience 基准测试，包含 482 个手动标注的对比性图像-文本对，全部基于现实世界场景（非合成数据），涵盖室内外、不同光照条件等多种情境，确保任务多样性。
*   **对比性结构:** 每个样本包含两张图像和两段描述，仅在物体位置或空间关系上有所不同，迫使模型真正理解空间关系，而非依赖语言统计规律或简单对象检测。
*   **数据采集与标注:** 图像使用 iPhone 13 Mini 在欧洲和美国拍摄，标注由一名作者完成并由两名作者审核，确保一致性；数据类别包括水平位置、垂直位置、深度、接近度和顺序，分布均衡。
*   **评估方法:** 测试了三类模型：CLIP 类双编码器模型、普通多模态大语言模型（MLLMs）以及基于推理的 MLLMs（如使用链式思维提示或强化学习的模型），通过文本分数、图像分数和组分数评估性能。
*   **分解分析:** 分别测试模型在物体定位和空间推理两个阶段的表现，通过定位任务（生成边界框）和推理任务（CoT 提示与非 CoT 提示对比）确定性能瓶颈。
*   **实验细节:** 图像预处理为 1024x1024 分辨率，API 模型温度设为 0 以减少输出波动，本地模型使用贪婪解码确保可重复性，硬件为 T4-GPU。

## Experiment

*   **有效性:** RocketScience 基准测试极具挑战性，大多数开源和商业 VLMs（包括 CLIP 类模型和普通 MLLMs）表现接近随机猜测（group score 0.01-0.24），远低于人类水平（0.95）；而基于链式思维（CoT）或内部推理的模型（如 Gemini 2.5 Pro 和 o4-mini）表现接近人类（group score 0.83 和 0.89），提升显著。
*   **瓶颈分析:** 分解实验表明，模型在物体定位上的表现差距不大（gpt-4o 和 o4-mini 定位准确率均超 90%），但在空间推理阶段，CoT 模型显著优于非 CoT 模型，证明空间推理是主要瓶颈。
*   **设置合理性:** 实验涵盖多种模型类别和任务类别（水平位置、垂直位置、深度、接近度和顺序等），数据分布均衡；人类测试验证任务低歧义性（准确率均值 0.985）；稳定性测试显示即使数据集规模减半，结果标准差仍低，表明结果可靠。
*   **局限性:** 由于成本限制，API 模型仅运行一次，可能存在轻微输出波动；数据集场景复杂度仍低于真实世界，部分对比对可能存在微小视角差异。

## Further Thoughts

链式思维（CoT）提示或内部推理机制显著提升了 VLMs 在空间理解任务上的表现，这启发我们未来模型改进应更多关注结构化推理能力的增强，而非仅提升视觉感知能力；
此外，RocketScience 的对比性设计理念（通过硬性否定避免模型依赖表面统计规律）可推广至其他领域，如情感理解或因果推理任务；
一个发散性思考是，是否可以通过联合训练文本与视觉推理轨迹，或设计自适应推理步骤的模型（根据任务难度动态分配计算资源），进一步提升空间推理能力？