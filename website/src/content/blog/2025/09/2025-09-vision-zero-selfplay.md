---
title: "Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play"
pubDatetime: 2025-09-29T21:55:55+00:00
slug: "2025-09-vision-zero-selfplay"
type: "arxiv"
id: "2509.25541"
score: 0.7017577170912062
author: "grok-3-latest"
authors: ["Qinsi Wang", "Bo Liu", "Tianyi Zhou", "Jing Shi", "Yueqian Lin", "Yiran Chen", "Hai Helen Li", "Kun Wan", "Wentian Zhao"]
tags: ["VLM", "Self-Play", "Reinforcement Learning", "Gamification", "Reasoning"]
institution: ["Duke University", "National University of Singapore", "University of Maryland", "Adobe Inc."]
description: "Vision-Zero 提出了一种无需人类标注的视觉-语言模型自我提升框架，通过战略性自博弈游戏和迭代优化算法显著提升跨任务性能并大幅降低训练成本。"
---

> **Summary:** Vision-Zero 提出了一种无需人类标注的视觉-语言模型自我提升框架，通过战略性自博弈游戏和迭代优化算法显著提升跨任务性能并大幅降低训练成本。 

> **Keywords:** VLM, Self-Play, Reinforcement Learning, Gamification, Reasoning

**Authors:** Qinsi Wang, Bo Liu, Tianyi Zhou, Jing Shi, Yueqian Lin, Yiran Chen, Hai Helen Li, Kun Wan, Wentian Zhao

**Institution(s):** Duke University, National University of Singapore, University of Maryland, Adobe Inc.


## Problem Background

当前视觉-语言模型（VLMs）的训练高度依赖人工标注数据集，导致数据稀缺（多模态标注成本极高）和知识上限（模型能力受限于人类监督）两大问题。
论文旨在通过自博弈机制，设计一个无需人类标注的训练框架，让 VLMs 通过自主生成的游戏数据实现自我提升，解决高成本和能力瓶颈问题。

## Method

*   **核心思想:** 提出 Vision-Zero 框架，通过‘谁是间谍’风格的视觉游戏，让模型在多角色互动中进行策略性推理，自主生成训练数据以实现自我提升。
*   **战略性自博弈环境:** 模型扮演‘间谍’或‘平民’角色，基于图像差异提供线索并投票识别间谍，间谍需隐藏身份，平民需推理找出差异；这种互动生成训练数据，无需人工标注。
*   **任意图像输入:** 框架支持从任意图像对（如 CLEVR 合成场景、图表、真实世界图像）生成游戏，增强模型跨领域泛化能力；具体通过自动化图像编辑工具或程序化渲染生成图像对，确保输入多样性。
*   **迭代自博弈策略优化（Iterative-SPO）:** 提出一种新型训练算法，交替进行自博弈和可验证奖励的强化学习（RLVR）；在线索阶段，采用零和奖励机制优化策略性线索生成；在决策阶段，通过群体归一化奖励优化投票准确性；动态切换阶段以避免性能平台期，确保长期持续改进。
*   **关键优势:** 不依赖任务特定数据，训练过程完全自动化，同时培养视觉推理、空间理解和语言策略等多能力，缓解跨能力负迁移问题。

## Experiment

*   **有效性:** Vision-Zero 在多个基准数据集上显著优于基线方法，例如在 Qwen2.5-VL-7B 模型上，MathVision 提升约 3%，ChartQA 提升约 1.1%，RealWorldQA 提升约 0.4%，超越依赖昂贵人工标注数据集的 SOTA 方法。
*   **泛化性:** 在不同类型数据集（CLEVR、图表、真实世界图像）上训练的模型均表现出跨任务性能提升，表明框架的领域无关性；例如 VisionZero-Qwen-7B (Chart) 在图表任务上有针对性提升，而 (CLEVR) 在视觉中心任务上表现更优。
*   **成本效率:** 数据集构建成本极低，CLEVR 数据仅需 6 小时 GPU 时间，图表和真实世界数据生成成本仅为几十美元，远低于传统方法数月的人工标注成本。
*   **实验设置合理性:** 实验覆盖多种模型（Qwen2.5-VL-7B, InternVL3-8B/14B）和 14 个任务，对比基线包括依赖人工数据的 RLVR 方法和游戏数据训练的 ViGaL，设置全面；但未深入探讨不同图像类型对性能提升的具体影响权重，可能是一个小局限。

## Further Thoughts

Vision-Zero 的自博弈与 RLVR 交替训练机制启发我们可以在其他多模态任务（如语音-文本模型）中设计类似的多阶段优化策略，动态平衡探索与稳定；其领域无关性设计也提示在机器人控制等领域利用廉价数据生成通用训练环境；此外，通过游戏机制同时提升多能力的思路可应用于多任务学习，引入竞争性互动以增强模型综合能力。