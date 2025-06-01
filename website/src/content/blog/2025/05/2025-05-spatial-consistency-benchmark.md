---
title: "Toward Memory-Aided World Models: Benchmarking via Spatial Consistency"
pubDatetime: 2025-05-29T01:28:57+00:00
slug: "2025-05-spatial-consistency-benchmark"
type: "arxiv"
id: "2505.22976"
score: 0.4133240304042977
author: "grok-3-latest"
authors: ["Kewei Lian", "Shaofei Cai", "Yilun Du", "Yitao Liang"]
tags: ["World Models", "Spatial Consistency", "Memory Module", "Benchmarking", "Navigation"]
institution: ["Peking University", "MIT"]
description: "本文提出基于循环轨迹的 Minecraft 数据集和基准测试，专门用于评估和推动世界模型在长距离空间一致性方面的研究，为记忆模块设计提供了宝贵资源。"
---

> **Summary:** 本文提出基于循环轨迹的 Minecraft 数据集和基准测试，专门用于评估和推动世界模型在长距离空间一致性方面的研究，为记忆模块设计提供了宝贵资源。 

> **Keywords:** World Models, Spatial Consistency, Memory Module, Benchmarking, Navigation

**Authors:** Kewei Lian, Shaofei Cai, Yilun Du, Yitao Liang

**Institution(s):** Peking University, MIT


## Problem Background

世界模型（World Models）在长时间预测中难以保持空间一致性（Spatial Consistency），导致生成的场景出现幻觉结构、与先前观察矛盾或视觉不连贯等问题，严重影响其在下游任务（如强化学习、自动驾驶和导航）中的可靠性。
论文指出，空间一致性的核心在于记忆模块（Memory Module），但现有数据集和基准测试缺乏对空间一致性的明确约束，难以推动相关研究。

## Method

* **数据集构建（LOOP NAV）**：在 Minecraft 开放世界环境中收集约 250 小时（2000 万帧）的循环导航视频，覆盖 147 个不同位置（村庄、生物群系、结构）。
  * 轨迹设计为 ABA 和 ABCA 两种循环模式，确保模型需从不同时间和视角重新访问相同位置，强制学习空间一致性。
  * 数据收集遵循视觉可区分性（Visual Discriminability）、循环闭合（Loop Closure）和课程式进展（Curriculum-Based Progression）原则，导航范围分为 5、15、30、50 块四个难度级别。
  * 使用 Mineflayer 平台和 A* 算法进行路径规划，限制动作空间（仅前进、跳跃、相机旋转），并以 20 Hz 频率记录 RGB 图像、动作、位置、相机方向和目标坐标等信息。
* **基准测试设计**：提出‘探索-生成’（Explore-then-Generate）评估方法，将轨迹分为探索阶段（提供上下文）和生成阶段（评估重建质量）。
  * 例如，在 ABA 轨迹中，A 到 B 作为探索输入，B 到 A 作为生成目标，专注于返回路径的空间一致性。
  * 评估指标包括 Fréchet Video Distance (FVD)、Learned Perceptual Image Patch Similarity (LPIPS) 和 Structural Similarity Index Measure (SSIM)，并强调定性评估的重要性。

## Experiment

* **有效性**：评估了四个基线模型（Oasis、Mineworld、DIAMOND、NWM），结果显示所有模型在空间一致性上的表现均不理想。SSIM 值普遍较低（0.28-0.40），LPIPS 值较高（0.64-0.86），FVD 值极高（1500-3700），表明生成的视频与真实轨迹差异显著。
* **定性分析**：随着预测步长增加，模型生成的图像逐渐崩溃，出现模糊和视觉失败，反映了缺乏有效记忆模块的局限。
* **实验设置合理性**：实验覆盖不同导航范围（5 到 50 块）和两种轨迹类型（ABA 和 ABCA），数据集的课程式设计提供了渐进式挑战。但测试集仅采样部分轨迹（每个范围 18 条），可能存在代表性偏差，且多为预训练模型直接评估，未针对性训练。
* **结论**：实验验证了当前模型在空间一致性上的不足，也凸显了论文提出数据集和基准测试的必要性。

## Further Thoughts

循环轨迹作为空间一致性监督的设计非常新颖，可推广至自动驾驶或机器人导航中的闭环路径规划；‘探索-生成’评估范式直击空间一致性问题，未来可作为评估框架的通用思路；Minecraft 作为测试平台的潜力值得挖掘，未来可引入动态元素或深度信息增强空间建模。发散性思考：是否可以通过强化学习让模型自适应优化记忆策略，或结合 3D 重建技术构建更强的空间记忆模块？