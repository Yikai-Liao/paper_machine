---
title: "CORE-3D: Context-aware Open-vocabulary Retrieval by Embeddings in 3D"
pubDatetime: 2025-09-29T09:43:00+00:00
slug: "2025-09-core-3d-retrieval"
type: "arxiv"
id: "2509.24528"
score: 0.4428693595651356
author: "grok-3-latest"
authors: ["Mohamad Amin Mirzaei", "Pantea Amoie", "Ali Ekhterachian", "Matin Mirzababaei"]
tags: ["LLM", "Vision-Language Model", "Semantic Segmentation", "Object Retrieval", "3D Scene Understanding"]
institution: ["Sharif University of Technology"]
description: "本文提出CORE-3D，一种无需训练的开放词汇3D场景理解框架，通过渐进式分割、上下文感知嵌入和多视角精炼，显著提升了3D语义分割和语言查询对象检索的性能。"
---

> **Summary:** 本文提出CORE-3D，一种无需训练的开放词汇3D场景理解框架，通过渐进式分割、上下文感知嵌入和多视角精炼，显著提升了3D语义分割和语言查询对象检索的性能。 

> **Keywords:** LLM, Vision-Language Model, Semantic Segmentation, Object Retrieval, 3D Scene Understanding

**Authors:** Mohamad Amin Mirzaei, Pantea Amoie, Ali Ekhterachian, Matin Mirzababaei

**Institution(s):** Sharif University of Technology


## Problem Background

3D场景理解对具身人工智能和机器人学至关重要，但现有开放词汇方法在复杂环境中面临挑战：2D分割掩码碎片化、语义嵌入缺乏上下文、多视角预测不一致，导致3D语义地图质量不高，难以支持精准的对象检索和语义分割。本文旨在通过改进分割和嵌入生成，构建高质量的3D语义地图，并实现基于自然语言查询的对象检索。

## Method

* **核心思想**：提出一种无需训练的管道，通过渐进式分割精炼和上下文感知嵌入生成，结合多视角3D一致性约束，构建高质量的3D语义地图，支持开放词汇的语义分割和对象检索。
* **掩码生成**：采用SemanticSAM，通过渐进式粒度调整生成对象级2D掩码。具体方法是按多个粒度级别（从粗到细）生成候选掩码，逐级筛选重叠面积低于阈值的掩码，移除小面积或边缘掩码，并用DBSCAN聚类合并碎片化部分，避免过分割问题。
* **上下文感知嵌入**：为每个掩码提取五种视觉裁剪（掩码裁剪、边界框裁剪、大上下文裁剪、巨大上下文裁剪、周围环境裁剪），通过CLIP编码后进行加权组合形成语义嵌入，其中周围环境嵌入以负权重引入以增强对象与背景的对比性，提升语义上下文的丰富性。
* **3D掩码合并与精炼**：利用深度图和相机姿态将2D掩码投影到3D空间，通过体积交集（IoV）准则合并多视角掩码（需满足高重叠和对称性条件），并用DBSCAN聚类分离空间上分开的实例，确保3D对象表示的一致性和准确性。
* **对象检索**：基于自然语言查询，通过LLM解析查询为结构化形式（目标对象、参考对象、方向约束），用CLIP相似性挖掘候选对象，结合VLM验证和方向约束，最后由LLM基于场景几何和查询推理确定目标对象。

## Experiment

* **语义分割效果**：在Replica数据集上，CORE-3D在mIoU（0.29 vs. 0.27）和fmIoU（0.56 vs. 0.48）上优于最强基线BBQ-CLIP；在ScanNet数据集上，mAcc（0.61 vs. 0.56）、mIoU（0.36 vs. 0.34）和fmIoU（0.46 vs. 0.36）均有提升，定性结果显示分割边界更准确，对困难类别识别更精细。
* **对象检索效果**：在Sr3D+数据集上，CORE-3D在A@0.1（41.8% vs. 34.2%）和A@0.25（35.6% vs. 22.7%）上大幅领先基线，尤其在‘Easy’和‘View-Independent’子集表现突出。
* **实验设置**：实验涵盖合成（Replica）和真实（ScanNet）场景，对象检索考虑查询难度和视角依赖性，评估指标全面（mAcc, mIoU, fmIoU, A@0.1, A@0.25），设置合理且结果显著，验证了方法的有效性。

## Further Thoughts

上下文感知嵌入的加权组合策略启发我们探索多模态任务中不同视角或模态的动态融合方式，例如结合时间维度或多传感器数据；渐进式粒度调整的分割方法提示可以在其他分割任务中尝试多尺度策略；语言查询与3D场景结合的方式表明LLM和VLM协同在空间推理中有潜力，未来可探索更复杂的空间关系建模或动态交互场景。