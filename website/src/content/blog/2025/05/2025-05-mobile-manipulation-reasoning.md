---
title: "MORE: Mobile Manipulation Rearrangement Through Grounded Language Reasoning"
pubDatetime: 2025-05-05T21:26:03+00:00
slug: "2025-05-mobile-manipulation-reasoning"
type: "arxiv"
id: "2505.03035"
score: 0.4243383826100432
author: "grok-3-latest"
authors: ["Mohammad Mohammadi", "Daniel Honerkamp", "Martin Büchner", "Matteo Cassinelli", "Tim Welschehold", "Fabien Despinoy", "Igor Gilitschenski", "Abhinav Valada"]
tags: ["Mobile Manipulation", "Scene Graph", "Language Reasoning", "Task Planning", "Environment Representation"]
institution: ["University of Freiburg", "University of Toronto", "Toyota Motor Europe"]
description: "MORE 提出了一种基于场景图和语言推理的移动操作方法，通过任务相关子图过滤和 LLM 规划，首次在 BEHAVIOR-1K 基准测试中完成大量日常重新排列任务，并在现实世界中展现应用潜力。"
---

> **Summary:** MORE 提出了一种基于场景图和语言推理的移动操作方法，通过任务相关子图过滤和 LLM 规划，首次在 BEHAVIOR-1K 基准测试中完成大量日常重新排列任务，并在现实世界中展现应用潜力。 

> **Keywords:** Mobile Manipulation, Scene Graph, Language Reasoning, Task Planning, Environment Representation

**Authors:** Mohammad Mohammadi, Daniel Honerkamp, Martin Büchner, Matteo Cassinelli, Tim Welschehold, Fabien Despinoy, Igor Gilitschenski, Abhinav Valada

**Institution(s):** University of Freiburg, University of Toronto, Toyota Motor Europe


## Problem Background

自主移动操作（Mobile Manipulation）在大型、未知环境中执行长距离重新排列任务时，面临场景动态性、未探索区域和错误恢复等多重挑战。
现有方法在处理大量对象和大规模环境时，规划时间会爆炸式增长，或因信息过载产生幻觉（Hallucinations），导致可靠性下降。
此外，现有研究多局限于已知环境或特定任务，缺乏对未知环境（包括室内和室外）的泛化能力。
论文的出发点是通过自然语言驱动的机器人推理，解决零样本（Zero-Shot）任务规划问题，特别是在 BEHAVIOR-1K 基准测试中的日常活动任务。

## Method

* **核心思想**：通过结合分层场景图（Scene Graph）和大型语言模型（LLM），在未知的大型环境中实现高效的任务规划，重点解决重新排列任务（Rearrangement Tasks）。
* **场景表示（Scene Representation）**：构建分层 3D 场景图，统一表示室内外环境，使用 Voronoi 图划分导航空间，并通过稀疏化算法减少节点数量以提高计算效率；场景图包含对象实例和属性信息，并将其转化为自然语言描述传递给语言模型。
* **场景图过滤（Scene Graph Filtering）**：提出基于 LLM 的过滤机制，从完整场景中提取与任务相关的子图（Subgraph），通过忽略无关对象和区域，将规划问题限制在可管理的范围内，显著减少幻觉和计算复杂度。
* **任务规划（Task Planning）**：利用 LLM 进行高层次任务规划，基于过滤后的子图、任务描述和机器人状态，动态生成子策略序列（如导航、探索、操作）；采用模型预测控制（MPC）风格，在每次场景更新后重新规划，确保适应动态环境。
* **子策略执行（Subpolicy Execution）**：定义多种对象中心的子策略，包括探索（Explore）、导航（Navigate）、抓取（Grasp）、放置（Place）等，通过‘魔法动作’（Magic Actions）简化物理模拟，提高评估效率。
* **关键创新**：不过度依赖预先已知的场景布局，而是通过动态场景图更新和任务相关过滤，适应未知环境；同时支持实例级操作和属性推理，提升任务执行精度。

## Experiment

* **有效性**：在 BEHAVIOR-1K 基准测试的 81 个重新排列任务中，MORE 取得了 48.1% 的成功率（SR），50.6% 的任务完成率（TTC），70.1% 的任务进度（TP），以及 80.1% 的相对任务进度（rTP），显著优于基线方法（如 BUMBLE 和 MoMa-LLM）。
* **优越性**：通过场景图过滤机制，MORE 有效缩小了搜索空间，减少了幻觉问题，尤其在处理大规模环境和多对象任务时表现突出；相比基线方法，其场景表示和实例区分能力显著提升了任务规划的可靠性。
* **实验设置**：实验涵盖模拟环境和现实世界测试，模拟中采用‘魔法动作’加速评估，现实世界在多房间公寓中测试了代表性任务（如设置餐桌、清理垃圾），验证了方法的可迁移性；评估指标包括成功率、任务完成率等，设置较为全面。
* **局限性**：任务描述模糊性可能导致模型无法判断任务是否完成；模拟器中对象生成问题（如生成在不可达区域）以及现实世界中 LLM 推理时间较长，影响运行效率；这些问题表明方法仍有改进空间。

## Further Thoughts

场景图过滤机制的理念可以通过 LLM 动态提取任务相关信息，未来可扩展到其他领域，如自然语言处理中的上下文聚焦或多模态任务中的信息精炼；
室内外统一表示的思路启发我们可以在跨域机器人任务中引入更多动态感知数据（如实时更新场景图），以增强适应性；
任务模糊性问题提示未来可通过人机交互或上下文推理机制，进一步提升任务规划的鲁棒性。