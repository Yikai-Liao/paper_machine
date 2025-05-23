---
title: "Safety Subspaces are Not Distinct: A Fine-Tuning Case Study"
pubDatetime: 2025-05-20T10:41:49+00:00
slug: "2025-05-safety-subspace-finetuning"
type: "arxiv"
id: "2505.14185"
score: 0.6081231496124564
author: "grok-3-latest"
authors: ["Kaustubh Ponkshe", "Shaan Shah", "Raghav Singhal", "Praneeth Vepakomma"]
tags: ["LLM", "Safety Alignment", "Fine-Tuning", "Subspace Analysis", "Representation Space"]
institution: ["Mohamed bin Zayed University of Artificial Intelligence", "University of California San Diego", "Massachusetts Institute of Technology"]
description: "本文通过实验证明大型语言模型的安全对齐不具备可分离的几何子空间，挑战了基于子空间的防御策略假设。"
---

> **Summary:** 本文通过实验证明大型语言模型的安全对齐不具备可分离的几何子空间，挑战了基于子空间的防御策略假设。 

> **Keywords:** LLM, Safety Alignment, Fine-Tuning, Subspace Analysis, Representation Space

**Authors:** Kaustubh Ponkshe, Shaan Shah, Raghav Singhal, Praneeth Vepakomma

**Institution(s):** Mohamed bin Zayed University of Artificial Intelligence, University of California San Diego, Massachusetts Institute of Technology


## Problem Background

大型语言模型（LLMs）通过指令微调和人类反馈强化学习（RLHF）实现安全对齐，但这种对齐非常脆弱，进一步微调（即使在良性数据上）也可能破坏安全性，重新引入有害行为。
论文的出发点是探究安全对齐是否对应于权重空间或激活空间中的特定几何子空间，如果存在这样的子空间，是否可以通过隔离或保护这些子空间来防御微调带来的不对齐风险。

## Method

* **核心思想：** 通过系统性实验验证安全对齐是否在权重空间或激活空间中具有可识别的几何结构（子空间），以判断基于子空间的防御策略是否可行。
* **具体方法：** 设计了四组实验，覆盖权重空间和激活空间两个维度：
  * **权重空间分析（实验1）：** 使用奇异值分解（SVD）从对齐矩阵（基模型到对齐模型的权重差异）中提取主导方向，定义对齐子空间；分析在有益和有害数据集上微调的更新是否集中在这些子空间，并测试其与安全性的关联。
  * **污染数据微调（实验2）：** 在混合有害和良性数据的污染数据集上微调，通过正交投影移除与对齐子空间重叠的更新部分，测试是否能选择性地减少有害行为，同时保留实用性。
  * **子空间重叠比较（实验3）：** 使用模式子空间重叠（Mode Subspace Overlap, MSO）指标，比较对齐更新、有益更新和有害更新的主导子空间是否共享一致结构，以判断是否存在安全特异性子空间。
  * **激活空间分析（实验4）：** 比较有益和有害提示在模型内部表征空间中的激活模式，计算其子空间重叠程度，判断安全相关输入是否具有独特的几何特性。
* **技术细节：** 引入能量保留比（Energy-Kept Ratio）量化更新与子空间的重叠程度，使用投影操作（如平行投影和正交投影）操控更新方向，并以随机子空间作为对照组，确保结果可靠性。

## Experiment

* **有效性：** 实验结果一致表明，无论在权重空间还是激活空间中，都没有发现专门编码安全对齐的子空间；对齐子空间（如 Top-K 方向）对行为影响显著，但同时放大有益和有害行为，反映的是通用学习敏感性，而非安全特异性。
* **实验设置：** 实验覆盖五个开源 LLMs（如 LLaMA 3.2 1B, Qwen-2.5 系列），在多个数据集（如 MetaMathQA, BeaverTails, AdvBench）上测试，设置全面且合理；污染数据实验模拟现实场景，增强了实践意义；随机子空间对照组确保结果不是偶然。
* **局限性：** 实验局限于中小型模型，未涉及更大规模模型或非 RLHF 对齐方法，可能影响泛化性；仅关注线性子空间，未探索非线性表征的潜在影响。

## Further Thoughts

论文揭示安全对齐与模型通用学习动态高度纠缠，而非几何可分离，这启发我们未来可能需要从全局学习机制入手，设计动态约束的微调方法，限制更新对高影响方向的过度依赖；此外，高影响方向对行为变化的显著作用提示可以探索激活级别的控制策略，或开发非线性表征分析工具，捕捉安全行为的潜在模式。