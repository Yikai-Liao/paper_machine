---
title: "Judging LLMs on a Simplex"
pubDatetime: 2025-05-28T04:50:41+00:00
slug: "2025-05-llm-judging-simplex"
type: "arxiv"
id: "2505.21972"
score: 0.3492147351327167
author: "grok-3-latest"
authors: ["Patrick Vossler", "Fan Xia", "Yifan Mai", "Jean Feng"]
tags: ["LLM", "Evaluation Framework", "Ranking Identifiability", "Uncertainty Quantification", "Bayesian Inference"]
institution: ["University of California, San Francisco", "Stanford University"]
description: "本文通过概率单纯形几何框架揭示 LLM 评判中排名可识别性的理论边界，并提出贝叶斯推断方法整合两种不确定性，显著提升排名准确性和不确定性校准。"
---

> **Summary:** 本文通过概率单纯形几何框架揭示 LLM 评判中排名可识别性的理论边界，并提出贝叶斯推断方法整合两种不确定性，显著提升排名准确性和不确定性校准。 

> **Keywords:** LLM, Evaluation Framework, Ranking Identifiability, Uncertainty Quantification, Bayesian Inference

**Authors:** Patrick Vossler, Fan Xia, Yifan Mai, Jean Feng

**Institution(s):** University of California, San Francisco, Stanford University


## Problem Background

随着大型语言模型（LLMs）的数量和更新频率增加，自由文本输出的自动化评估成为一大挑战，尤其是在缺乏金标准标签的情况下，如何使用 LLMs 自身作为评判者（LLM-as-a-Judge）来准确恢复候选模型的真实排名仍缺乏理论支持。
论文聚焦于解决排名可识别性问题，探索在不同评分系统下（如二元评分和多级评分）能否通过 LLM 评判者恢复真实排名，并关注排名不确定性来源——数据固有的随机性（aleatoric uncertainty）和假设不确定性（epistemic uncertainty）。

## Method

*   **几何框架**：提出了一种新颖的概率单纯形（probability simplex）表示方法，将评判者和候选模型的评分分布映射为单纯形上的点，利用几何性质（如重心坐标、凸包）直观分析排名可识别性（identifiability）。
*   **理论分析**：通过几何论证，研究二元评分系统（2-level scoring）和多级评分系统（3+ level scoring）的排名可识别性条件，发现二元评分系统在较弱的一致性假设（constancy assumptions）和单调性假设下即可识别排名，而多级评分系统在缺乏强先验知识时排名不可识别。
*   **贝叶斯推断框架**：设计了一种贝叶斯模型，通过先验分布显式控制评判者一致性假设和评判者质量，整合 aleatoric 和 epistemic 两种不确定性；同时引入随机效应（random effects）放松一致性假设，并通过敏感性分析评估不同假设对排名估计的影响。
*   **实现细节**：贝叶斯推断使用 Hamiltonian Monte Carlo（HMC）进行后验采样，计算效率高（普通笔记本电脑上不到一分钟），并通过调整超参数（如随机效应强度、评判者质量先验）适应不同任务特性。

## Experiment

*   **数据集与设置**：在多个基准数据集（GPQA、MMLU Pro、MTBench、TLDR、Omni-MATH）上进行实验，涵盖二元评分和多级评分任务，考虑任务难度分层和评判者自我偏好等因素，实验设计全面合理。
*   **排名准确性**：贝叶斯排名方法在所有数据集上均显著优于基线（如简单平均、Bradley-Terry），例如在 GPQA 上 Spearman 相关性达到 0.818（调整自我评分后），相比简单平均（0.720）提升明显。
*   **不确定性校准**：贝叶斯方法的 95% 可信区间覆盖率大幅提高，例如在 GPQA 上达到 0.852，而简单平均仅为 0.093，表明其对不确定性量化更加可靠。
*   **稳健性与局限**：排名估计对一致性假设放松和超参数变化表现出较强稳健性，尤其在 MTBench 等区分度高数据集中接近完美；但在 Omni-MATH 等高难度任务中排名不确定性较高，通过调整随机效应参数可部分缓解（覆盖率从 0.561 提升至 0.719）。

## Further Thoughts

几何框架将复杂评分分布映射到概率单纯形上，为评估问题提供了直观工具，这种方法或可推广至其他多维度评估任务；区分 aleatoric 和 epistemic 不确定性的思路启发我们在缺乏金标准时通过先验约束问题空间；此外，放松评判者一致性假设的随机效应模型为识别评判偏见提供了新视角，或许可以结合更复杂的评判者行为模型进一步提升评估鲁棒性。