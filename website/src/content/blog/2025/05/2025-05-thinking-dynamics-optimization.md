---
title: "Don't Think Longer, Think Wisely: Optimizing Thinking Dynamics for Large Reasoning Models"
pubDatetime: 2025-05-27T20:59:29+00:00
slug: "2025-05-thinking-dynamics-optimization"
type: "arxiv"
id: "2505.21765"
score: 0.7050014710333109
author: "grok-3-latest"
authors: ["Sohyun An", "Ruochen Wang", "Tianyi Zhou", "Cho-Jui Hsieh"]
tags: ["LLM", "Reasoning Efficiency", "Thinking Patterns", "Preference Optimization", "Test Time Scaling"]
institution: ["University of California, Los Angeles", "University of Maryland, College Park"]
description: "本文提出动态思维模式优化框架（DTO），通过分割和优化推理路径中的思维模式，显著提升大型推理模型的推理效率和准确率，同时减少计算开销。"
---

> **Summary:** 本文提出动态思维模式优化框架（DTO），通过分割和优化推理路径中的思维模式，显著提升大型推理模型的推理效率和准确率，同时减少计算开销。 

> **Keywords:** LLM, Reasoning Efficiency, Thinking Patterns, Preference Optimization, Test Time Scaling

**Authors:** Sohyun An, Ruochen Wang, Tianyi Zhou, Cho-Jui Hsieh

**Institution(s):** University of California, Los Angeles, University of Maryland, College Park


## Problem Background

大型推理模型（Large Reasoning Models, LRMs）通过强化学习优化最终答案准确率显著提升了推理能力，但也带来了‘过度思考’（overthinking）问题，即生成过长或过于复杂的推理路径，导致计算资源浪费甚至性能下降。
作者假设这种低效源于模型无法在推理过程中动态选择合适的‘思维模式’（thinking patterns），因此论文致力于解决如何优化推理路径以提高效率，同时保持或提升准确率的关键问题。

## Method

*   **核心思想:** 提出动态思维模式优化框架（DTO），通过将推理路径分割为模块化的‘思维模式’片段，评估并优化其贡献，减少无用推理步骤，提升效率。
*   **具体实现步骤:**
    *   **推理路径分割:** 将模型生成的推理轨迹分割为不同的思维模式（如假设生成、自验证、中间总结），通过语言线索（如‘Wait’、‘Alternatively’）识别这些片段。
    *   **终止点确定:** 使用蒙特卡洛估计方法计算每个思维模式后生成正确答案的概率（p_i），找到最早超过预设阈值（T）的点作为推理终止点，截断后续无用部分。
    *   **剪枝与优化:** 借助辅助大语言模型（LLM）评估每个思维模式的贡献，移除冗余、无意义或有害的片段，同时通过快速解码验证确保剪枝后仍能得出正确答案。
    *   **偏好优化:** 基于优化后的推理路径和次优路径构建成对数据集，使用偏好优化技术（如 SimPO）进一步训练模型，引导其生成更高效的推理行为。
*   **创新点:** 相较于以往基于启发式截断或简单长度指标的方法，DTO 在片段级别上进行细粒度优化，明确建模每个推理片段的贡献，确保推理路径既简洁又有效。

## Experiment

*   **效率提升:** 在多个数学推理基准数据集（如 MATH、GSM8K、Gaokao、AMC、AIME）上，DTO 方法显著降低计算开销，例如在 DeepSeek-R1 模型上注意力计算 FLOPs 减少了 47%，在 DeepScaleR 上减少了 40%，同时保持了正确答案的准确率。
*   **准确率改进:** 对于原本错误的推理路径，DTO 通过优化思维模式将部分错误答案转化为正确答案，准确率提升了 15.6%（DeepSeek-R1）和 7.8%（DeepScaleR）。
*   **对比优势:** 与基线方法（如 Fast Prompt、SFT、O1-Pruner、DAST、FCS+Ref）相比，DTO 在几乎所有数据集上取得了最高的效率指标（η），尤其在困难数据集（如 AMC 和 AIME）上，准确率提升约 7%，token 使用量减少约 1700 个。
*   **实验设置合理性:** 实验覆盖多种基准数据集和两款不同模型，评估指标包括准确率、token 使用量和效率指标，设置了多种对比方法，采样参数和超参数明确，增强了可重复性；此外，实验还在非数学领域（MMLU-Pro）验证了泛化性。
*   **局限性:** 实验依赖多响应采样可能在资源受限场景下不适用，且主要在数学推理领域验证，泛化性需进一步探索。

## Further Thoughts

论文中‘思维模式’的模块化概念及其动态优化是一个值得关注的启发，将推理路径拆分为独立功能性片段并通过概率估计和辅助模型评估其贡献，为推理效率研究提供了细粒度优化的新思路；此外，错误推理路径中往往包含有价值的中间步骤，通过适当终止和重组可以‘挽救’错误答案，这提示可以在推理中引入更多中间检查点或自适应调整机制；最后，偏好优化结合成对数据集的使用为模型学习高效推理行为提供了有效途径，值得在开放域问答等任务中进一步探索。