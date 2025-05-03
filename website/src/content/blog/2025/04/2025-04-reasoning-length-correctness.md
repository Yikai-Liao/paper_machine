---
title: "Between Underthinking and Overthinking: An Empirical Study of Reasoning Length and correctness in LLMs"
pubDatetime: 2025-04-30T18:48:06+00:00
slug: "2025-04-reasoning-length-correctness"
type: "arxiv"
id: "2505.00127"
score: 0.8571206656026057
author: "grok-3-latest"
authors: ["Jinyan Su", "Jennifer Healey", "Preslav Nakov", "Claire Cardie"]
tags: ["LLM", "Reasoning", "Chain of Thought", "Sampling", "Preference Optimization"]
institution: ["Cornell University", "Adobe Research", "MBZUAI"]
description: "本文通过实证分析揭示了大型语言模型推理长度与正确性之间的非线性关系，并通过长度偏好优化实验提出了一种在无监督条件下减少生成长度的有效方法，为自适应推理研究提供了新视角。"
---

> **Summary:** 本文通过实证分析揭示了大型语言模型推理长度与正确性之间的非线性关系，并通过长度偏好优化实验提出了一种在无监督条件下减少生成长度的有效方法，为自适应推理研究提供了新视角。 

> **Keywords:** LLM, Reasoning, Chain of Thought, Sampling, Preference Optimization

**Authors:** Jinyan Su, Jennifer Healey, Preslav Nakov, Claire Cardie

**Institution(s):** Cornell University, Adobe Research, MBZUAI


## Problem Background

大型语言模型（LLMs）在推理任务中常被优化为生成较长的推理链（Chain-of-Thought, CoT），以提升性能，但过长的推理可能导致‘过度思考’（Overthinking），引入错误，而对较难问题则可能‘思考不足’（Underthinking），未能生成足够推理步骤。
论文旨在系统研究推理长度与答案正确性之间的关系，揭示模型对问题难度误判导致的长度不适应性问题，并探索优化生成长度以平衡准确性和效率的方法。

## Method

*   **样本级分析（Sample-Level Analysis）**：针对同一问题，通过采样生成多个推理路径（N=10），分析推理长度与正确性之间的关系，排除问题难度差异的影响，重点观察长度变化对准确率的影响趋势。
*   **问题级分析（Question-Level Analysis）**：根据模型对问题的回答正确率，将问题分为易（Easy）、中（Medium）、难（Hard）三类，研究模型是否能根据感知的问题难度调整推理长度，并通过统计分析（如平均长度、相关性）和困惑度（Perplexity）指标，探索长度与正确性之间的关联。
*   **长度偏好优化实验（Length Preference Optimization）**：采用 Simple Preference Optimization (SimPO) 算法，通过构造偏好较短输出的训练对（不依赖正确性标签），在无监督条件下优化模型生成长度，测试是否能在减少 token 数量的同时维持准确性。
*   **实验设置**：使用两个推理模型（DeepSeek-1.5B-Distill 和 DeepScaler-1.5B-Preview）和两个数学推理数据集（GSM8K 和 MATH），结合多种采样参数（如温度 T=1.0）和统计方法（如 t-test、相关性分析）进行全面评估。

## Experiment

*   **样本级结果**：推理长度与正确性呈非单调关系，准确率随长度增加先上升后下降，存在最佳长度范围；超过60%的题目在最短采样响应中已正确，表明过长推理可能引入错误。
*   **问题级结果**：错误回答的平均推理长度显著长于正确回答（如 MATH 数据集上，错误回答超6000 token，正确回答不到3000 token），长度与正确性呈负相关；模型对易题能较好感知难度并调整长度，但对难题常‘思考不足’，未能延长推理。
*   **长度优化效果**：通过 SimPO 训练，生成长度减少30%-60%，准确率保持在可接受范围；长度减少主要来自错误回答的压缩，但正确回答长度也有10%-25%减少，显示过度思考倾向。
*   **实验设置评价**：实验覆盖不同难度问题和多采样结果，设置较为全面，但样本数量（N=10）和模型数量有限，可能影响统计可靠性；数据支持长度优化在效率和准确性间的平衡，但对难题适应性仍需改进。

## Further Thoughts

论文揭示了模型对问题难度的感知和推理长度自适应调整的不足，启发我们可以通过引入难度预测模块或动态终止机制增强模型的自我评估能力；此外，长度偏好优化在无监督条件下的成功应用，提示可以在无标签数据场景中进一步探索简单的偏好训练策略；另一个发散性思考是，是否可以通过多模型协作或分层推理，让小模型快速处理易题，大模型深度推理难题，实现计算资源的动态分配。