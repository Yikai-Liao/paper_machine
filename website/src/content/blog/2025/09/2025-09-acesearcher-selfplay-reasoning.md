---
title: "AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play"
pubDatetime: 2025-09-29T02:14:30+00:00
slug: "2025-09-acesearcher-selfplay-reasoning"
type: "arxiv"
id: "2509.24193"
score: 0.6431808978472298
author: "grok-3-latest"
authors: ["Ran Xu", "Yuchen Zhuang", "Zihan Dong", "Jonathan Wang", "Yue Yu", "Joyce C. Ho", "Linjun Zhang", "Haoyu Wang", "Wenqi Shi", "Carl Yang"]
tags: ["LLM", "Retrieval Augmented Generation", "Reasoning", "Self-Play", "Reinforcement Learning"]
institution: ["Emory University", "Georgia Institute of Technology", "Rutgers University", "SUNY Albany", "UT Southwestern Medical Center"]
description: "AceSearcher 通过协作自博弈框架和两阶段微调策略，显著提升了大型语言模型在复杂推理和检索任务中的性能，尤其在资源受限环境下的参数效率。"
---

> **Summary:** AceSearcher 通过协作自博弈框架和两阶段微调策略，显著提升了大型语言模型在复杂推理和检索任务中的性能，尤其在资源受限环境下的参数效率。 

> **Keywords:** LLM, Retrieval Augmented Generation, Reasoning, Self-Play, Reinforcement Learning

**Authors:** Ran Xu, Yuchen Zhuang, Zihan Dong, Jonathan Wang, Yue Yu, Joyce C. Ho, Linjun Zhang, Haoyu Wang, Wenqi Shi, Carl Yang

**Institution(s):** Emory University, Georgia Institute of Technology, Rutgers University, SUNY Albany, UT Southwestern Medical Center


## Problem Background

大型语言模型（LLMs）在复杂推理任务中面临挑战，尤其是在需要多跳检索和整合多条信息进行推理时，现有的检索增强生成（RAG）方法往往局限于简单问题，无法有效应对复杂场景。
核心问题是如何提升LLM在复杂推理任务中的检索和推理能力，尤其是在资源受限的情况下，避免对大型闭源模型的依赖，同时减少对中间标注数据的依赖。

## Method

*   **核心思想:** 提出 AceSearcher，一个基于协作自博弈（cooperative self-play）的框架，通过训练单个大型语言模型（LLM）扮演两个角色——分解者（decomposer）和求解者（solver）——来解决复杂推理任务。
*   **角色分工:** 
    *   分解者负责将复杂问题拆解为一系列子问题模板，指导检索过程，子问题数量和内容可能依赖于前面的答案。
    *   求解者则根据子问题、之前的中间答案和检索到的上下文，逐步生成中间答案和最终答案。
*   **训练策略:** 
    *   第一阶段是监督微调（Supervised Fine-Tuning, SFT），使用多样化的数据集（包括上下文丰富的问答、问题分解和思维链数据）增强模型在检索和推理任务上的基础能力。
    *   第二阶段是强化微调（Reinforcement Fine-Tuning, RFT），通过基于最终答案准确性的奖励机制（Exact Match 和格式奖励）优化模型，使用偏好优化（Direct Preference Optimization, DPO）而非传统的在线强化学习，降低计算成本。
*   **优化细节:** 在 RFT 阶段，通过采样多组分解和求解路径，构建偏好数据集（包含最佳和最差轨迹），联合优化分解者和求解者的策略，确保分解质量有助于最终答案的准确性。
*   **适用性:** 方法适用于不同规模的模型（1.5B 到 32B 参数），并通过迭代偏好优化实现高效训练，避免对中间标注的依赖。

## Experiment

*   **有效性:** AceSearcher 在多跳问答和事实验证任务上平均提升了7.6%的准确率（Exact Match, EM），在10个数据集上的表现优于多种基线模型，包括指令微调模型、提示引导的多步检索模型和强化学习增强的搜索模型。
*   **参数效率:** 在文档级推理任务中，AceSearcher-32B 的性能与参数量是其20倍以上的 DeepSeek-V3 相当；较小规模的 AceSearcher（1.5B 和 8B）也超越了参数量大9倍的现有 RAG 模型，显示出极高的参数效率。
*   **实验设置合理性:** 实验覆盖了三个任务类型（多跳问答、事实验证和文档级推理），涉及10个公开数据集，包含不同规模模型的对比和消融研究，验证了 SFT 和 RFT 阶段的重要性；数据效率研究表明即使在较小数据子集上也能达到较好性能。
*   **局限性:** 实验未探讨实时工具或对话任务的适用性，且检索器在训练和推理中固定，可能会限制进一步优化。

## Further Thoughts

AceSearcher 的自博弈框架和偏好优化机制启发了我，特别是其通过分解者和求解者角色分工协作的思路，可以推广到多模态任务中，如图像与文本的协同推理；此外，固定检索器的限制提示未来可以探索检索与推理的联合优化，例如通过动态调整检索策略进一步提升性能；另一个方向是利用自博弈机制生成高质量训练数据，减少对外部数据集的依赖。