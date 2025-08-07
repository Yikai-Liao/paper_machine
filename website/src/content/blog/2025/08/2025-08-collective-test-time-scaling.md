---
title: "CTTS: Collective Test-Time Scaling"
pubDatetime: 2025-08-05T11:19:08+00:00
slug: "2025-08-collective-test-time-scaling"
type: "arxiv"
id: "2508.03333"
score: 0.6573923199539807
author: "grok-3-latest"
authors: ["Zhende Song", "Shengji Tang", "Peng Ye", "Jiayuan Fan", "Tao Chen"]
tags: ["LLM", "Test Time Scaling", "Multi-Agent Collaboration", "Reward Model", "Inference Optimization"]
institution: ["Fudan University", "Shanghai Artificial Intelligence Laboratory", "The Chinese University of Hong Kong"]
description: "本文提出集体测试时扩展（CTTS）范式及CTTS-MM框架，通过多代理和多奖励模型协作，在推理时显著提升大型语言模型性能，超越现有方法和领先模型。"
---

> **Summary:** 本文提出集体测试时扩展（CTTS）范式及CTTS-MM框架，通过多代理和多奖励模型协作，在推理时显著提升大型语言模型性能，超越现有方法和领先模型。 

> **Keywords:** LLM, Test Time Scaling, Multi-Agent Collaboration, Reward Model, Inference Optimization

**Authors:** Zhende Song, Shengji Tang, Peng Ye, Jiayuan Fan, Tao Chen

**Institution(s):** Fudan University, Shanghai Artificial Intelligence Laboratory, The Chinese University of Hong Kong


## Problem Background

大型语言模型（LLM）在测试时扩展（Test-Time Scaling, TTS）是一个新兴研究领域，旨在不增加额外训练成本的情况下提升模型性能。
现有单测试时扩展（STTS）方法，如 Best-of-N 和 Self-Consistency，通常依赖单一代理与单一奖励模型（SA-SR）的交互，存在模型能力上限受限和选择偏见的问题，阻碍了性能进一步提升。
论文提出集体测试时扩展（CTTS），通过多代理和多奖励模型协作，试图突破这些限制，释放预训练模型的潜力。

## Method

*   **核心思想:** 通过多代理和多奖励模型的协作，克服单一测试时扩展（STTS）框架的局限，在推理时动态优化答案选择，提升大型语言模型（LLM）的性能。
*   **范式设计:** 提出了三种CTTS范式：
    *   单代理多奖励模型（SA-MR）：单一LLM代理生成多个候选答案，由多个奖励模型评估选择。
    *   多代理单奖励模型（MA-SR）：多个LLM代理生成候选答案，由单一奖励模型评估选择。
    *   多代理多奖励模型（MA-MR）：多个LLM代理生成候选答案，多个奖励模型协作评估选择，实验证明为最优范式。
*   **具体框架（CTTS-MM）:** 基于MA-MR范式，设计了CTTS-MM框架，包含以下关键组件：
    *   **代理协作搜索（Agent Collaboration Search, ACS）**：从候选模型池中动态选择最有效的LLM代理组合，采用贪婪搜索策略，结合早期停止机制以提高效率，并通过残差聚合（Residual Aggregation）避免信息丢失。具体流程为：初始化搜索集（top-k答案），迭代检查是否添加新候选答案能提升聚合结果，若无改进则终止搜索。
    *   **奖励模型混合（Mixture of Reward Models, MoR）**：构建多样化问题池（Question Pool）作为先验知识，通过先验奖励模型集合选择（Prior Reward Model Ensemble Selection, PRES）基于成对奖励排名（Pair-wise Reward Ranking, PRR）指标，动态选择最优奖励模型或其加权组合（如线性加权、Softmax加权），为搜索提供高质量反馈。
    *   **统一管道:** 将ACS和MoR整合到搜索-奖励-搜索的迭代优化流程中，确保答案质量持续提升。
*   **关键创新:** 不依赖手动选择模型或固定组合，而是通过动态适配任务需求实现泛化性，同时多模型协作突破单一模型性能上限。

## Experiment

*   **有效性:** CTTS-MM在七个主流基准数据集（涵盖数学推理、复杂问答、指令跟随和代码生成）上显著优于现有方法，平均准确率达78.84%，相比Best-of-N提升4.82%，Self-Consistency提升7.68%，甚至超越闭源模型GPT-4.1（提升7.06%）和开源模型DeepSeek-R1-Distill-Qwen-32B（提升7.76%）。
*   **范式对比:** MA-MR范式平均性能比SA-SR提升10.84%，比MA-SR提升1.9%，验证了多代理和多奖励模型协作的必要性和优越性。
*   **组件消融:** 各组件（ACS、MoR、残差聚合）均有贡献，组合后效果最佳，例如在MATH-500上从90.8%提升至93.0%。
*   **扩展性:** 随着奖励模型数量增加，性能持续提升（如LiveCodeBench上从41.8%提升至52.28%），表明MoR具有良好的扩展性。
*   **鲁棒性:** 跨领域问题池对性能影响较小（如MATH数据集使用MBPP问题池仅下降0.4%），显示方法的稳定性。
*   **实验设置:** 覆盖多领域任务，使用10个开源LLM和8个奖励模型，数据集划分为验证集和测试集以构建问题池，设置全面合理。

## Further Thoughts

多代理和多奖励模型协作的思路启发我们可以在模型池中引入更多异构模型（如不同架构、不同训练数据的模型），以进一步提升协作效果；动态选择机制（ACS和MoR）可以通过强化学习或元学习优化，可能提升搜索效率和选择精度；问题池作为先验知识的泛化性思路可扩展到其他任务，如构建领域无关的评估基准，用于奖励模型预筛选或自适应调整。