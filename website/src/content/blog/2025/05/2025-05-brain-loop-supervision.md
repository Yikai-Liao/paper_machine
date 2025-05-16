---
title: "Human-like Cognitive Generalization for Large Models via Brain-in-the-loop Supervision"
pubDatetime: 2025-05-14T02:39:10+00:00
slug: "2025-05-brain-loop-supervision"
type: "arxiv"
id: "2505.09085"
score: 0.44254884410014167
author: "grok-3-latest"
authors: ["Jiaxuan Chen", "Yu Qi", "Yueming Wang", "Gang Pan"]
tags: ["LLM", "Representation Learning", "Cognitive Generalization", "Brain Supervision", "Graph Matching"]
institution: ["Zhejiang University"]
description: "本文提出脑机循环监督学习框架，通过少量脑信号将人类大脑概念结构转移到深度神经网络中，显著提升了模型在抽象概念理解和复杂认知任务上的表现。"
---

> **Summary:** 本文提出脑机循环监督学习框架，通过少量脑信号将人类大脑概念结构转移到深度神经网络中，显著提升了模型在抽象概念理解和复杂认知任务上的表现。 

> **Keywords:** LLM, Representation Learning, Cognitive Generalization, Brain Supervision, Graph Matching

**Authors:** Jiaxuan Chen, Yu Qi, Yueming Wang, Gang Pan

**Institution(s):** Zhejiang University


## Problem Background

当前深度神经网络（DNNs）和大型语言模型（LLMs）在图像和自然语言理解任务上表现出色，但难以达到人类水平的认知能力，尤其是在抽象概念理解、推理和适应新场景方面。
单纯通过增加模型参数和训练数据规模无法有效提升抽象概念的表现，甚至可能导致性能停滞或下降，揭示了AI在认知泛化上的根本瓶颈。
作者提出通过引入人类大脑的结构化表征作为监督信号，增强模型对抽象概念的理解和泛化能力。

## Method

*   **核心思想:** 提出一种脑机循环监督学习（Brain-in-the-loop Supervision）框架，通过少量脑信号（fMRI数据）将人类大脑的概念结构转移到深度神经网络（DNNs）中，以增强其认知能力。
*   **具体实现:** 
    *   **表征对齐:** 利用图匹配（Graph Matching）技术，通过最优传输（Optimal Transport）和Gromov-Wasserstein距离等方法，将DNNs的图像嵌入与人类大脑的fMRI信号嵌入进行结构对齐，优化两个表征之间的结构相似性。
    *   **框架设计:** 包含三个可学习模块：fMRI编码器（基于ViT）、图像编码器（基于MLP）和图神经网络（GNN），通过迭代优化寻找最佳对应关系并更新表征。
    *   **训练过程:** 使用少量对象类别（150个训练类别）进行对齐训练，采用可微分技术（如Gumbel-Softmax）确保端到端优化，不依赖任务特定损失函数。
    *   **泛化目标:** 通过对齐少量数据，使模型的概念结构泛化到未见过的概念（50个测试类别），模拟人类大脑的结构化认知模式。
*   **创新点:** 不同于传统的规模扩展方法，该框架直接从人类大脑中提取结构化知识作为监督信号，注重表征结构的对齐而非单纯的数据驱动学习。

## Experiment

*   **有效性:** 脑机循环监督显著提升了模型在抽象概念理解上的表现，例如在单样本学习（One-shot Learning）任务中，CLIP-base模型在抽象概念分类准确率上提升了20.5%，超越参数量大4.9倍的基线模型。
*   **泛化能力:** 对未见过概念的表征距离减少（Pearson相关系数高达-0.91），表明对齐效果能泛化到新数据，且在分布外识别（OOD Recognition）任务中性能提升了11.5%。
*   **任务多样性:** 在多种复杂任务（如零样本学习、脑图检索、图像检索）中均有显著提升，概念层次结构与人类WordNet高度一致，三元组奇异判断任务中与人类判断的符合率从43.74%提升到49.94%。
*   **实验设置:** 实验覆盖了多个DNN架构（SimCLR、CLIP、DINOv2）、不同参数规模和多个被试的脑信号，统计显著性检验（P值极低）支持结论，但数据规模（150个训练类别）和被试数量（仅3人）可能限制普适性。
*   **合理性与局限:** 实验设计较为全面，但依赖特定被试的fMRI数据，可能因个体差异影响泛化能力，未来需探索更大规模的神经影像数据。

## Further Thoughts

本文通过脑机循环监督将人类大脑的概念结构引入AI模型，启发我们思考是否可以进一步探索其他生物系统的结构化表征（如动物视觉系统）来增强模型能力；此外，少量数据对齐即可泛化到未见过概念，是否意味着人类大脑表征具有通用性，可作为AI的先验知识？未来是否能从更大规模神经影像数据中提炼普适概念结构，或扩展到其他模态（如语言、动作）与大脑信号的对齐，构建更全面的人类认知模型？