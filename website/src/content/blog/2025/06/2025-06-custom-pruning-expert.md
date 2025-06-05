---
title: "Pruning General Large Language Models into Customized Expert Models"
pubDatetime: 2025-06-03T07:47:30+00:00
slug: "2025-06-custom-pruning-expert"
type: "arxiv"
id: "2506.02561"
score: 0.8403263687679162
author: "grok-3-latest"
authors: ["Yiran Zhao", "Guizhen Chen", "Kenji Kawaguchi", "Lidong Bing", "Wenxuan Zhang"]
tags: ["LLM", "Model Pruning", "Customization", "Efficiency"]
institution: ["National University of Singapore", "Nanyang Technological University, Singapore", "DAMO Academy, Alibaba Group, Singapore", "MiroMind", "Singapore University of Technology and Design"]
description: "本文提出 Cus-Prun 方法，通过‘语言’、‘领域’和‘任务’三个维度的细粒度神经元剪枝，将大型语言模型定制为轻量级专家模型，无需后训练即可显著提升专家能力并保留通用性能。"
---

> **Summary:** 本文提出 Cus-Prun 方法，通过‘语言’、‘领域’和‘任务’三个维度的细粒度神经元剪枝，将大型语言模型定制为轻量级专家模型，无需后训练即可显著提升专家能力并保留通用性能。 

> **Keywords:** LLM, Model Pruning, Customization, Efficiency

**Authors:** Yiran Zhao, Guizhen Chen, Kenji Kawaguchi, Lidong Bing, Wenxuan Zhang

**Institution(s):** National University of Singapore, Nanyang Technological University, Singapore, DAMO Academy, Alibaba Group, Singapore, MiroMind, Singapore University of Technology and Design


## Problem Background

大型语言模型（LLMs）因其庞大的参数规模而带来高昂的计算成本和资源需求，现有剪枝方法多关注通用能力的保留，忽视了用户对特定场景（如语言、领域、任务）的需求，导致粗粒度剪枝可能移除关键参数，影响性能或需大量后训练恢复能力。
本文旨在解决如何在无需后训练的情况下，将大型通用模型剪枝为针对特定场景的轻量级专家模型，同时尽可能保留通用能力和专家能力。

## Method

*   **核心思想:** 提出 Cus-Prun（Custom Pruning）方法，通过‘语言’、‘领域’和‘任务’三个维度进行细粒度的神经元级剪枝，定制化地构建轻量级专家模型，而无需后训练。
*   **具体步骤:** 
    *   **维度定位:** 将专家模型需求定义在‘语言’（如英语、德语）、‘领域’（如医疗、法律）和‘任务’（如问答、摘要）三个维度上，支持单维度、双维度或三维度定制。
    *   **语料构建与神经元评估:** 针对每个维度，构建对应语料库（如特定语言跨领域和任务的文档集合），通过移除神经元后对模型输出影响的评估（基于输出变化阈值），识别‘无关神经元’（irrelevant neurons）。
    *   **多维度剪枝:** 取各维度无关神经元的交集进行剪枝，确保保留对目标场景重要的参数，同时移除冗余部分。
    *   **实现细节:** 采用并行神经元检测方法加速识别过程，并通过预定义剪枝比率（如 25%）控制剪枝程度。
*   **创新点:** 相较于传统粗粒度剪枝（如移除整个层或模块），Cus-Prun 在神经元级别操作，避免破坏特定场景能力；同时支持灵活的定制粒度，适应多样化的用户需求。

## Experiment

*   **有效性:** Cus-Prun 在多个模型（如 Llama3-8B, Mistral-12B, Llama2-13B, Llama3-70B）上，以 25% 剪枝比率显著优于基线方法（如 SliceGPT, ShortGPT, LLM-Pruner），例如在 Llama3-8B 上，多语言专家能力平均得分为 38.9（其他方法最高 15.5），多领域和多任务设置中也有类似提升。
*   **通用能力保留:** 在通用能力（如 MMLU, GSM8K）上损失较小，例如 Llama3-8B 的通用能力平均得分为 51.4，远高于其他方法的 17.7-22.3。
*   **鲁棒性与全面性:** 实验覆盖高资源和低资源语言（如德语、泰语），不同领域（如医疗、电商）和任务（如摘要、问答），并在不同剪枝比率（25% 至 45%）下测试，即使在高剪枝比率下仍保持较高性能，尤其在生成任务（如 XLSum）中表现突出。
*   **实验设置合理性:** 实验设计考虑了多维度场景，数据集选择具有代表性（如多语言的 MGSM, XQuAD，多领域的 MedMCQ），并通过对比分析验证了方法的优越性。

## Further Thoughts

Cus-Prun 的多维度定制化思路启发我们是否可以进一步扩展维度（如用户偏好、输入格式）以实现更精细的模型定制；其基于神经元功能特异性的剪枝方法可延伸至模型解释性或推理效率优化研究；此外，无需后训练的设计为其他压缩技术（如量化、蒸馏）提供了新思路，特别是在资源受限环境下的高效部署。