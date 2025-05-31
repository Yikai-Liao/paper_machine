---
title: "Unraveling LoRA Interference: Orthogonal Subspaces for Robust Model Merging"
pubDatetime: 2025-05-28T23:28:12+00:00
slug: "2025-05-lora-merging-orthogonal"
type: "arxiv"
id: "2505.22934"
score: 0.8160227700962749
author: "grok-3-latest"
authors: ["Haobo Zhang", "Jiayu Zhou"]
tags: ["LLM", "Model Merging", "LoRA", "Parameter Interference", "Orthogonal Subspace"]
institution: ["University of Michigan"]
description: "本文提出OSRM方法，通过在微调前约束LoRA子空间与无关任务数据分布正交，显著提升了模型合并的多任务性能，同时保持单任务准确性。"
---

> **Summary:** 本文提出OSRM方法，通过在微调前约束LoRA子空间与无关任务数据分布正交，显著提升了模型合并的多任务性能，同时保持单任务准确性。 

> **Keywords:** LLM, Model Merging, LoRA, Parameter Interference, Orthogonal Subspace

**Authors:** Haobo Zhang, Jiayu Zhou

**Institution(s):** University of Michigan


## Problem Background

大型语言模型（LLMs）针对每个任务单独微调会导致存储和部署成本高昂，而多任务学习需要同时访问所有任务数据，计算开销大且扩展性受限；模型合并作为一种无需额外训练的解决方案受到关注，但现有方法在处理低秩适应（LoRA）微调模型时性能显著下降，原因是任务间参数与数据分布的干扰导致输出偏移。

## Method

*   **核心思想:** 提出Orthogonal Subspaces for Robust Model Merging (OSRM)方法，在微调前约束LoRA子空间，使其与无关任务的数据分布正交，从而减少合并时的参数干扰。
*   **具体实现:** 
    *   **子空间初始化:** 利用任务数据的隐特征（latent features）计算样本协方差矩阵，通过特征分解找到最小特征值对应的特征向量，初始化LoRA矩阵A，使其与无关任务的隐特征正交，减少干扰项的输出偏移。
    *   **放松约束:** 理论上可固定A矩阵，但为避免单任务性能下降，允许A在微调中更新，初始化仍保持正交性以最小化初始干扰。
    *   **多任务扩展:** 对于多个任务，针对每个任务t，收集其他任务的隐特征构建正交子空间，并通过平均样本特征减少内存和隐私问题。
    *   **无缝集成:** OSRM作为前处理步骤，可与现有合并方法（如Task Arithmetic, Fisher Merging等）结合，提升合并效果。
*   **创新点:** 从数据-参数交互视角出发，强调数据分布对合并效果的影响，而非单纯依赖数据无关的权重正交化。

## Experiment

*   **有效性:** OSRM在八个GLUE数据集上，结合多种合并方法（如Task Arithmetic, RegMean等），显著提升了多任务性能，例如在RoBERTa-large上结合TA方法时，平均准确率从70.04%提升到76.59%，在T5-large上结合RegMean时提升了3.76%。
*   **单任务性能:** 对单任务准确性影响极小，平均性能差距小于1%，部分数据集甚至超越基线。
*   **鲁棒性:** 对超参数（如缩放系数λ、样本数k、任务数N）变化表现出较强鲁棒性，尤其在任务数较多时优势更明显。
*   **实验设置:** 覆盖不同架构（RoBERTa-large, T5-large, Llama3.2-1B）和规模（Llama3.2-3B, Llama3-8B）的模型，数据集选择具有代表性（GLUE基准），设置全面合理，但未涉及跨架构合并或全微调模型场景。
*   **局限性:** 随着模型规模增大，OSRM的相对提升幅度减小，可能是大模型自身知识量增加导致合并性能自然提高。

## Further Thoughts

OSRM强调数据-参数交互的视角启发了我，可以将这种思想扩展到联邦学习中不同客户端模型的合并，或通过少量隐特征在隐私敏感场景下实现高效合并；此外，是否可以在微调过程中动态调整正交性以适应数据变化，或者通过映射到共享特征空间探索跨架构模型合并的可能性？