---
title: "Accelerating RL for LLM Reasoning with Optimal Advantage Regression"
pubDatetime: 2025-05-27T03:58:50+00:00
slug: "2025-05-antidistillation-sampling"
type: "arxiv"
id: "2505.20686"
score: 0.6836483324098579
author: "grok-3-latest"
authors: ["Yash Savani", "Asher Trockman", "Zhili Feng", "Avi Schwarzschild", "Alexander Robey", "Marc Finzi", "J. Zico Kolter"]
tags: ["LLM", "Proxy Model", "Distillation", "Sampling", "Reasoning"]
institution: ["Carnegie Mellon University", "Google", "Peking University"]
description: "本文提出了一种反蒸馏采样方法，通过代理模型辅助在推理时调整token概率分布，毒化大型语言模型的推理轨迹以干扰模型蒸馏，同时维持原始性能。"
---

> **Summary:** 本文提出了一种反蒸馏采样方法，通过代理模型辅助在推理时调整token概率分布，毒化大型语言模型的推理轨迹以干扰模型蒸馏，同时维持原始性能。 

> **Keywords:** LLM, Proxy Model, Distillation, Sampling, Reasoning

**Authors:** Yash Savani, Asher Trockman, Zhili Feng, Avi Schwarzschild, Alexander Robey, Marc Finzi, J. Zico Kolter

**Institution(s):** Carnegie Mellon University, Google, Peking University


## Problem Background

大型语言模型（LLM）在生成详细推理过程（Reasoning Traces）时，虽然展现了强大的能力，但也暴露了一个潜在风险：竞争对手可以通过模型蒸馏（Model Distillation）技术，利用公开的推理数据廉价地训练出性能相近的模型，从而导致知识产权泄露和安全隐患（如绕过模型的安全限制）。
本文旨在解决这一关键问题：如何在不牺牲教师模型性能的前提下，保护其推理过程免受蒸馏攻击。

## Method

*   **核心思想**：提出一种反蒸馏采样策略，通过在推理时动态调整token的概率分布，‘毒化’生成的推理轨迹，从而干扰学生模型的蒸馏学习过程，同时尽可能保留教师模型的原始性能。
*   **具体实现**：
    1. 在生成每个token时，不直接依据教师模型的概率分布采样，而是引入一个反蒸馏调整项。
    2. 该调整项通过一个轻量级的代理模型（Proxy Model）计算，代理模型结合下游任务的损失梯度，估计哪些token的选择会对蒸馏过程产生负面影响（即降低学生模型的学习效果）。
    3. 基于调整后的概率分布进行采样，确保生成的推理轨迹对蒸馏‘有害’，但对教师模型的任务完成影响最小。
*   **技术细节**：
    - 代理模型通常是一个较小的模型，用于快速估计token对蒸馏的影响，降低计算开销。
    - 反蒸馏调整项的强度可以通过超参数控制，以平衡抗蒸馏效果和教师模型性能。
    - 该方法无需修改教师模型的权重，仅在推理阶段介入，是一种非侵入式的保护机制。
*   **创新点**：相比传统的防御方法（如简单增加采样随机性），这种方法通过有针对性的概率调整，实现了更高效的抗蒸馏效果。

## Experiment

*   **有效性验证**：实验在多个基准数据集（如GSM8K、MATH）上进行，结果表明反蒸馏采样在保持教师模型准确率基本不变的情况下，显著降低了学生模型通过蒸馏获得的性能（准确率下降幅度可达30%-50%，具体取决于数据集和模型规模）。
*   **对比分析**：与简单的采样温度调整（会导致教师模型性能明显下降）相比，反蒸馏采样在性能-抗蒸馏能力的权衡上表现更优。
*   **实验设置**：实验设计较为全面，涵盖了不同规模的教师模型和学生模型，以及多种任务类型（数学推理、逻辑推理等），同时测试了不同强度反蒸馏调整项的影响。
*   **计算开销**：主要额外开销来自每次token生成时代理模型的前向计算（两次前向传播），但由于代理模型规模较小，整体开销在可接受范围内。
*   **局限性**：实验未充分探讨极端对抗场景（如学生模型尝试通过数据清洗绕过毒化），未来可进一步验证方法的鲁棒性。

## Further Thoughts

反蒸馏采样的思想启发了我对模型安全保护的新思考：是否可以设计更复杂的代理模型（如结合多任务学习或对抗训练）来进一步提升抗蒸馏效果？此外，这种方法是否能推广到其他生成式模型（如图像生成模型）的输出保护中？另一个有趣的方向是探索推理轨迹毒化与学生模型反制之间的博弈，是否会催生新的对抗性攻防技术？