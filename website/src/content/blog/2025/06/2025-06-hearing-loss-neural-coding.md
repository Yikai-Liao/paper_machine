---
title: "Modelling the Effects of Hearing Loss on Neural Coding in the Auditory Midbrain with Variational Conditioning"
pubDatetime: 2025-06-03T17:12:21+00:00
slug: "2025-06-hearing-loss-neural-coding"
type: "arxiv"
id: "2506.03088"
score: 0.5045827931405918
author: "grok-3-latest"
authors: ["Lloyd Pellatt", "Shievanie Sabesan", "Fotios Drakopoulos", "Nicholas A. Lesica"]
tags: ["Neural Coding", "Hearing Loss", "Auditory Midbrain", "Variational Model", "Personalization"]
institution: ["University College London, Ear Institute"]
description: "本文提出变分条件模型 ψ-ICNet，通过少量参数编码听力损失，直接从听觉中脑神经活动数据学习个体差异，成功模拟不同听力状态下的神经编码，并通过贝叶斯优化快速适配新个体。"
---

> **Summary:** 本文提出变分条件模型 ψ-ICNet，通过少量参数编码听力损失，直接从听觉中脑神经活动数据学习个体差异，成功模拟不同听力状态下的神经编码，并通过贝叶斯优化快速适配新个体。 

> **Keywords:** Neural Coding, Hearing Loss, Auditory Midbrain, Variational Model, Personalization

**Authors:** Lloyd Pellatt, Shievanie Sabesan, Fotios Drakopoulos, Nicholas A. Lesica

**Institution(s):** University College London, Ear Institute


## Problem Background

听力损失对听觉系统的神经编码有复杂影响，现有模型主要聚焦于耳蜗和听觉神经，无法捕捉中央听觉通路的变化，且难以泛化到不同听力状态的个体；本文旨在开发一个参数化模型，通过少量参数编码听力损失的多样性，模拟听觉中脑的神经活动，为个性化听力补偿提供基础。

## Method

* **核心思想**：提出一种变分条件模型（ψ-ICNet），通过少量条件参数 ψ（每动物仅 6 个）编码听力损失的低维表示，直接从听觉中脑神经活动数据中学习个体差异，而非依赖外部测量（如听力阈值）。
* **模型架构**：基于 ICNet 框架，包含共享的声音卷积编码器（提取声音潜在特征）、ψ 编码器（处理条件参数 ψ，生成个性化特征）、瓶颈融合网络（结合共享和个性化特征）和共享解码器（输出多单位神经活动，MUA）。
* **时间退化处理**：引入时间转移函数，通过可训练参数调整 ψ 的均值，模拟实验过程中神经响应的退化，捕捉类似听力损失加重的动态变化。
* **训练策略**：采用复合损失函数，包括交叉熵损失（衡量预测与真实 MUA 的差异）、KL 散度（正则化 ψ 分布，确保不同动物分布有重叠）和 ABR 惩罚项（在前 10 个 epoch 鼓励相似听力损失动物在 ψ 空间聚集）。
* **对比实验**：实现 ABRNet（基于听觉脑干响应阈值的条件模型）作为基准，验证直接学习 ψ 参数的优越性；同时测试注意力机制（cross-attention）以增强特征融合，但效果有限。
* **新个体适配**：通过贝叶斯优化快速搜索 ψ 空间，为未见过的动物找到接近最优的条件参数，实现泛化。

## Experiment

* **有效性**：ψ-ICNet（6 参数版本）在听力受损动物上的 FEVE（解释方差分数）达到 68%，接近单动物模型（70.7%），在正常听力动物上为 62%（单动物模型为 66.9%）；KL 散度显示其在分布相似性上甚至优于单动物模型。
* **泛化性**：通过贝叶斯优化，模型能在 15-30 次迭代内为未见过动物找到接近最优 ψ 参数，FEVE 在未见过动物上达到 26.9%（训练 20 只动物时），虽低于训练集内动物（50.7%），但显示出一定泛化能力。
* **对比分析**：ABRNet 性能显著低于 ψ-ICNet，验证了直接从数据学习条件参数比依赖听力阈值更有效；注意力机制未显著提升性能。
* **合理性与局限**：实验覆盖多种声音类型（语音、音乐、噪声中语音等）和听力状态（正常与受损），数据预处理（如通道对齐）减少非听力相关差异；增加训练动物数量（9 只到 20 只）提升未见过动物性能，但仍低于训练集内表现，表明模型对新个体的泛化能力有待进一步提升。

## Further Thoughts

ψ 参数学习了一个平滑的听力损失空间，这种参数化个体差异的方法可推广至其他个性化建模任务；直接从数据学习条件参数避免了外部测量的偏差，对处理复杂或隐藏性问题（如隐藏性听力损失）具有启发；贝叶斯优化快速适配新个体的能力为实时个性化应用（如助听器调整）提供了可能；此外，中脑层面的神经编码建模比耳蜗模型更接近整体听觉感知，未来可用于开发直接修复神经编码失真的高级听力补偿策略。