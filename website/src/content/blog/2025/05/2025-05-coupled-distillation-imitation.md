---
title: "Coupled Distributional Random Expert Distillation for World Model Online Imitation Learning"
pubDatetime: 2025-05-04T19:32:48+00:00
slug: "2025-05-coupled-distillation-imitation"
type: "arxiv"
id: "2505.02228"
score: 0.4800988277796432
author: "grok-3-latest"
authors: ["Shangzhe Li", "Zhiao Huang", "Hao Su"]
tags: ["Imitation Learning", "World Model", "Density Estimation", "Reward Modeling", "Latent Space"]
institution: ["UNC Chapel Hill", "Hillbot", "University of California, San Diego"]
description: "本文提出基于密度估计的耦合分布随机专家蒸馏（CDRED）奖励模型，显著提升了世界模型在线模仿学习的稳定性和性能，成功应对了对抗性训练带来的挑战。"
---

> **Summary:** 本文提出基于密度估计的耦合分布随机专家蒸馏（CDRED）奖励模型，显著提升了世界模型在线模仿学习的稳定性和性能，成功应对了对抗性训练带来的挑战。 

> **Keywords:** Imitation Learning, World Model, Density Estimation, Reward Modeling, Latent Space

**Authors:** Shangzhe Li, Zhiao Huang, Hao Su

**Institution(s):** UNC Chapel Hill, Hillbot, University of California, San Diego


## Problem Background

模仿学习（Imitation Learning, IL）在机器人、自动驾驶和医疗等领域通过专家演示学习复杂行为取得了显著成功，但现有方法在世界模型框架中依赖对抗性奖励或价值函数时，常常面临训练不稳定的问题，尤其是在高维任务或长期在线训练中，导致策略无法达到专家水平或性能下降。本文旨在通过基于密度估计的奖励模型替代对抗性方法，解决训练不稳定问题，同时保持专家级性能。

## Method

* **核心思想**：提出耦合分布随机专家蒸馏（Coupled Distributional Random Expert Distillation, CDRED），通过随机网络蒸馏（RND）在世界模型的潜在空间中进行密度估计，构建奖励模型，替代对抗性训练以提高稳定性。
* **奖励模型设计**：使用两个预测器网络（专家预测器和行为预测器）共享一组随机目标网络，在潜在空间中同时估计专家分布和行为分布；奖励通过两者的差异计算，平衡探索和利用。
* **耦合分布估计**：联合估计专家和行为分布，解决初始策略分布与专家分布差异过大导致的学习困难问题，促进早期探索。
* **一致性校正**：引入偏差校正项和状态-动作对出现频率的估计，确保在线训练中奖励估计的一致性，避免奖励分布与实际分布不匹配。
* **世界模型集成**：将 CDRED 奖励模型嵌入无解码器世界模型（如 TD-MPC 系列），通过潜在空间的动态模型进行状态转换，并采用模型预测路径积分（MPPI）方法进行策略规划和决策。
* **关键优势**：避免对抗性训练的 min-max 优化问题，理论上更稳定；在潜在空间操作更适合高维任务，且与世界模型的动态感知特性结合紧密。

## Experiment

* **有效性**：在 DMControl、Meta-World 和 ManiSkill2 等多个基准测试中，CDRED 在运动控制和机器人操作任务中达到了专家级性能，尤其是在 Meta-World 和 ManiSkill2 的操作任务中，成功率显著高于基线方法（如 IQ-MPC、IQL+SAC）。
* **稳定性**：相比基于对抗性训练的方法（如 IQ-MPC），CDRED 的梯度范数明显更小，表明训练过程更稳定，避免了过于强大的判别器和长期不稳定问题。
* **全面性**：实验覆盖了低维和高维任务、状态和视觉观察，展示了方法的广泛适用性；消融研究表明 CDRED 在少量专家演示（仅 5 条轨迹）下仍能有效学习。
* **局限性**：在某些视觉任务（如 Walker Run）与 IQ-MPC 性能相当，未完全超越，可能与视觉输入的复杂性有关。

## Further Thoughts

论文中在潜在空间构建奖励模型的思路启发了我，未来可以探索不同潜在空间表示对奖励估计的影响，例如是否可以通过自监督学习进一步优化潜在表示的动态感知能力；此外，耦合分布估计的概念可能适用于多专家或多任务学习场景，通过联合估计多个分布来提升泛化能力；最后，RND 替代对抗性训练的思想或许可以推广到其他需要稳定训练的领域，如生成模型或策略优化，减少优化过程中的不稳定性。