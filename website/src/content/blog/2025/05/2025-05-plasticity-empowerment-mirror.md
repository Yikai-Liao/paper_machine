---
title: "Plasticity as the Mirror of Empowerment"
pubDatetime: 2025-05-15T14:52:16+00:00
slug: "2025-05-plasticity-empowerment-mirror"
type: "arxiv"
id: "2505.10361"
score: 0.4284228681620238
author: "grok-3-latest"
authors: ["David Abel", "Michael Bowling", "André Barreto", "Will Dabney", "Shi Dong", "Steven Hansen", "Anna Harutyunyan", "Khimya Khetarpal", "Clare Lyle", "Razvan Pascanu", "Georgios Piliouras", "Doina Precup", "Jonathan Richens", "Mark Rowland", "Tom Schaul", "Satinder Singh"]
tags: ["Agent Design", "Plasticity", "Empowerment", "Information Theory", "Reinforcement Learning"]
institution: ["Google DeepMind", "Amii, University of Alberta", "Mila"]
description: "本文通过广义定向信息（GDI）定义了智能体的可塑性，揭示了其与赋能的镜像关系和张力，为智能体设计和能动性研究提供了新的理论基础。"
---

> **Summary:** 本文通过广义定向信息（GDI）定义了智能体的可塑性，揭示了其与赋能的镜像关系和张力，为智能体设计和能动性研究提供了新的理论基础。 

> **Keywords:** Agent Design, Plasticity, Empowerment, Information Theory, Reinforcement Learning

**Authors:** David Abel, Michael Bowling, André Barreto, Will Dabney, Shi Dong, Steven Hansen, Anna Harutyunyan, Khimya Khetarpal, Clare Lyle, Razvan Pascanu, Georgios Piliouras, Doina Precup, Jonathan Richens, Mark Rowland, Tom Schaul, Satinder Singh

**Institution(s):** Google DeepMind, Amii, University of Alberta, Mila


## Problem Background

智能体（Agent）的两个基本能力——被环境观察塑造的能力（可塑性，Plasticity）和影响未来环境观察的能力（赋能，Empowerment）——被认为是能动性（Agency）的核心。
尽管赋能在人工智能和认知科学中已被广泛研究，但可塑性作为一个对等概念尚未被充分定义和探索，尤其是在与赋能的关系及其对智能体设计的启示上。
论文旨在定义和量化可塑性，揭示其与赋能的联系，并探讨这种联系对智能体设计的意义。

## Method

*   **核心思想:** 提出可塑性作为智能体被环境塑造能力的量化指标，并通过信息论工具揭示其与赋能的镜像关系和张力。
*   **广义定向信息（GDI）:** 引入一个新的信息论量度——广义定向信息（Generalized Directed Information），扩展了传统定向信息，允许对任意时间区间的序列进行影响测量，而不限于相同长度或从时间起点开始的序列。
*   **可塑性定义:** 使用 GDI 量化环境观察对智能体动作的影响，即 P(λ, E) = max_{e∈E} I(O_{a:b} → A_{c:d})，其中 O 表示观察序列，A 表示动作序列，I 表示 GDI。
*   **赋能定义:** 同样基于 GDI，量化智能体动作对未来观察的影响，即 E(Λ, e) = max_{λ∈Λ} I(A_{a:b} → O_{c:d})。
*   **镜像关系:** 利用智能体与环境的对称性，证明智能体的可塑性等于环境的赋能，反之亦然（Proposition 4.6）。
*   **张力关系:** 通过信息守恒定律（Theorem 3.5），证明可塑性与赋能之间存在张力（Theorem 4.8），即智能体无法同时最大化两者，需在设计中权衡。
*   **理论基础:** 方法以最小的假设条件（不依赖特定模型或马尔可夫假设）构建，适用于广泛的智能体-环境交互场景。

## Experiment

*   **理论性质:** 本文是一篇理论性研究，未包含具体的实验数据或实证结果，而是通过数学证明和理论分析支持观点。
*   **分析手段:** 作者通过 GDI 的性质证明（Proposition 3.2-3.5）、可塑性与赋能的镜像关系（Proposition 4.6）以及张力关系（Theorem 4.8）等理论工具进行分析。
*   **直观示例:** 论文提供了扑克游戏和多房间环境等思想实验来帮助理解概念，但未基于数据验证。
*   **未来方向:** 由于是理论框架，实验效果的提升或设置的全面性无法直接评估，但为未来实验研究（如设计 GDI 估计器、验证张力对智能体性能的影响）奠定了基础。

## Further Thoughts

可塑性与赋能的镜像关系和张力为智能体设计提供了新视角，例如在强化学习中，探索（Exploration）可能与可塑性相关，而利用（Exploitation）可能与赋能相关，如何动态平衡两者可能成为优化智能体学习效率的关键。
此外，广义定向信息（GDI）作为一个通用工具，可能不仅限于智能体研究，还能在因果分析或通信系统中找到应用。
最后，作者提出的智能体定义（需同时具有非零可塑性和赋能）可能引发人工智能哲学和理论基础的新讨论，例如如何在多智能体系统中利用镜像关系设计合作机制。