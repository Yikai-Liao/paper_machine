---
title: "VISOR++: Universal Visual Inputs based Steering for Large Vision Language Models"
pubDatetime: 2025-09-29T21:43:18+00:00
slug: "2025-09-visor-visual-steering"
type: "arxiv"
id: "2509.25533"
score: 0.5248883192234157
author: "grok-3-latest"
authors: ["Ravikumar Balakrishnan", "Mansi Phute"]
tags: ["VLM", "Behavioral Steering", "Visual Input", "Transferability", "Adversarial Optimization"]
institution: ["HiddenLayer Inc.", "Georgia Institute of Technology"]
description: "VISOR++ 通过优化视觉输入实现对视觉语言模型的行为转向，无需访问模型内部参数，展现了跨模型通用性和实际部署潜力。"
---

> **Summary:** VISOR++ 通过优化视觉输入实现对视觉语言模型的行为转向，无需访问模型内部参数，展现了跨模型通用性和实际部署潜力。 

> **Keywords:** VLM, Behavioral Steering, Visual Input, Transferability, Adversarial Optimization

**Authors:** Ravikumar Balakrishnan, Mansi Phute

**Institution(s):** HiddenLayer Inc., Georgia Institute of Technology


## Problem Background

视觉语言模型（VLMs）在安全关键领域（如医疗、自动驾驶）的广泛应用使得行为对齐和抗对抗性操纵变得至关重要。
传统行为控制方法存在局限：系统提示容易被用户指令覆盖，而基于激活的转向向量需要侵入式访问模型内部参数，无法适用于 API 服务或闭源模型。
此外，寻找跨多个 VLMs 通用的转向方法仍是一个开放性研究问题，亟需一种无需模型内部访问且具有通用性的解决方案。

## Method

*   **核心思想:** 通过优化视觉输入（图像）来模拟传统转向向量在激活空间中的行为控制效果，将转向机制从模型内部操作转移到输入空间，避免对模型内部的侵入性访问。
*   **问题形式化:** 针对一组 VLMs，优化一个通用图像，使其在不同模型、不同文本提示和不同层级上诱导目标激活模式，确保转向效果对模型架构、提示变化和层级深度的鲁棒性。
*   **技术实现:**
    *   **可微预处理管道:** 重新实现图像预处理操作（如调整大小、归一化）为可微的张量操作，确保从模型输出到输入图像的梯度流完整性，支持端到端优化。
    *   **优化策略:** 对于单模型转向图像优化，采用 PGD（Projected Gradient Descent）方法，通过期望变换（EoT）提升鲁棒性；对于跨模型通用图像优化，采用 CWA-SSA（Common Weakness Approach with Spectral Simulation Attack）框架，利用双重动量更新和频谱增强技术，确保优化收敛性和跨模型迁移性。
    *   **行为维度:** 针对三种关键行为维度进行转向控制，包括拒绝（refusal，抑制或增强对有害请求的拒绝）、奉承（sycophancy，抑制或增强对用户的盲目同意）和生存本能（survival instinct，抑制或增强自我保护行为）。
*   **关键优势:** 方法不依赖模型内部参数访问，适用于闭源模型和 API 部署场景，同时通过通用图像优化追求跨模型的转向一致性。

## Experiment

*   **有效性:** VISOR++ 在行为控制上与传统转向向量表现相当甚至更优，例如在 IDEFICS2 模型上，拒绝行为的动态范围（0.231-0.94）优于转向向量（0.3-0.817），展现出更强的行为调整能力；相比系统提示方法，VISOR++ 的行为修改效果提升 2-3 倍，尤其在抑制行为（如负向转向）方面表现突出。
*   **通用性:** 单个通用 VISOR++ 图像在多个模型（如 LLaVA-1.5-7B 和 IDEFICS2-8B）上实现了接近单模型优化的转向效果，尤其在拒绝和生存本能任务上动态范围相当，奉承任务的负向转向效果稍逊但仍优于系统提示。
*   **迁移性:** 对未见模型（包括开源模型如 LLaVA-NeXT 和闭源模型如 GPT-4 系列）的负向转向表现出方向一致性，尽管效果幅度较小（如拒绝率降低 0.007-0.048），表明通用图像具有一定跨模型迁移潜力。
*   **无关任务影响:** 在 MMLU 数据集（14,000 个样本）上，VISOR++ 图像对无关任务性能的影响微乎其微（保持 99.9% 性能），显示出方法对行为转向的特异性。
*   **实验设置合理性:** 实验涵盖了架构不同的模型、多种行为维度和迁移性测试，设置较为全面；但对闭源模型的评估基于行为观察比例而非概率计算，可能存在偏差；奉承任务优化收敛较慢，提示优化效率有待提升。

## Further Thoughts

VISOR++ 揭示了输入空间优化可以有效模拟激活空间操作的潜力，这启发我们可以在其他模态（如音频、视频）或多模态组合中探索类似转向方法；此外，通用图像对未见模型的迁移性表明模型间可能存在共享表征或‘共同弱点’，未来可研究更大规模模型集合的通用转向优化策略；另一个方向是探索动态调整输入图像以适应不同上下文或用户输入的可能性，进一步提升转向的灵活性和鲁棒性。