---
title: "Blending Supervised and Reinforcement Fine-Tuning with Prefix Sampling"
pubDatetime: 2025-07-02T13:04:09+00:00
slug: "2025-07-prefix-rft-hybrid-training"
type: "arxiv"
id: "2507.01679"
score: 0.7630678825649105
author: "grok-3-latest"
authors: ["Zeyu Huang", "Tianhao Cheng", "Zihan Qiu", "Zili Wang", "Yinghui Xu", "Edoardo M. Ponti", "Ivan Titov"]
tags: ["LLM", "Supervised Fine-Tuning", "Reinforcement Fine-Tuning", "Hybrid Training", "Reasoning"]
institution: ["University of Edinburgh", "Fudan University", "Alibaba Group", "Stepfun", "University of Amsterdam"]
description: "本文提出 Prefix-RFT，通过从示范数据中采样前缀引导在线探索，统一监督微调与强化微调范式，显著提升大型语言模型在数学推理任务上的性能。"
---

> **Summary:** 本文提出 Prefix-RFT，通过从示范数据中采样前缀引导在线探索，统一监督微调与强化微调范式，显著提升大型语言模型在数学推理任务上的性能。 

> **Keywords:** LLM, Supervised Fine-Tuning, Reinforcement Fine-Tuning, Hybrid Training, Reasoning

**Authors:** Zeyu Huang, Tianhao Cheng, Zihan Qiu, Zili Wang, Yinghui Xu, Edoardo M. Ponti, Ivan Titov

**Institution(s):** University of Edinburgh, Fudan University, Alibaba Group, Stepfun, University of Amsterdam


## Problem Background

大型语言模型（LLMs）的后训练主要通过监督微调（SFT）和强化微调（RFT）两种范式进行，但 SFT 因行为克隆问题导致泛化能力受限，而 RFT 虽能提升性能却依赖初始策略且易产生意外行为。
论文旨在探索如何整合 SFT 的知识注入与 RFT 的目标导向优化，克服两者局限性，提升模型在复杂任务如数学推理上的表现。

## Method

*   **核心思想:** 提出 Prefix-RFT，一种混合后训练方法，通过从离线示范数据中采样前缀（prefix）作为引导，结合在线策略生成后续内容，形成混合轨迹用于强化微调（RFT），以平衡示范学习和自主探索。
*   **具体实现:**
    *   **前缀采样:** 从示范数据中截取部分序列作为前缀，长度由余弦衰减调度器（cosine decay scheduler）控制，初期依赖较多示范数据，后期逐渐减少，引入课程学习效果并缓解位置偏差。
    *   **混合轨迹生成:** 使用当前策略（policy）生成前缀的后续内容，形成示范前缀与在线生成的混合序列，与纯在线轨迹一起参与 RFT 更新。
    *   **基于熵的剪切策略:** 针对前缀中的高熵 token（即模型不确定性较高的 token）进行梯度更新，限制离线数据对训练的过度影响，避免训练退化为单纯的 SFT。
    *   **统一优化框架:** 将 SFT 和 RFT 的梯度更新统一到一个框架中，通过动态权重（如 PPO 中的 advantage 和 clipping 机制）平衡两种学习范式。
*   **优势:** 方法对现有 RFT 框架（如 PPO）修改最小，易于集成，同时兼顾 SFT 的稳定知识获取和 RFT 的探索能力。

## Experiment

*   **有效性:** Prefix-RFT 在数学推理任务（如 AIME, AMC, MATH-500）上显著优于单独的 SFT（平均得分 43.7%）和 RFT（46.8%），达到 50.8%；在通用推理任务（如 ARC-c, GPQA, MMLU-Pro）上得分 58.7%，也超越基线和并行工作（如 LUFFY）。
*   **鲁棒性:** 实验在不同模型规模（Qwen2.5-Math-7B, 1.5B）和架构（LLaMA-3.1-8B）上验证了方法的普适性；消融研究表明即使示范数据量减少至 1% 或质量较低，性能下降有限（从 40.8% 降至 37.6%）。
*   **动态调整:** 分析实验显示 Prefix-RFT 对困难问题更依赖示范数据（SFT 损失下降更多），对简单问题则倾向于自主探索，展现了自适应性。
*   **实验设置合理性:** 实验涵盖多种基准、模型和数据场景，设置全面；但在最难任务（如 AIME）上，高质量大规模示范数据仍不可替代，提示方法在极端场景下的优化空间。

## Further Thoughts

Prefix-RFT 的动态过渡机制（通过优势值调整示范与探索的权重）启发了我思考是否可以在其他任务中引入基于不确定性或任务难度的自适应训练策略；基于熵的剪切策略提示利用模型不确定性作为学习信号的潜力；余弦衰减调度器引入课程学习的思想，是否可以通过更复杂的动态调度进一步优化训练效果？这些 idea 不仅适用于 LLM 后训练，也可能推广至其他混合学习场景。