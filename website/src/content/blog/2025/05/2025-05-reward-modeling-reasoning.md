---
title: "RM-R1: Reward Modeling as Reasoning"
pubDatetime: 2025-05-05T06:11:12+00:00
slug: "2025-05-reward-modeling-reasoning"
type: "arxiv"
id: "2505.02387"
score: 0.8303192989445728
author: "grok-3-latest"
authors: ["Xiusi Chen", "Gaotang Li", "Ziqi Wang", "Bowen Jin", "Cheng Qian", "Yu Wang", "Hongru Wang", "Yu Zhang", "Denghui Zhang", "Tong Zhang", "Hanghang Tong", "Heng Ji"]
tags: ["LLM", "Reward Modeling", "Reasoning", "Distillation", "RLHF"]
institution: ["University of Illinois Urbana-Champaign", "University of California, San Diego", "Texas A&M University", "Stevens Institute of Technology"]
description: "本文提出将奖励建模作为推理任务，通过推理导向的蒸馏和强化学习训练 RM-R1 模型家族，显著提升了奖励模型的性能和可解释性。"
---

> **Summary:** 本文提出将奖励建模作为推理任务，通过推理导向的蒸馏和强化学习训练 RM-R1 模型家族，显著提升了奖励模型的性能和可解释性。 

> **Keywords:** LLM, Reward Modeling, Reasoning, Distillation, RLHF

**Authors:** Xiusi Chen, Gaotang Li, Ziqi Wang, Bowen Jin, Cheng Qian, Yu Wang, Hongru Wang, Yu Zhang, Denghui Zhang, Tong Zhang, Hanghang Tong, Heng Ji

**Institution(s):** University of Illinois Urbana-Champaign, University of California, San Diego, Texas A&M University, Stevens Institute of Technology


## Problem Background

奖励模型（Reward Model, RM）在大型语言模型（LLM）对齐中至关重要，尤其是在通过人类反馈的强化学习（RLHF）中，用于提供可扩展的人类评价代理。
然而，现有奖励模型存在局限：标量奖励模型（Scalar RM）输出单一分数，缺乏透明性和解释性；生成式奖励模型（GenRM）虽能生成文本判断，但推理过程往往浅显，难以应对复杂的偏好任务。
论文的出发点是探索是否能将奖励建模转化为一个推理任务，通过引入深层次推理能力，提升模型在通用领域（generalist reward modeling）中的性能和可解释性。

## Method

*   **核心理念**：提出推理奖励模型（Reasoning Reward Models, REAS RMS），将奖励建模任务定义为推理任务，要求模型在评分或判断前生成详细的推理链或评价标准，以增强透明度和准确性。
*   **训练流程**：采用两阶段训练策略：
    1. **推理链蒸馏（Distillation of Reasoning Traces）**：利用强大的‘oracle’模型（如 Claude-3.7-Sonnet 和 OpenAI-O3）合成高质量推理链数据，对初始指令调整模型（如 Qwen-2.5-Instruct）进行监督微调，以奠定推理能力基础。
    2. **强化学习（Reinforcement Learning with Verifiable Rewards, RLVR）**：通过基于正确性的奖励函数进一步优化模型，使用 Group Relative Policy Optimization (GRPO) 算法，结合 KL 正则化防止过度偏离参考模型。
*   **具体策略**：
    - 引入‘Chain-of-Rubrics’（CoR）提示框架，模型根据任务类型（聊天 Chat 或推理 Reasoning）采取不同推理策略：聊天任务生成评价标准（rubrics）并基于此评估；推理任务先自行解决问题，再对比候选答案。
    - 蒸馏阶段使用小规模高质量数据（约 9K 样本）进行预训练，强化学习阶段使用更大规模偏好数据（约 64K 样本）优化。
*   **技术细节**：训练中采用 Fully Sharded Data Parallel (FSDP) 和 vLLM 优化内存和推理效率，采样参数（如 temperature=1.0）保持默认设置，确保生成多样性。
*   **创新点**：通过推理导向的训练，不仅输出最终判断，还生成逻辑清晰的推理过程，显著提升模型的可解释性和跨领域适用性。

## Experiment

*   **性能表现**：RM-R1 模型家族在多个基准数据集（RewardBench, RM-Bench, RMB）上表现出色，RM-R1-Qwen-Instruct-32B 在 RewardBench 上整体准确率达 92.9%，超越了更大规模的模型如 Llama3.1-405B 和 GPT-4o（最高提升 13.8%）；在 RM-Bench 上，RM-R1-DeepSeek-Distilled-Qwen-32B 在数学和代码领域分别达到 91.8% 和 74.1% 的准确率，显著优于先前最佳模型。
*   **实验设置**：实验覆盖了不同模型规模（7B 到 32B），对比了标量奖励模型、生成式奖励模型及其他推理增强模型，数据集包括 Skywork Reward Preference 80K 等，并通过清洗数据避免偏差；此外，分析了模型规模和推理计算预算（inference-time compute）对性能的影响，设置全面且合理。
*   **数据效率**：RM-R1 在较小训练数据量（蒸馏阶段仅 8.7K 样本）下仍取得竞争性结果，显示出较高的数据效率。
*   **局限性与成本**：训练涉及蒸馏和强化学习两阶段，计算成本较高，需多节点 GPU 集群支持；推理时生成长推理链增加了额外计算开销，但性能提升显著。

## Further Thoughts

论文将奖励建模转化为推理任务的思路非常具有启发性，提示我们可以在模型对齐中更注重过程监督，而不仅是结果导向；任务分类与定制化推理策略的结合，启发我们在多任务场景中根据任务特性动态调整模型行为；此外，蒸馏与强化学习结合的训练范式表明，先从高质量数据中学习知识，再通过探索优化，可能是一种通用的模型能力提升策略，尤其对中小规模模型有借鉴意义；最后，推理计算预算对性能的影响启发我们可以在实际应用中动态调整推理资源，以在成本和效果间找到平衡。