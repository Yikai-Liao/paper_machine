---
title: "Seek in the Dark: Reasoning via Test-Time Instance-Level Policy Gradient in Latent Space"
pubDatetime: 2025-05-19T16:26:02+00:00
slug: "2025-05-latent-seek-reasoning"
type: "arxiv"
id: "2505.13308"
score: 0.7881037867902163
author: "grok-3-latest"
authors: ["Hengli Li", "Chenxi Li", "Tong Wu", "Xuekai Zhu", "Yuxuan Wang", "Zhaoxin Yu", "Eric Hanchen Jiang", "Song-Chun Zhu", "Zixia Jia", "Ying Nian Wu", "Zilong Zheng"]
tags: ["LLM", "Latent Space", "Test-Time Scaling", "Reasoning", "Policy Gradient"]
institution: ["Institute for Artificial Intelligence, Peking University", "NLCo Lab, Beijing Institute for General Artificial Intelligence", "Department of Automation, Tsinghua University", "Shanghai Jiao Tong University", "Institute of Automation, Chinese Academy of Sciences", "University of California, Los Angeles"]
description: "本文提出 LATENT SEEK 框架，通过在潜在空间中进行测试时实例级适应，利用策略梯度优化潜在表示，显著提升大型语言模型的推理能力，同时保持计算效率和模型参数不变。"
---

> **Summary:** 本文提出 LATENT SEEK 框架，通过在潜在空间中进行测试时实例级适应，利用策略梯度优化潜在表示，显著提升大型语言模型的推理能力，同时保持计算效率和模型参数不变。 

> **Keywords:** LLM, Latent Space, Test-Time Scaling, Reasoning, Policy Gradient

**Authors:** Hengli Li, Chenxi Li, Tong Wu, Xuekai Zhu, Yuxuan Wang, Zhaoxin Yu, Eric Hanchen Jiang, Song-Chun Zhu, Zixia Jia, Ying Nian Wu, Zilong Zheng

**Institution(s):** Institute for Artificial Intelligence, Peking University, NLCo Lab, Beijing Institute for General Artificial Intelligence, Department of Automation, Tsinghua University, Shanghai Jiao Tong University, Institute of Automation, Chinese Academy of Sciences, University of California, Los Angeles


## Problem Background

大型语言模型（LLMs）在推理能力上仍面临挑战，尤其是在需要结构化思维和逐步分析的任务中，传统训练方法（如监督微调和强化学习）因高计算成本和灾难性遗忘问题受限，而训练数据的稀缺性进一步阻碍性能提升；论文提出通过测试时扩展（test-time scaling），即在不更新参数的情况下增加测试时计算量，特别是在潜在空间（latent space）中优化推理过程，以克服这些问题。

## Method

*   **核心思想:** 提出 LATENT SEEK 框架，通过测试时实例级适应（Test-Time Instance-Level Adaptation, TTIA），在模型的潜在空间中动态优化潜在表示（latent representations），以提升推理能力，而不修改预训练模型参数。
*   **潜在空间定义:** 将 Transformer 模型在最终语言模型头（LM head）之前的输出空间视为潜在空间，潜在表示为对应令牌的隐藏状态向量。
*   **优化过程:** 针对每个推理问题实例，初始化潜在表示（可通过 Chain-of-Thought 推理序列作为起点），然后采用策略梯度方法（如 REINFORCE 算法）迭代更新潜在表示，以最大化奖励信号；更新过程中假设潜在表示相互独立，逐个令牌地进行采样和优化。
*   **奖励机制:** 主要使用自奖励机制（self-rewarding），即依靠模型内部能力生成奖励信号，无需外部信息；同时测试了完美稀疏奖励模型（Perfect Sparse Reward Model, PSRM）以探索优化上限。
*   **增强技术:** 包括 CoT 初始化（利用 CoT 序列作为优化起点）和部分序列优化（仅优化潜在表示序列的一部分，通过超参数 ρ 控制优化范围，减少计算成本并提升稳定性）。
*   **关键优势:** 方法轻量高效，仅在测试时操作，避免了训练时的计算开销和参数更新带来的风险，同时充分利用潜在空间中的语义信息引导推理路径。

## Experiment

*   **有效性:** LATENT SEEK 在多个推理基准数据集（GSM8K, MATH-500, AIME2024）上显著优于基线方法，如在 GSM8K 上平均提升 10.75%，MATH-500 上提升 3.93%，AIME2024 上提升 4.73%；以 LLaMA3.1-8B-Instruct 为骨干模型时，相比 Genius 和 SimpleRL-Zoo 等方法提升达 5.4%-20.0%。
*   **优越性:** 相比训练无关的 Best-of-N 方法，在 GSM8K 和 MATH-500 上分别提升 7.7% 和 3.4%；相比需要参数更新的监督微调和强化学习方法，展现出更高效率；使用完美稀疏奖励模型（PSRM）时，性能进一步提升，平均比 CoT 高出 19.12%，凸显潜在空间优化潜力。
*   **测试时扩展性:** 增加迭代次数可线性提升性能，尤其在理想奖励模型下效果显著，验证了潜在空间中测试时扩展的有效性。
*   **效率:** 算法通常在 2 次迭代内收敛（GSM8K 和 MATH-500 平均迭代次数为 0.86 和 1.23），计算开销低，答案长度与 CoT 相当，避免冗长输出。
*   **实验设置合理性:** 实验覆盖多种模型家族（Qwen2, LLaMA3.1, Mistral）和规模（1.5B 到 14B 参数），测试不同提示类型，并在多个难度级别的数据集上验证泛化性；基线选择全面，包括 CoT, BoN, SFT 和 RL 方法，对比充分。

## Further Thoughts

潜在空间作为推理优化的新领域展示了巨大潜力，启发我们探索更复杂的潜在表示操作（如聚类或投影）以挖掘隐含知识；测试时扩展在潜在空间中的高效性提示未来可研究动态迭代策略或智能奖励设计；小模型性能的显著提升表明可开发轻量级方法激活其潜力，减少对大模型依赖；奖励函数质量对优化效果的影响启发我们探索多代理协作或外部验证器的奖励机制。