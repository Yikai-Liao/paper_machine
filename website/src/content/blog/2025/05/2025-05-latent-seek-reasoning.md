---
title: "Seek in the Dark: Reasoning via Test-Time Instance-Level Policy Gradient in Latent Space"
pubDatetime: 2025-05-19T16:26:02+00:00
slug: "2025-05-latent-seek-reasoning"
type: "arxiv"
id: "2505.13308"
score: 0.7881037867902163
author: "grok-3-latest"
authors: ["Hengli Li", "Chenxi Li", "Tong Wu", "Xuekai Zhu", "Yuxuan Wang", "Zhaoxin Yu", "Eric Hanchen Jiang", "Song-Chun Zhu", "Zixia Jia", "Ying Nian Wu", "Zilong Zheng"]
tags: ["LLM", "Latent Space", "Test Time Scaling", "Reasoning", "Policy Gradient"]
institution: ["Institute for Artificial Intelligence, Peking University", "NLCo Lab, Beijing Institute for General Artificial Intelligence", "Department of Automation, Tsinghua University", "Shanghai Jiao Tong University", "Institute of Automation, Chinese Academy of Sciences", "University of California, Los Angeles"]
description: "本文提出 LATENT SEEK 框架，通过在潜在空间中进行测试时实例级适应，显著提升大型语言模型的推理能力，同时避免参数更新的高成本。"
---

> **Summary:** 本文提出 LATENT SEEK 框架，通过在潜在空间中进行测试时实例级适应，显著提升大型语言模型的推理能力，同时避免参数更新的高成本。 

> **Keywords:** LLM, Latent Space, Test Time Scaling, Reasoning, Policy Gradient

**Authors:** Hengli Li, Chenxi Li, Tong Wu, Xuekai Zhu, Yuxuan Wang, Zhaoxin Yu, Eric Hanchen Jiang, Song-Chun Zhu, Zixia Jia, Ying Nian Wu, Zilong Zheng

**Institution(s):** Institute for Artificial Intelligence, Peking University, NLCo Lab, Beijing Institute for General Artificial Intelligence, Department of Automation, Tsinghua University, Shanghai Jiao Tong University, Institute of Automation, Chinese Academy of Sciences, University of California, Los Angeles


## Problem Background

大型语言模型（LLMs）在复杂推理任务中面临挑战，传统训练方法（如监督微调和强化学习）存在灾难性遗忘和高计算成本等问题，同时新型训练数据的有限性也限制了性能提升。
论文提出通过测试时扩展（test-time scaling），在不更新模型参数的情况下增强推理能力，解决传统方法的局限性。

## Method

*   **核心思想:** 提出 LATENT SEEK 框架，通过测试时实例级适应（Test-Time Instance-Level Adaptation, TTIA）在潜在空间（latent space）中优化推理过程，而无需更新模型参数。
*   **潜在空间操作:** 在语言模型头（LM head）之前的隐藏状态空间中，优化潜在表示（latent representations），利用其语义丰富性（semantic richness）来指导推理。
*   **策略梯度优化:** 采用策略梯度方法（policy gradient method, REINFORCE），以自生成的奖励信号（self-generated reward signals）为指导，迭代更新潜在表示，针对每个问题实例动态调整推理路径。
*   **独立采样与更新:** 假设 token 间的条件独立性，逐个 token 更新潜在表示，降低计算复杂度，同时通过贪婪解码（greedy decoding）提高效率。
*   **奖励机制:** 奖励函数采用自奖励机制（self-rewarding），依赖模型内部能力评估推理质量；此外，测试了完美稀疏奖励模型（Perfect Sparse Reward Model, PSRM）以探索优化上限。
*   **增强技术:** 包括 CoT 初始化（Chain-of-Thought initialization），利用 CoT 推理序列作为起点；以及部分序列优化（fractional sequence optimization），仅优化潜在表示子序列以提高稳定性和效率。
*   **关键优势:** 避免了训练时的高计算成本和灾难性遗忘问题，同时通过潜在空间操作实现高效的测试时推理优化。

## Experiment

*   **性能提升:** LATENT SEEK 在多个推理基准数据集（GSM8K, MATH-500, AIME2024）上显著优于基线方法，如在 GSM8K 上平均提升 10.75%，MATH-500 上提升 3.93%，AIME2024 上提升 4.73%；使用 LLaMA3.1-8B-Instruct 作为骨干模型时，相比 Genius 和 SimpleRL-Zoo 分别提升 12.7% 和 18.1%。
*   **测试时扩展效果:** 性能随测试时迭代次数增加而提升，尤其在完美稀疏奖励模型（PSRM）下，平均提升达 19.12%，证明潜在空间中测试时扩展的潜力。
*   **计算效率:** 对于平均复杂度的任务，LATENT SEEK 通常在 2 次迭代内收敛（GSM8K 和 MATH-500 平均迭代次数为 0.86 和 1.23），显示出高效性。
*   **泛化性:** 方法在不同模型家族（Qwen2, LLaMA3.1, Mistral）和规模上均表现优异，尤其在小规模模型（如 Qwen2.5-1.5B-Instruct）上，通过潜在空间优化激活隐含知识，性能接近大模型。
*   **实验设置合理性:** 实验覆盖多种模型、数据集和提示类型，与多种基线（无训练提示、强化学习、监督微调）对比全面；但自奖励机制的准确性可能受限，小规模模型表现不足，需进一步改进。

## Further Thoughts

潜在空间作为推理优化的新领域展示了巨大潜力，启发我们探索多层潜在表示联合优化或跨任务应用（如多模态推理）；测试时扩展的高效性提示未来模型设计可在训练和测试阶段平衡计算资源；自奖励与稀疏奖励的对比启发探索更高效的奖励设计，如结合外部验证信号或多智能体框架；小规模模型隐含知识的激活表明轻量化模型结合测试时优化可能成为未来研究方向。