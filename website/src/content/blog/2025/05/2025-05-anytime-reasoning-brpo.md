---
title: "Optimizing Anytime Reasoning via Budget Relative Policy Optimization"
pubDatetime: 2025-05-19T17:58:44+00:00
slug: "2025-05-anytime-reasoning-brpo"
type: "arxiv"
id: "2505.13438"
score: 0.7331534715916481
author: "grok-3-latest"
authors: ["Penghui Qi", "Zichen Liu", "Tianyu Pang", "Chao Du", "Wee Sun Lee", "Min Lin"]
tags: ["LLM", "Test Time Scaling", "Reasoning", "Sampling", "Reinforcement Learning"]
institution: ["Sea AI Lab", "National University of Singapore"]
description: "本文提出 *AnytimeReasoner* 框架，通过预算采样、密集奖励和 BRPO 方差减少技术优化大型语言模型的随时推理性能，在不同 token 预算下显著提升准确率和 token 效率。"
---

> **Summary:** 本文提出 *AnytimeReasoner* 框架，通过预算采样、密集奖励和 BRPO 方差减少技术优化大型语言模型的随时推理性能，在不同 token 预算下显著提升准确率和 token 效率。 

> **Keywords:** LLM, Test Time Scaling, Reasoning, Sampling, Reinforcement Learning

**Authors:** Penghui Qi, Zichen Liu, Tianyu Pang, Chao Du, Wee Sun Lee, Min Lin

**Institution(s):** Sea AI Lab, National University of Singapore


## Problem Background

大型语言模型（LLMs）在推理任务中通过测试时计算扩展（test-time compute scaling）可以显著提升性能，但现有强化学习（RL）方法通常只优化固定大预算下的最终结果，导致训练和部署效率低下，尤其在 token 预算受限或推理中断时，模型无法有效总结答案。
论文旨在解决‘随时推理’（Anytime Reasoning）问题，即在任意 token 预算下，模型都能从不完整推理过程中提取最佳答案，以适应在线服务中动态变化的资源约束和用户需求。

## Method

*   **核心思想:** 提出 *AnytimeReasoner* 框架，通过从先验分布中采样 token 预算，强制模型在不同预算下截断推理并总结答案，从而优化随时推理性能，同时引入可验证的密集奖励（verifiable dense rewards）以提升 RL 训练效率。
*   **具体实现:**
    *   **预算采样（Budget Sampling）:** 从先验分布（如均匀分布、线性分布）中采样 token 预算，截断推理过程（thinking process），并要求模型基于截断内容总结答案。这种方式将稀疏奖励问题转化为密集奖励问题，为每个预算点提供可验证的反馈信号，改善 RL 中的信用分配（credit assignment）。
    *   **解耦优化（Decoupled Optimization）:** 将推理策略（thinking policy）和总结策略（summary policy）分开优化，使用不同的预算分布（如总结策略采用均匀分布），确保总结能力在各种预算下均表现良好，同时减少计算开销（总结 token 远少于推理 token）。
    *   **预算相对策略优化（Budget Relative Policy Optimization, BRPO）:** 提出一种新的方差减少技术，通过结合当前推理进度的奖励（基于历史预算的加权平均）和组内推理轨迹的平均回报，构建更稳定的优势估计（advantage estimation），相比传统 GRPO 方法显著降低训练过程中的方差。
*   **关键特点:** 不直接修改模型参数，而是通过预算采样和策略优化调整推理行为，确保模型在任意中断点都能输出合理答案，同时保持最终性能。

## Experiment

*   **有效性:** 在多个数学推理数据集（如 AIME2024, AMC2022, MATH500）上，*AnytimeReasoner* 的所有变体（uniform, linear, base）均显著优于 GRPO，尤其在小预算下 uniform 变体表现最佳，随时准确率（平均预算）从 GRPO 的 43.0% 提升至 46.0%，最终准确率（最大预算 8000 token）在 AIME2024 上从 28.9% 提升至 32.7%。
*   **优越性:** 相比 GRPO，密集奖励缩短了平均推理长度（token efficiency 提升），解耦优化显著改善了总结策略的性能，BRPO 进一步降低了训练方差，整体实现了更好的性能-效率权衡。
*   **实验设置合理性:** 实验覆盖了多种预算分布（uniform, linear, base）和多个基准数据集，消融研究详细分析了密集奖励、解耦优化和 BRPO 的独立贡献，设置较为全面；但未测试不同模型规模的效果，计算成本较高（30 小时，8 个 A100 GPU），可能限制方法在更大规模上的应用。

## Further Thoughts

论文中的预算采样和密集奖励设计启发了我，是否可以将这种‘随时推理’理念扩展到其他资源受限场景，如内存或计算时间限制下的任务优化？此外，密集奖励是否能应用于对话生成或代码生成等任务，通过为中间步骤提供更丰富的反馈信号来提升学习效率？解耦优化的思想也可能对多任务学习有借鉴意义，例如将复杂任务分解为子模块分别优化，但如何动态平衡各模块的训练目标仍需进一步探索。