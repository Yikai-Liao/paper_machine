---
title: "Afterburner: Reinforcement Learning Facilitates Self-Improving Code Efficiency Optimization"
pubDatetime: 2025-05-29T12:14:29+00:00
slug: "2025-05-afterburner-code-efficiency"
type: "arxiv"
id: "2505.23387"
score: 0.7083087868215676
author: "grok-3-latest"
authors: ["Mingzhe Du", "Luu Tuan Tuan", "Yue Liu", "Yuhao Qing", "Dong Huang", "Xinyi He", "Qian Liu", "Zejun Ma", "See-kiong Ng"]
tags: ["LLM", "Code Generation", "Efficiency Optimization", "Reinforcement Learning", "Iterative Framework"]
institution: ["Nanyang Technological University", "National University of Singapore", "The University of Hong Kong", "Xi’an Jiaotong University", "ByteDance"]
description: "本文提出了一种迭代优化框架（IOF），通过强化学习（GRPO）和实时执行反馈显著提升了大型语言模型生成代码的计算效率，并在迭代过程中展现了持续自改进能力。"
---

> **Summary:** 本文提出了一种迭代优化框架（IOF），通过强化学习（GRPO）和实时执行反馈显著提升了大型语言模型生成代码的计算效率，并在迭代过程中展现了持续自改进能力。 

> **Keywords:** LLM, Code Generation, Efficiency Optimization, Reinforcement Learning, Iterative Framework

**Authors:** Mingzhe Du, Luu Tuan Tuan, Yue Liu, Yuhao Qing, Dong Huang, Xinyi He, Qian Liu, Zejun Ma, See-kiong Ng

**Institution(s):** Nanyang Technological University, National University of Singapore, The University of Hong Kong, Xi’an Jiaotong University, ByteDance


## Problem Background

大型语言模型（LLMs）在代码生成中虽然能确保功能正确性，但在计算效率（computational efficiency）方面表现不佳，这在资源受限或时间敏感的实际应用中构成了性能瓶颈。
现有方法（如提示工程或微调）在提升代码效率上效果有限，缺乏自适应和持续优化的能力，因此本文致力于开发一种测试时迭代优化框架，以显著提高 LLM 生成代码的效率，同时维持功能正确性。

## Method

*   **核心框架：Iterative Optimization Framework (IOF)**：提出了一种闭环系统，通过两个核心组件实现代码效率优化：
    *   **Afterburner**：一个代码优化模型，基于输入的问题描述、效率指令和当前代码，生成改进后的代码版本，同时输出推理内容。
    *   **Monolith**：一个高保真代码执行沙箱，执行生成的代码并提供实证性能反馈（如执行时间、峰值内存使用、综合分数），用于指导后续优化。
*   **训练策略**：探索了三种方法训练 Afterburner 模型：
    *   **Supervised Fine-Tuning (SFT)**：通过成对的低效和高效代码样本，训练模型学习从低效到高效的转换模式，依赖于训练数据中的显式模式，但缺乏深层理解和泛化能力。
    *   **Direct Preference Optimization (DPO)**：基于离线偏好数据，通过直接优化对高效代码的偏好，使模型在生成时倾向于更优解决方案，相比 SFT 更具判断力，但受限于初始数据集的多样性。
    *   **Group Relative Policy Optimization (GRPO)**：一种基于强化学习的在线优化方法，通过 Monolith 的实时反馈生成多组候选代码，利用组内相对优势更新策略，强调探索和自适应能力，支持持续改进。
*   **奖励机制**：GRPO 的奖励函数综合考虑了格式控制（确保输出结构符合预定格式）、功能正确性（优先通过测试用例）和效率提升（基于相对性能改进），通过加权组合引导模型优化。
*   **数据集支持**：引入了新的 Venus 数据集，包含大量任务和人类解决方案，支持多语言效率评估，并基于此构建了 SFT、DPO 和 GRPO 的训练子集。
*   **迭代过程**：在推理时，IOF 通过多轮迭代优化代码，Afterburner 提出改进版本，Monolith 评估性能，若新版本优于当前版本则更新，否则保留当前最佳代码，直至达到预定迭代次数。

## Experiment

*   **有效性**：在 Venus 和 APPS 基准测试中，GRPO 显著优于 SFT 和 DPO，PASS@1（功能正确性）从 47% 提升至 62%，效率指标（如 BEYOND-T）从 31% 提升至 45%，表明其在保持正确性的同时大幅提升了代码效率。
*   **对比分析**：与未优化的 vanilla LLMs（如 OpenAI o4 mini）相比，GRPO 调优后的模型在效率上更接近甚至超越人类解决方案，尤其在时间效率上（8% 的代码优于所有人类方案），显示出显著改进。
*   **实验设置合理性**：实验覆盖了多种模型（开源和闭源）、多语言支持、以及不同效率目标（时间、内存、综合），Venus 数据集提供了丰富的参考解决方案（平均每任务 106.6 个），通过 bootstrapping 方法报告 95% 置信区间，确保了评估的统计可靠性。
*   **局限性与成本**：实验主要聚焦于算法竞赛类任务，未涉及复杂软件工程项目；迭代优化增加了推理时间成本（相比单次生成），可能不适用于所有场景，但论文认为这在长期部署中可通过高效代码的运行节省来抵消。

## Further Thoughts

GRPO 的在线强化学习机制通过实时反馈实现自适应优化，启发我思考是否可以将类似闭环系统应用于其他生成任务（如文本或图像生成），通过动态反馈优化特定目标；此外，IOF 的生成-评估-优化框架可能扩展到自动驾驶或游戏 AI 等领域，利用迭代改进提升策略性能；奖励函数的多维度设计也提示我在 RL 系统中需平衡多目标，尤其是在目标冲突时通过加权或动态调整来优化模型行为。