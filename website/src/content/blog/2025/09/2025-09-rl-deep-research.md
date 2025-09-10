---
title: "Reinforcement Learning Foundations for Deep Research Systems: A Survey"
pubDatetime: 2025-09-08T14:27:23+00:00
slug: "2025-09-rl-deep-research"
type: "arxiv"
id: "2509.06733"
score: 0.6475027433829486
author: "grok-3-latest"
authors: ["Wenjun Li", "Zhi Chen", "Jingru Lin", "Hannan Cao", "Wei Han", "Sheng Liang", "Zhi Zhang", "Kuicai Dong", "Dexun Li", "Chen Zhang", "Yong Liu"]
tags: ["Reinforcement Learning", "Agentic AI", "Data Synthesis", "Reward Design", "Multi-Modal Integration"]
institution: ["Huawei Technologies Co., Ltd"]
description: "本文系统梳理了强化学习在深度研究系统中的应用，提出从数据合成到训练框架的全面视角，为构建健壮的代理型AI提供了理论和实践指导。"
---

> **Summary:** 本文系统梳理了强化学习在深度研究系统中的应用，提出从数据合成到训练框架的全面视角，为构建健壮的代理型AI提供了理论和实践指导。 

> **Keywords:** Reinforcement Learning, Agentic AI, Data Synthesis, Reward Design, Multi-Modal Integration

**Authors:** Wenjun Li, Zhi Chen, Jingru Lin, Hannan Cao, Wei Han, Sheng Liang, Zhi Zhang, Kuicai Dong, Dexun Li, Chen Zhang, Yong Liu

**Institution(s):** Huawei Technologies Co., Ltd


## Problem Background

深度研究系统（Deep Research Systems）作为代理型AI，旨在通过协调推理、开放网络搜索和工具使用来解决复杂多步骤任务。然而，传统训练方法如监督微调（SFT）和直接偏好优化（DPO）存在显著局限性：SFT 受限于模仿偏差和暴露偏差，无法有效利用环境反馈；DPO 依赖于人为设计的决策点和子技能，难以处理长距离信用分配和多目标权衡问题。本文聚焦于通过强化学习（RL）克服这些局限性，实现端到端的轨迹级策略优化，支持探索、恢复行为和多步骤决策，同时减少对人类先验和标注的依赖。

## Method

* **数据合成与Curation**：提出构建复杂查询和高质量训练数据的策略，包括跨文档组合（整合多源证据）、结构驱动路径增长（通过超链接图或集合操作扩展任务难度）和难度分级变换（通过重写或步骤监督逐步提升难度）。此外，采用污染/新颖性过滤、结果验证和过程质量筛选等 curation 手段，确保数据支持长距离推理和工具使用。
* **RL方法**：包括训练体制优化、奖励设计与信用分配以及多模态集成。具体而言：
  * **训练体制优化**：采用可选的冷启动（SFT/RSFT）以稳定早期训练，结合课程学习（如从发现到精炼的阶段性难度提升）和动态采样策略（如DUPO）提高样本效率；优化器多采用PPO或GRPO，结合KL正则化和工具返回掩码以稳定长距离任务训练；此外，提出上下文控制（如历史压缩）和搜索必要性学习（决定何时搜索）以降低计算成本。
  * **奖励设计与信用分配**：设计结果级奖励（评估最终答案正确性，如EM/F1）和步骤级奖励（评估工具调用、信息增益等中间行为），通过轨迹级或回合级估计器（如MT-GRPO）实现精细信用分配，解决长距离任务中的延迟反馈问题。
  * **多模态集成**：针对多模态研究代理，提出感知作为行动（Perception-as-Action）的概念，将视觉操作（如裁剪、标注）纳入统一动作空间，并通过证据必要性学习决定查询模态和工具使用顺序。
* **代理型RL训练框架**：从系统视角解决长距离任务训练瓶颈，提出异步采样（如AReaL的actor-learner设计）、分布式经验收集（如AWorld的集群级协调）和过程监督（如OpenR的PRM引导解码），提升采样吞吐量和训练稳定性。

## Experiment

* **有效性**：RL方法在多跳QA（如HotpotQA）、VQA和工具交互任务（如GAIA）上表现出优于SFT/DPO的性能，尤其在长距离推理、工具使用效率和恢复行为方面有显著提升。例如，MHGPO在多跳QA任务中优于标准MAPPO，显示出联合优化的潜力。
* **全面性**：实验设置覆盖了多个基准数据集（包括QA、VQA、长篇文本生成和领域特定任务），任务类型多样，评估了从单模态到多模态、从静态语料到动态网络环境的广泛场景，较为合理。
* **局限性与开销**：尽管RL方法在性能上占优，但训练成本较高，尤其是在长距离任务中，采样和计算开销显著（如需要大量工具交互和奖励计算）；此外，对真实环境非确定性（如网络搜索动态变化）的处理可能不足，实验结果可能高估了实际部署效果。

## Further Thoughts

论文提出的‘证据必要性’（Evidence Necessity）概念非常具有启发性，即代理学习决定何时以及从哪个模态（文本或图像）获取证据。这一思想可以扩展到更广泛的资源分配问题，例如在多工具环境中动态选择最优工具组合，甚至在计算资源受限时优化任务优先级。此外，课程学习和动态采样策略也启发我们是否可以通过自适应难度调整和基于不确定性的任务选择，进一步提高训练效率，尤其是在边缘设备或低资源场景下的应用。