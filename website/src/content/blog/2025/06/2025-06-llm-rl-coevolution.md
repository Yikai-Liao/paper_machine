---
title: "Think Twice, Act Once: A Co-Evolution Framework of LLM and RL for Large-Scale Decision Making"
pubDatetime: 2025-06-03T06:52:37+00:00
slug: "2025-06-llm-rl-coevolution"
type: "arxiv"
id: "2506.02522"
score: 0.5437310967968302
author: "grok-3-latest"
authors: ["Xu Wan", "Wenyue Xu", "Chao Yang", "Mingyang Sun"]
tags: ["LLM", "Reinforcement Learning", "Decision Making", "Trajectory Refinement", "Industrial Control"]
institution: ["Zhejiang University", "Tongji University", "Alibaba DAMO Academy", "Peking University"]
description: "本文提出ACE框架，通过LLM的离线双角色轨迹精炼和RL的在线决策执行，显著提升了大规模工业决策任务中的控制效果和学习效率。"
---

> **Summary:** 本文提出ACE框架，通过LLM的离线双角色轨迹精炼和RL的在线决策执行，显著提升了大规模工业决策任务中的控制效果和学习效率。 

> **Keywords:** LLM, Reinforcement Learning, Decision Making, Trajectory Refinement, Industrial Control

**Authors:** Xu Wan, Wenyue Xu, Chao Yang, Mingyang Sun

**Institution(s):** Zhejiang University, Tongji University, Alibaba DAMO Academy, Peking University


## Problem Background

大规模工业决策任务（如电力系统运营）对复杂推理和快速响应提出了高要求。
大型语言模型（LLMs）在战略规划和推理方面表现出色，但由于自回归生成特性导致的延迟和长序列决策能力不足，无法满足工业控制的实时性和精细控制需求。
强化学习（RL）虽擅长数值优化，但在面对大规模动作空间时存在样本效率低和次优解问题。
现有方法无法同时解决高效学习和实时决策的挑战，因此需要一种新的协同框架结合两者的优势。

## Method

*   **框架概述：** 提出Agents Co-Evolution (ACE)框架，将LLM和RL的角色分离为离线训练和在线部署两个阶段，避免LLM在实时决策中的延迟问题，同时利用其推理能力提升RL学习效率。
*   **双角色轨迹精炼机制：** 在训练阶段，LLM扮演两个角色：
    *   **Policy Actor：** 针对RL生成的次优动作，通过多步推理和环境验证进行精炼，生成更优决策。具体方法是将状态和动作转化为自然语言描述，结合任务上下文输入LLM，输出修正后的动作，并通过环境模拟验证效果。
    *   **Value Critic：** 通过轨迹级别的奖励重塑，进行时间信用分配，解决长时依赖问题。LLM对关键决策点进行反事实推理，调整奖励值（离散化为四个级别），以更准确地评估长期影响。
*   **RL直接交互：** RL采用Soft Actor-Critic (SAC)算法，通过离线策略训练提高样本利用率，生成初始决策轨迹，存储于回放缓冲区。
*   **协同进化机制：** 构建混合经验缓冲区，结合RL交互数据和LLM精炼轨迹，通过奖励优先级采样策略优化数据质量；同时，RL生成的高质量数据用于LLM的在线微调，形成双向改进循环。
*   **实现细节：** 包括坏案例推理（将RL的次优动作作为负面示例输入LLM）和多轮推理（通过环境模拟验证LLM精炼动作效果，必要时进行多次修正），确保干预的针对性和有效性。

## Experiment

*   **性能提升：** 在三个L2RPN电力系统运营挑战（动作空间超60,000）中，ACE显著优于基线方法，包括专家指导的RL、纯LLM和LLM引导的RL。例如，在WCCI 2020挑战中，ACE回合奖励比纯RL高22.2%，比纯LLM高超130%；在WCCI 2022挑战中，比专家指导RL提升145%。
*   **实时性：** ACE保持了与专家指导RL相当的测试时间（例如WCCI 2020中为38.7秒），远低于纯LLM方法（高达数千秒），满足工业实时性需求。
*   **样本效率：** 通过少量LLM精炼（例如WCCI 2020中仅287次精炼），ACE显著提升RL收敛速度，相比传统方法需10万至20万样本，效率提升明显。
*   **消融研究：** 证明Actor和Critic模块对性能至关重要，去除任一模块导致奖励和生存率下降；坏案例推理和多轮推理策略也显著提升决策质量。
*   **实验设置合理性：** 实验覆盖多个难度和规模的电力系统挑战，数据来源于真实Grid2Op平台，包含负载波动、线路维护等现实场景，基线方法涵盖主流技术，对比公平且全面。

## Further Thoughts

ACE框架中LLM与RL的离线-在线分离协同模式启发我们思考如何在其他实时性要求高的领域（如自动驾驶、金融交易）中应用类似机制，利用LLM的推理能力提升学习效率而避免延迟问题；此外，双角色机制（Actor和Critic）展示了LLM在多任务应用中的潜力，可能适用于游戏AI或机器人控制等场景；混合缓冲区驱动的协同进化机制也为构建不同模型间的闭环学习系统提供了新思路，或许对未来多模态AI系统设计有借鉴意义。