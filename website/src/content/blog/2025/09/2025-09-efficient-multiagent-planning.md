---
title: "ELHPlan: Efficient Long-Horizon Task Planning for Multi-Agent Collaboration"
pubDatetime: 2025-09-29T03:15:56+00:00
slug: "2025-09-efficient-multiagent-planning"
type: "arxiv"
id: "2509.24230"
score: 0.7205549193990173
author: "grok-3-latest"
authors: ["Shaobin Ling", "Yun Wang", "Chenyou Fan", "Tin Lun Lam", "Junjie Hu"]
tags: ["LLM", "Multi-Agent Collaboration", "Task Planning", "Efficiency Metrics", "Dynamic Adaptation"]
institution: ["The Chinese University of Hong Kong, Shenzhen", "City University of Hong Kong", "South China Normal University"]
description: "本文提出 ELHPlan 框架，通过引入绑定意图的 Action Chain 和验证-精炼机制，实现多智能体长程任务规划中效率与适应性的平衡，显著降低计算成本并保持任务成功率。"
---

> **Summary:** 本文提出 ELHPlan 框架，通过引入绑定意图的 Action Chain 和验证-精炼机制，实现多智能体长程任务规划中效率与适应性的平衡，显著降低计算成本并保持任务成功率。 

> **Keywords:** LLM, Multi-Agent Collaboration, Task Planning, Efficiency Metrics, Dynamic Adaptation

**Authors:** Shaobin Ling, Yun Wang, Chenyou Fan, Tin Lun Lam, Junjie Hu

**Institution(s):** The Chinese University of Hong Kong, Shenzhen, City University of Hong Kong, South China Normal University


## Problem Background

多智能体（多机器人）协作在动态环境中完成复杂长程任务是一个核心挑战。传统声明式规划方法缺乏适应性，无法应对部分可观测或动态变化的环境，而迭代式规划方法虽然灵活，但计算成本高，尤其在团队规模和任务复杂性增加时，计算开销和延迟显著上升。此外，多智能体协作中，智能体间的意图沟通和冲突协调成本高或易出错。论文旨在解决如何在多智能体长程任务规划中平衡灵活性和计算效率，同时实现有效的意图共享和资源优化。

## Method

* **核心思想：** 提出 ELHPlan 框架，引入 'Action Chain' 作为基本规划单元，即绑定子目标意图的动作序列，通过一次性生成一段动作序列而非逐个动作，减少规划频率和计算开销，同时通过意图绑定减少意图推断成本。
* **具体实现：** ELHPlan 采用四阶段循环流程：
  * **构建阶段（Construction）：** 通过单次大型语言模型（LLM）调用生成每个智能体的 Action Chain，包含动作序列和明确意图，动态调整链长度以适应子目标复杂性，并在不确定性高的位置插入 'replan' 占位符以便后续调整。
  * **验证阶段（Validation）：** 检查 Action Chain 中未执行动作的可行性（基于当前环境状态和动作前提条件）以及智能体间冲突（如多个智能体同时操作同一对象），确保动作有效性。
  * **精炼阶段（Refinement）：** 针对验证中发现的问题，采用三种机制调整 Action Chain：链精炼（Chain Refinement）替换无效动作、冲突解决（Conflict Resolution）调整冲突智能体的计划、链插入（Chain Insertion）在 'replan' 处生成新链。
  * **执行阶段（Execution）：** 将验证通过的动作交给执行模块，进入下一轮循环直到任务完成。
* **关键特点：** 通过共享内存支持集中式协调，避免频繁的自然语言通信；通过分块规划和动态精炼，兼顾了声明式规划的效率和迭代式规划的适应性。

## Experiment

* **有效性：** 在 TDW-MAT 和 C-WAH 两个模拟环境中，ELHPlan 在任务成功率（如 Transport Rate 在 TDW-MAT 中达 0.816）上与最先进方法（如 REVECA）相当，同时显著降低了计算成本，Token 消耗仅为基线方法的 24%-35.7%，推理时间减少了 74.2%-90.7%。
* **优越性：** 相比基线方法（如 CoELA 和 REVECA），ELHPlan 在计算效率和响应速度上具有明显优势，尤其在实时性要求高的场景中表现出色；此外，方法对不同 LLM 模型（如 GPT-4o, Llama 3.1）的性能波动较小，展现了较强的鲁棒性。
* **实验设置：** 实验覆盖了多种任务场景（24 个 TDW-MAT 任务和 10 个 C-WAH 任务），并引入了新的效率度量（Token Consumption 和 Inference Time），评估全面且合理；消融实验进一步验证了 Action Chain 和精炼模块对效率提升的关键作用。
* **不足：** 在路径规划上稍显劣势（Move Distance 较长），可能与空间推理能力不足有关；实验局限于模拟环境，缺乏真实机器人部署数据。

## Further Thoughts

Action Chain 的设计为任务规划提供了一种新的抽象方式，不仅适用于多智能体协作，也可能扩展到单智能体任务或跨领域问题（如游戏AI）；此外，论文提出的效率度量（如 Token Consumption）为 LLM 驱动系统的评估提供了新视角，未来可应用于资源受限场景（如边缘设备）；另一个启发是，验证-精炼机制是否可以结合强化学习，通过长期交互优化冲突解决策略，或通过轻量级预训练模型替代部分 LLM 调用，进一步降低成本。