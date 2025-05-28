---
title: "LLM-Agent-Controller: A Universal Multi-Agent Large Language Model System as a Control Engineer"
pubDatetime: 2025-05-26T06:30:13+00:00
slug: "2025-05-llm-control-agent"
type: "arxiv"
id: "2505.19567"
score: 0.4937603920620269
author: "grok-3-latest"
authors: ["Rasoul Zahedifar", "Sayyed Ali Mirghasemi", "Mahdieh Soleymani Baghshah", "Alireza Taheri"]
tags: ["LLM", "Multi-Agent System", "Control Theory", "Reasoning", "Task Automation"]
institution: ["Sharif University of Technology"]
description: "本文提出 LLM-Agent-Controller，一个基于多智能体的大型语言模型系统，通过自动化控制工程任务和用户友好的自然语言交互，显著提升了系统的适应性和可访问性。"
---

> **Summary:** 本文提出 LLM-Agent-Controller，一个基于多智能体的大型语言模型系统，通过自动化控制工程任务和用户友好的自然语言交互，显著提升了系统的适应性和可访问性。 

> **Keywords:** LLM, Multi-Agent System, Control Theory, Reasoning, Task Automation

**Authors:** Rasoul Zahedifar, Sayyed Ali Mirghasemi, Mahdieh Soleymani Baghshah, Alireza Taheri

**Institution(s):** Sharif University of Technology


## Problem Background

控制工程（Control Engineering）领域涉及动态系统的设计与分析，通常需要深厚的专业知识和复杂软件工具。随着现代系统复杂性的增加以及对用户友好解决方案的需求，传统方法的局限性日益凸显，尤其是在用户可访问性和跨学科知识整合方面。本文提出利用大型语言模型（LLM）的多智能体系统，旨在通过自动化和直观交互，降低对控制理论专业背景的依赖，提供一个集成且高效的解决方案。

## Method

* **核心思想**：构建一个名为 LLM-Agent-Controller 的多智能体系统，通过一个中央控制器智能体（Controller Agent）和多个辅助智能体协作，解决控制工程中的广泛问题。
* **系统架构**：系统包含一个监督者（Supervisor）负责高级决策和流程协调，中央控制器智能体处理核心任务（如控制器设计、系统仿真），辅助智能体（如 Retriever、Planner、Reasoner、Critic 等）分别负责数据检索、任务规划、逻辑推理和结果验证等功能。
* **工具支持**：控制器智能体集成了超过140个工具，基于 Python Control Library，覆盖系统表示、控制分析、控制器设计和时域/频域仿真等多个方面，确保任务执行的全面性。
* **高级技术**：系统采用检索增强生成（Retrieval-Augmented Generation, RAG）从外部知识库获取信息，使用思维链（Chain-of-Thought, CoT）和树状思维（Tree-of-Thought, ToT）提升推理能力，并通过自我批评与修正机制提高响应准确性。
* **用户交互**：支持自然语言输入，用户无需控制理论背景即可提出问题并获得实时解决方案，同时系统具备记忆功能，可存储和调用历史对话以提供个性化服务。
* **动态协作**：通过监督者的协调，智能体之间根据任务需求动态分配职责，确保工作流程的高效性和适应性。

## Experiment

* **有效性**：系统在五个控制理论任务类别（系统表示、控制分析、控制器设计、时域仿真等）上的整体成功率为83%，各智能体平均成功率为87%，表明方法在大多数任务中表现良好。
* **任务表现差异**：系统表示和时域仿真任务的综合得分最高（分别为0.95和0.94），而控制器设计任务得分最低（0.83），反映出复杂任务中规划和调试的挑战。
* **模型对比**：在三种大型语言模型上的测试显示，ChatGPT-4o 表现最佳（综合得分0.89），但成本最高；DeepSeek-V3 成本最低（每运行仅$0.0005），但性能较差（综合得分0.81）；Claude 3.7 Sonnet 在性能和成本之间取得平衡（综合得分0.87）。
* **实验设置合理性**：实验设计较为全面，涵盖多种任务类别和模型，引入10个量化指标（如 Efficiency Score、Completion Score）评估智能体和系统整体表现，并通过每类别20次运行减少随机性影响。
* **不足与成本**：某些智能体（如 Debugger 和 Communicator）在特定任务中表现不稳定，高性能模型（如 ChatGPT-4o）的运行时间和成本较高（每运行$0.1566），可能限制大规模应用。

## Further Thoughts

多智能体架构通过任务分解和协作显著提升了大型语言模型在专业领域的应用能力，这种设计理念可推广至其他技术领域，如机械设计或医疗诊断，探索如何优化智能体间的职责分配和协作效率。此外，检索增强生成（RAG）与思维链（CoT）的结合为处理知识密集型任务提供了新思路，未来可以研究如何动态调整外部知识检索的精度和推理深度，以适应不同领域的复杂需求。