---
title: "PokéAI: A Goal-Generating, Battle-Optimizing Multi-agent System for Pokemon Red"
pubDatetime: 2025-06-30T10:09:13+00:00
slug: "2025-06-pokeai-multiagent-pokemon"
type: "arxiv"
id: "2506.23689"
score: 0.45265767348080743
author: "grok-3-latest"
authors: ["Zihao Liu", "Xinhang Sui", "Yueran Song", "Siwen Wang"]
tags: ["LLM", "Multi-Agent System", "Game AI", "Planning", "Reasoning"]
institution: ["Qingdao Academy"]
description: "本文提出 PokéAI，一个完全基于文本的多智能体 LLM 框架，通过规划、执行和评估智能体的闭环协作，实现在《Pokémon Red》中自主目标生成与任务执行，战斗模块胜率达 80.8%，接近人类水平。"
---

> **Summary:** 本文提出 PokéAI，一个完全基于文本的多智能体 LLM 框架，通过规划、执行和评估智能体的闭环协作，实现在《Pokémon Red》中自主目标生成与任务执行，战斗模块胜率达 80.8%，接近人类水平。 

> **Keywords:** LLM, Multi-Agent System, Game AI, Planning, Reasoning

**Authors:** Zihao Liu, Xinhang Sui, Yueran Song, Siwen Wang

**Institution(s):** Qingdao Academy


## Problem Background

设计能够在开放环境中进行推理、规划和适应的通用智能体是 AI 领域的核心挑战。传统方法如强化学习或模仿学习依赖低级动作和手工奖励，适应性差且难以扩展，而现有基于大型语言模型（LLM）的系统多依赖多模态输入，计算成本高。本文以《Pokémon Red》为测试平台，旨在构建一个完全基于文本的多智能体 LLM 框架，解决如何在无需多模态输入的情况下实现目标生成、任务执行和结果验证的问题，降低计算成本并提升可扩展性。

## Method

* **系统架构**：提出 PokéAI，一个由三个专门智能体组成的闭环多智能体系统，分别为 Planning Agent（规划智能体）、Execution Agent（执行智能体）和 Critique Agent（评估智能体），全部由大型语言模型（LLM）驱动。
* **Planning Agent**：负责高层次决策，接收游戏状态信息（如玩家位置），从向量记忆库中提取长期上下文知识，生成游戏里程碑目标（如击败道馆），并分解为具体任务序列，动态调整策略以适应游戏变化。
* **Execution Agent**：接收任务并在游戏环境中执行，配备工具集（如导航工具，通过生成按键序列移动到指定位置），可通过 LLM 的函数调用能力触发工具。若任务无法完成，则请求规划智能体重新生成任务。遇到战斗时，自动触发战斗模块，通过监控游戏内存地址（0xD057）检测战斗状态，执行战斗决策。
* **Critique Agent**：任务完成后验证结果，检查游戏状态是否达到预期目标（如是否到达指定坐标），若未完成则要求执行智能体重试，完成后将控制权返回规划智能体，形成闭环。
* **战斗模块细节**：作为初步实现重点，战斗模块通过四个阶段运作：读取游戏内存中的战斗状态信息、将信息发送给 LLM、接收 LLM 响应、将响应转化为游戏内动作，形成独立闭环决策系统，直至战斗结束。

## Experiment

* **战斗模块性能**：在 Mt. Moon 场景中测试战斗 AI，对 50 次野生 Pokémon 遭遇战平均胜率为 80.8%，与经验丰富的人类玩家（86%）相差仅 6%，表明在简单 PvE 战斗中接近人类水平。
* **消融研究**：移除特定功能后，禁用道具使用对胜率影响最大（降至 32.6%），禁用 Pokémon 切换次之（58.8%），禁用逃跑影响最小（79.6%），显示道具管理和切换策略对持续战斗至关重要。
* **LLM 性能对比**：测试多个 LLM 后端，性能大致与 LLM Arena 语言任务评分成正比，但存在例外（如 Claude 3.5 Sonnet 胜率 75.8%，超出部分更高评分模型），不同 LLM 展现独特战斗风格（如 GPT-4o 激进，DeepSeek-v3 保守）。
* **长期记忆试点**：通过 Letta Agent 框架注入历史战斗记录，智能体能基于过去失败经验在类似场景中选择逃跑，初步表明长期记忆可提升决策质量。
* **实验设置评价**：实验设置较为全面，涵盖模块功能、模型选择和记忆机制的影响，但局限于战斗模块和简单场景，未测试完整系统协同效果，泛化性有待验证。

## Further Thoughts

多智能体闭环设计（规划-执行-评估）可推广至其他复杂任务，如自动驾驶或机器人控制，模块化分工提升系统适应性；不同 LLM 展现独特策略风格，提示通过调整模型参数或训练数据可定制智能体行为，模拟人类多样化决策；长期记忆增强决策的初步成功启发未来上下文学习和记忆增强研究，或许可以通过记忆分层或优先级机制进一步优化智能体在动态环境中的表现。