---
title: "SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning"
pubDatetime: 2025-06-30T17:58:13+00:00
slug: "2025-06-spiral-self-play"
type: "arxiv"
id: "2506.24119"
score: 0.578716336838253
author: "grok-3-latest"
authors: ["Bo Liu", "Leon Guertler", "Simon Yu", "Zichen Liu", "Penghui Qi", "Daniel Balcells", "Mickel Liu", "Cheston Tan", "Weiyan Shi", "Min Lin", "Wee Sun Lee", "Natasha Jaques"]
tags: ["LLM", "Self-Play", "Reinforcement Learning", "Reasoning", "Multi-Agent"]
institution: ["National University of Singapore", "Centre for Frontier AI Research (CFAR), A*STAR", "Northeastern University", "Sea AI Lab", "Plastic Labs", "University of Washington"]
description: "SPIRAL 通过自博弈在零和语言游戏中训练语言模型，利用多智能体多轮强化学习和角色条件优势估计（RAE），显著提升推理能力并实现跨领域迁移。"
---

> **Summary:** SPIRAL 通过自博弈在零和语言游戏中训练语言模型，利用多智能体多轮强化学习和角色条件优势估计（RAE），显著提升推理能力并实现跨领域迁移。 

> **Keywords:** LLM, Self-Play, Reinforcement Learning, Reasoning, Multi-Agent

**Authors:** Bo Liu, Leon Guertler, Simon Yu, Zichen Liu, Penghui Qi, Daniel Balcells, Mickel Liu, Cheston Tan, Weiyan Shi, Min Lin, Wee Sun Lee, Natasha Jaques

**Institution(s):** National University of Singapore, Centre for Frontier AI Research (CFAR), A*STAR, Northeastern University, Sea AI Lab, Plastic Labs, University of Washington


## Problem Background

当前大型语言模型（LLM）在推理能力提升上依赖人工设计的奖励函数、领域特定数据集和专家监督，这种方式在扩展到通用智能时面临可扩展性瓶颈。
SPIRAL 旨在通过自博弈（Self-Play）机制，在零和语言游戏中训练模型，消除对人类监督的依赖，解决如何在无领域特定数据的情况下培养通用推理能力并实现跨领域迁移的问题。

## Method

*   **核心思想:** 通过多智能体多轮强化学习，利用自博弈在零和语言游戏中训练语言模型，自动生成训练数据并提升推理能力。
*   **具体实现:** 
    *   **自博弈框架:** 模型与自身副本在游戏（如 TicTacToe、Kuhn Poker、Simple Negotiation）中对战，双方共享同一策略（policy），通过角色条件提示区分玩家身份，创造自适应难度（automatic curriculum）。
    *   **多轮交互设计:** 采用 turn-level Markov Decision Process (MDP)，每轮生成完整响应（包括推理过程和动作），训练模型在多步决策中维持上下文和策略规划能力。
    *   **角色条件优势估计（RAE）:** 针对多智能体环境的高方差问题，为每个游戏和角色维护独立基线，通过指数移动平均（EMA）更新基线，计算优势值（advantage），减少奖励估计方差，稳定训练。
    *   **分布式训练架构:** 使用 actor-learner 架构，多个 actor 并行生成游戏轨迹，集中 learner 进行策略梯度更新，提升计算效率。
*   **关键创新:** 自博弈避免了对静态对手的过拟合，RAE 防止了思维崩溃（thinking collapse），多游戏训练培养多样化推理技能。

## Experiment

*   **推理能力提升:** 仅在 Kuhn Poker 上训练的 SPIRAL 模型显著提升数学推理（如 MATH500 提升 10.6%，Minerva Math 提升 18.1%）和通用推理（如 MMLU-Pro 提升 10.5%），Qwen3-4B-Base 平均提升 8.7%，优于在 25,000 个专家轨迹上进行监督微调（SFT）的模型。
*   **自博弈优势:** 自博弈胜率稳定在 50-52%，避免了固定对手训练（如 Gemini）的过拟合问题（初期胜率 0%，后期 62.5%），推理迁移效果更优（数学推理 40% vs 35%）。
*   **游戏特异性与协同效应:** 不同游戏培养不同技能（TicTacToe 空间推理，Kuhn Poker 概率推理，Simple Negotiation 策略优化），多游戏训练平均胜率 44.9%，高于单一游戏专家（最高 34.4%）。
*   **RAE 必要性:** 无 RAE 时，模型 200 步后推理轨迹长度骤降至近零，数学推理性能下降 66%（35% 至 12%），证明 RAE 对稳定训练至关重要。
*   **实验设置合理性:** 涵盖多种游戏、基准测试（MATH500, GPQA 等）和模型规模（Qwen3-4B, DeepSeek-7B），数据量充足（51,200 游戏转换，46,792 数学解题），结果显著；不足是计算成本高（8 个 H100 GPU，25 小时/实验）。

## Further Thoughts

零和游戏作为‘推理健身房’的概念令人启发，是否可以通过设计特定游戏环境针对性提升模型在伦理判断或常识推理等弱项上的能力？此外，推理模式（如逐例分析）的高迁移率提示通用底层结构的可能，是否可以通过跨领域数据分析进一步提炼这些结构？多游戏训练的协同效应也表明多样化认知挑战的重要性，未来是否可以扩展到合作游戏或部分可观测环境以丰富训练多样性？