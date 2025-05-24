---
title: "Self-GIVE: Associative Thinking from Limited Structured Knowledge for Enhanced Large Language Model Reasoning"
pubDatetime: 2025-05-21T03:30:55+00:00
slug: "2025-05-self-give-associative-reasoning"
type: "arxiv"
id: "2505.15062"
score: 0.6075085729446923
author: "grok-3-latest"
authors: ["Jiashu He", "Jinxuan Fan", "Bowen Jiang", "Ignacio Houine", "Dan Roth", "Alejandro Ribeiro"]
tags: ["LLM", "Knowledge Graph", "Associative Thinking", "Reinforcement Learning", "Reasoning"]
institution: ["University of Pennsylvania", "University of California, Berkeley"]
description: "Self-GIVE 通过检索与强化学习结合，提出高效的关联性思维框架，使 LLMs 在有限结构化知识下显著提升知识密集型任务推理能力，同时降低 token 消耗并适配小型模型。"
---

> **Summary:** Self-GIVE 通过检索与强化学习结合，提出高效的关联性思维框架，使 LLMs 在有限结构化知识下显著提升知识密集型任务推理能力，同时降低 token 消耗并适配小型模型。 

> **Keywords:** LLM, Knowledge Graph, Associative Thinking, Reinforcement Learning, Reasoning

**Authors:** Jiashu He, Jinxuan Fan, Bowen Jiang, Ignacio Houine, Dan Roth, Alejandro Ribeiro

**Institution(s):** University of Pennsylvania, University of California, Berkeley


## Problem Background

大型语言模型（LLMs）在知识密集型任务（如生物医学问答）中，由于缺乏新知识或直接上下文，推理能力受限，容易产生幻觉或推理失败。
现有方法如检索增强生成（RAG）和基于知识图谱（KG）的推理框架（如 GIVE）存在效率低、token 消耗高、以及对小型模型不友好的问题，论文提出 Self-GIVE，旨在通过自动化的关联性思维（associative thinking）增强 LLMs 的推理能力，尤其是在结构化知识有限的情况下。

## Method

*   **核心思想:** 通过检索有限结构化知识并结合强化学习（RL），让 LLMs 学习关联性思维，将查询与不完整的外部知识有效连接，模拟人类在缺乏直接信息时的推理过程。
*   **具体实现步骤:**
    *   **结构化知识与实体组构建:** 从查询中提取关键实体（queried concepts），基于语义相似性在知识图谱（KG）中构建包含查询实体及其相似实体的实体组（entity groups），然后检索实体组之间的 KG 三元组（triplets），以提供与查询间接相关的结构化知识，避免传统 RAG 中检索无关信息的缺陷。
    *   **强化学习优化关联性思维:** 使用 GRPO（Grouped Reinforced Policy Optimization）算法对模型进行微调，奖励函数基于答案准确性和推理格式（而非显式监督推理过程），训练时预计算检索的知识三元组和实体组，直接嵌入上下文提示中，推理时避免重复模型调用。
    *   **提示设计:** 提示模板简洁，包含查询、实体组和知识三元组三部分，避免复杂指令，鼓励模型自主探索关联性思维。
*   **关键创新:** 不依赖完整 KG 或直接上下文，通过实体组和 RL 训练，让模型学习从有限知识中推断新关系，适用于小型模型（3B/7B），并大幅降低 token 消耗。

## Experiment

*   **有效性:** 在知识密集型生物医学问答数据集（PubmedQA、BioASQ、ProcessBank）上，Self-GIVE 显著提升模型性能，Qwen2.5 3B 模型准确率提升高达 42.8%（从 0.110 到 0.510），7B 模型提升高达 31%（从 0.230 到 0.540），尤其在未见样本上表现优异。
*   **与基线对比:** 相比传统方法（如 k-shot 提示、CoT、RAG）以及基于 GPT3.5T 的代理框架（如 GraphRAG、ToG、GIVE），Self-GIVE 在小型模型上表现更优，7B 模型甚至超越 GIVE+GPT3.5T 的性能。
*   **效率:** token 消耗降低超过 90%，推理时模型调用次数为 O(1)，而非线性增长。
*   **消融研究:** 实体组中额外实体数量（p）最优值在 1-3 之间，知识三元组密度（信息增益）需适中，过稀疏或过密集均导致性能下降。
*   **实验设置合理性:** 实验涵盖不同模型大小（3B/7B）、版本（Base/Instruct）、数据集和任务类型，使用小型 UMLS 知识图谱（135 节点），验证了方法的通用性和显著性。

## Further Thoughts

关联性思维的概念可以扩展到其他外部知识形式（如文本语料或图像），探索通用关联性思维框架是一个有趣方向；强化学习中奖励设计的简单性启发更细粒度的奖励机制（如推理链忠实度）以提升可解释性；Self-GIVE 对小型模型的适配性表明未来可聚焦资源受限环境下的轻量级微调和检索策略。