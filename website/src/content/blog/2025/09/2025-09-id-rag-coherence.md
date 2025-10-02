---
title: "ID-RAG: Identity Retrieval-Augmented Generation for Long-Horizon Persona Coherence in Generative Agents"
pubDatetime: 2025-09-29T16:54:51+00:00
slug: "2025-09-id-rag-coherence"
type: "arxiv"
id: "2509.25299"
score: 0.6322103789822203
author: "grok-3-latest"
authors: ["Daniel Platnick", "Mohamed E. Bengueddache", "Marjan Alirezaie", "Dava J. Newman", "Alex Sandy Pentland", "Hossein Rahnama"]
tags: ["LLM", "Generative Agents", "Retrieval-Augmented Generation", "Persona Coherence", "Knowledge Graph"]
institution: ["Flybits Labs", "Toronto Metropolitan University", "Massachusetts Institute of Technology", "Stanford University"]
description: "本文提出 ID-RAG 机制，通过结构化的身份知识图谱和检索增强生成技术，显著提升了生成式智能体在长时间任务中的人格一致性和行动对齐度，同时提高了模拟效率。"
---

> **Summary:** 本文提出 ID-RAG 机制，通过结构化的身份知识图谱和检索增强生成技术，显著提升了生成式智能体在长时间任务中的人格一致性和行动对齐度，同时提高了模拟效率。 

> **Keywords:** LLM, Generative Agents, Retrieval-Augmented Generation, Persona Coherence, Knowledge Graph

**Authors:** Daniel Platnick, Mohamed E. Bengueddache, Marjan Alirezaie, Dava J. Newman, Alex Sandy Pentland, Hossein Rahnama

**Institution(s):** Flybits Labs, Toronto Metropolitan University, Massachusetts Institute of Technology, Stanford University


## Problem Background

生成式智能体（Generative Agents）在长时间任务中面临人格一致性（Persona Coherence）挑战，随着长期记忆上下文的增长，智能体容易出现身份漂移（Identity Drift），导致行为矛盾、易受外界影响以及多智能体系统中幻觉传播等问题，影响其可靠性和系统稳定性。
作者的出发点是设计一种机制，通过显式的身份模型锚定智能体人格，确保其在长期交互中的一致性和可解释性。

## Method

*   **核心思想:** 提出 Identity Retrieval-Augmented Generation (ID-RAG) 机制，将智能体的身份信息从一般长期记忆中分离，存储在一个动态的知识图谱（Chronicle）中，并在每次决策时通过检索相关身份上下文增强工作记忆，指导行动生成。
*   **具体实现步骤:** 
    *   **感知与记忆构建:** 智能体首先感知环境输入，从长期记忆中检索相关经历，构建初始工作记忆。
    *   **身份查询与检索:** 基于工作记忆生成查询，从身份知识图谱中检索相关身份信息（如信念、特质、价值观），包括目标节点及其邻域扩展。
    *   **上下文增强:** 将检索到的身份信息格式化为自然语言，合并到工作记忆中，形成增强后的上下文。
    *   **行动生成:** 使用增强后的工作记忆条件化策略模型（通常为大型语言模型），生成与身份一致的行动。
    *   **身份更新（可选）:** 根据环境反馈和长期记忆更新身份知识图谱，适应长期变化。
*   **架构设计:** 引入 Human-AI Agents (HAis) 架构实现 ID-RAG，Chronicle 知识图谱基于 Perspective-Aware AI (PAi) 结构，存储智能体的核心身份信息。
*   **关键创新:** 通过显式身份模型和动态检索机制，避免身份信息被长期记忆稀释，确保长期一致性，同时提高行动的可解释性。

## Experiment

*   **实验设置:** 在 Concordia 框架中模拟‘Riverbend Elections’市长选举场景，测试三种条件：基线生成式智能体（无显式身份模型）、HAis 完全身份检索（模拟理想情况）和 HAis 实际 ID-RAG 实现，使用 GPT-4o、GPT-4o mini 和 Qwen2.5-7B 作为策略模型，评估指标包括身份回忆分数（Identity Recall Score）、行动一致性分数（Action Alignment Score）和模拟收敛时间。
*   **效果显著性:** ID-RAG 在所有模型上显著提升了身份回忆和行动一致性，尤其在后期时间步长缓解了身份漂移问题，例如 GPT-4o mini 在第 7 个时间步长时身份回忆分数从 0.51 提升到 0.58（Alice）；模拟收敛时间大幅缩短，如 GPT-4o mini 上减少了 58%。
*   **优越性与合理性:** 相比基线，ID-RAG 提供了更稳定的自我认知和行为一致性，尤其对较弱模型（如 GPT-4o mini）效果更明显，针对性上下文检索避免了信息过载；实验设置全面，涵盖不同模型和多轮模拟，但 Chronicle 规模较小且手动构建，身份更新模块未完全实现。
*   **开销与局限:** 主要增加了每次时间步长的身份检索计算开销，但效率提升（如模拟时间减少）弥补了这一成本；Qwen2.5-7B 模型稳定性较差，影响部分结果评估。

## Further Thoughts

ID-RAG 将身份作为显式、可检索的知识结构的设计非常具有启发性，这种分离式存储不仅提高了人格一致性，还增强了智能体的可解释性，为在安全关键任务中应用生成式智能体提供了可能性；此外，角色一致性（Role Coherence）的概念也值得关注，通过定义角色身份模型，可以确保智能体在特定任务中遵循协议和约束，这对构建可信赖的 AI 系统至关重要，或许未来可以探索如何动态平衡角色身份与个体人格之间的冲突。