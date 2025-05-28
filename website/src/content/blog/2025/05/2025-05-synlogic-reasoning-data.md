---
title: "SynLogic: Synthesizing Verifiable Reasoning Data at Scale for Learning Logical Reasoning and Beyond"
pubDatetime: 2025-05-26T07:59:36+00:00
slug: "2025-05-synlogic-reasoning-data"
type: "arxiv"
id: "2505.19641"
score: 0.6180656174203997
author: "grok-3-latest"
authors: ["Junteng Liu", "Yuanxiang Fan", "Zhuo Jiang", "Han Ding", "Yongyi Hu", "Chi Zhang", "Yiqi Shi", "Shitong Weng", "Aili Chen", "Shiqi Chen", "Yunan Huang", "Mozhi Zhang", "Pengyu Zhao", "Junjie Yan", "Junxian He"]
tags: ["LLM", "Reasoning", "Synthetic Data", "Reinforcement Learning", "Generalization"]
institution: ["The Hong Kong University of Science and Technology", "MiniMax", "The City University of Hong Kong"]
description: "本文提出 SynLogic 框架和数据集，通过合成 35 个多样化、可验证的逻辑推理任务数据，显著提升大型语言模型的逻辑推理能力，并通过混合训练展示跨领域泛化的潜力。"
---

> **Summary:** 本文提出 SynLogic 框架和数据集，通过合成 35 个多样化、可验证的逻辑推理任务数据，显著提升大型语言模型的逻辑推理能力，并通过混合训练展示跨领域泛化的潜力。 

> **Keywords:** LLM, Reasoning, Synthetic Data, Reinforcement Learning, Generalization

**Authors:** Junteng Liu, Yuanxiang Fan, Zhuo Jiang, Han Ding, Yongyi Hu, Chi Zhang, Yiqi Shi, Shitong Weng, Aili Chen, Shiqi Chen, Yunan Huang, Mozhi Zhang, Pengyu Zhao, Junjie Yan, Junxian He

**Institution(s):** The Hong Kong University of Science and Technology, MiniMax, The City University of Hong Kong


## Problem Background

当前大型语言模型（LLMs）在推理能力上的进步主要集中于数学和编码领域，而通用推理能力的开发因缺乏多样化、可验证的训练数据而受限。
作者假设逻辑推理是通用推理的基础，提出关键问题：如何生成大规模、多样化、难度可控且可验证的逻辑推理数据，以通过强化学习（RL）提升 LLMs 的通用推理能力？

## Method

*   **框架设计：SynLogic 数据合成框架**：提出一个包含 35 个逻辑推理任务的数据合成框架，覆盖从经典逻辑谜题（如数独、24 点游戏）到基准任务（如 BBH、BBEH）的广泛类型，确保任务多样性。
*   **难度控制机制**：通过任务特定参数（如数独网格大小）调整难度，使用强推理模型（如 DeepSeek R1 和 OpenAI-o3-mini）设定难度上下限，确保数据既具挑战性又可学习；同时为不同规模模型设计了 SynLogic-Easy 和 SynLogic-Hard 两个版本。
*   **数据生成与验证**：为每个任务开发规则化的生成代码，确保生成实例符合逻辑约束，并配备任务特定的验证器，自动检查答案正确性，支持 RL 训练中的可验证奖励设计。
*   **自然语言转化**：将抽象逻辑实例通过任务特定模板转化为自然语言提示，便于 LLMs 的训练和评估。
*   **强化学习训练**：基于 GRPO 算法和 DAPO 技术，使用 SynLogic 数据对 Qwen2.5 模型（7B 和 32B）进行强化学习，奖励函数结合格式合规性和答案正确性（二元奖励：正确且格式符合为 1，否则为 0）。
*   **混合训练探索**：将 SynLogic 数据与数学和编码数据混合，用于 RL 训练，探索逻辑推理对其他领域任务的促进作用和跨领域泛化能力。

## Experiment

*   **逻辑推理性能提升**：SynLogic 训练的模型在逻辑推理基准上表现优异，7B 模型在 KOR-Bench 上比 Qwen2.5-7B-Instruct 提升近 10 个百分点，32B 模型在 BBEH 上比 DeepSeek-R1-Distill-Qwen-32B 提升 6 个百分点，确立了开源逻辑推理数据集的领先地位。
*   **跨领域泛化效果**：尽管主要训练于逻辑数据，模型在数学基准上也显著提升，7B 模型在 AIME 2024 上从 0.3% 提升至 10.0%，32B 模型从 4.5% 提升至 19.6%，表明逻辑推理能力对数学任务有正向迁移。
*   **混合训练效率**：将 SynLogic 数据与数学或编码数据混合训练后，在相同训练步数下性能相当，但消耗的领域特定数据更少（如数学数据），显示逻辑推理数据提升了其他领域的训练效率。
*   **实验设置合理性**：实验覆盖多个逻辑和数学基准，采用 zero-shot 评估，难度控制通过 avg@8 和 pass@8 指标验证合理；针对不同模型规模设计 Easy 和 Hard 数据集，适应性强；但受限于计算资源，未对每个任务精细调优难度，也未实现动态难度调整。
*   **行为观察**：训练过程中，模型响应长度增加（7B 约 2500 token，32B 约 4000 token），反思行为比例上升，表明逻辑推理任务与长思考模式契合。

## Further Thoughts

SynLogic 框架的可扩展性启发我们是否可以将类似的数据合成方法应用于其他推理领域（如常识推理或情感推理），以解决数据稀缺问题；
难度动态调整的未实现潜力提示是否可以通过自适应算法根据模型性能实时调整任务难度，模拟人类学习的渐进挑战；
混合训练的协同效应表明不同推理任务间存在共享机制，未来是否可以构建一个‘推理任务图谱’，系统分析任务间的依赖和互补性，优化训练策略？