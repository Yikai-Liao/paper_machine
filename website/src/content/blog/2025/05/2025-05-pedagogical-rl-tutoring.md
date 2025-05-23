---
title: "From Problem-Solving to Teaching Problem-Solving: Aligning LLMs with Pedagogy using Reinforcement Learning"
pubDatetime: 2025-05-21T15:00:07+00:00
slug: "2025-05-pedagogical-rl-tutoring"
type: "arxiv"
id: "2505.15607"
score: 0.8422083643746894
author: "grok-3-latest"
authors: ["David Dinucu-Jianu", "Jakub Macina", "Nico Daheim", "Ido Hakimi", "Iryna Gurevych", "Mrinmaya Sachan"]
tags: ["LLM", "Reinforcement Learning", "Pedagogical Alignment", "Multi-Turn Dialogue", "Synthetic Data"]
institution: ["ETH Zurich", "ETH AI Center", "Ubiquitous Knowledge Processing Lab, TU Darmstadt", "Hessian Center for AI"]
description: "本文提出了一种基于在线强化学习的框架，通过合成学生-辅导者交互数据将大型语言模型对齐到教学目标，实现了教学质量与学生成功之间的可控权衡，同时保持了推理能力。"
---

> **Summary:** 本文提出了一种基于在线强化学习的框架，通过合成学生-辅导者交互数据将大型语言模型对齐到教学目标，实现了教学质量与学生成功之间的可控权衡，同时保持了推理能力。 

> **Keywords:** LLM, Reinforcement Learning, Pedagogical Alignment, Multi-Turn Dialogue, Synthetic Data

**Authors:** David Dinucu-Jianu, Jakub Macina, Nico Daheim, Ido Hakimi, Iryna Gurevych, Mrinmaya Sachan

**Institution(s):** ETH Zurich, ETH AI Center, Ubiquitous Knowledge Processing Lab, TU Darmstadt, Hessian Center for AI


## Problem Background

大型语言模型（LLMs）在教育领域具有巨大潜力，尤其是在个性化辅导中，但它们通常被优化为直接提供答案，而不是通过引导学生独立解决问题来促进学习。
这种倾向与有效的教育学原则相悖，优秀的辅导应注重学生的主动学习和问题解决能力的培养，而非单纯的结果输出。
因此，论文提出‘教学对齐’（Pedagogical Alignment）的概念，旨在将 LLMs 从‘答题者’转变为‘辅导者’，解决其在教育场景中缺乏教学策略的问题。

## Method

*   **核心思想:** 通过在线强化学习（RL）框架，将大型语言模型对齐到教学目标，使其在多轮对话中引导学生解决问题，而非直接泄露答案。
*   **具体实现:** 
    *   **在线 RL 训练:** 采用在线策略（on-policy）方法，模型直接从自身生成的对话中学习，避免离线数据带来的上下文漂移问题。
    *   **奖励函数设计:** 奖励函数结合学生解决问题的成功率（post-dialog solve rate）和教学质量（pedagogical quality），通过可调的惩罚权重（λ）平衡这两个目标，探索 Pareto 前沿。
    *   **合成数据生成:** 通过模拟学生-辅导者交互生成训练数据，避免对昂贵人工标注的依赖，降低成本。
    *   **多轮交互模拟:** 强调多轮对话的重要性，模拟真实辅导场景中学生与辅导者的动态交互，支持学生或辅导者发起对话的两种场景。
    *   **思考标签（Thinking Tags）:** 引入结构化的思考标签以提高模型的可解释性，允许辅导模型在隐藏标签内规划教学策略而不直接泄露给学生。
*   **关键点:** 该方法不依赖大规模人工标注数据，通过合成交互实现高效训练，同时保持模型的推理能力，适用于资源受限的场景。

## Experiment

*   **有效性:** 实验基于 BigMath 数据集和 MathTutorBench 基准，使用 Qwen2.5-7B-Instruct 模型，通过调整惩罚权重 λ，实现了学生解决率（∆ Solve rate）和答案泄露率（Leak Solution）之间的可控权衡，例如 λ=0.75 时在学生成功率和教学质量上达到平衡。
*   **对比优势:** 与监督微调（SFT）和多轮偏好优化（MDPO）等基线相比，RL 方法在减少答案泄露和提升教学质量（Ped-RM 评分）方面表现更好，同时接近甚至超过专有模型（如 LearnLM）的性能。
*   **推理能力保持:** 在 MMLU、GSM8K 和 MATH500 等基准上，模型未显著降低推理能力，优于 SocraticLM 等方法，表明教学对齐未以牺牲核心能力为代价。
*   **实验设置合理性:** 实验涵盖领域内（BigMath）和领域外（MathTutorBench）测试，评估多维度指标（学生解决率、答案泄露率、教学奖励模型分数），设置较为全面；但局限在于仅使用单一学生模型模拟交互，未反映真实学生多样性，且奖励信号为合成生成，未经真实学生验证。

## Further Thoughts

奖励函数中通过调整权重实现教学支持与学生成功之间的动态平衡，这一思想可扩展到其他交互式领域，如医疗对话或客户服务，平衡信息提供与用户自主性；此外，在线 RL 避免上下文漂移的特性适用于动态适应任务，而合成数据的低成本策略和思考标签的透明性机制也为资源受限场景和 AI 决策可解释性提供了新思路。