---
title: "Satori-SWE: Evolutionary Test-Time Scaling for Sample-Efficient Software Engineering"
pubDatetime: 2025-05-29T16:15:36+00:00
slug: "2025-05-evolutionary-test-scaling"
type: "arxiv"
id: "2505.23604"
score: 0.5976100478698898
author: "grok-3-latest"
authors: ["Guangtao Zeng", "Maohao Shen", "Delin Chen", "Zhenting Qi", "Subhro Das", "Dan Gutfreund", "David Cox", "Gregory Wornell", "Wei Lu", "Zhang-Wei Hong", "Chuang Gan"]
tags: ["LLM", "Test Time Scaling", "Sampling", "Reasoning", "RLHF"]
institution: ["Singapore University of Technology and Design", "MIT", "UMass Amherst", "Harvard", "MIT-IBM Watson AI Lab, IBM Research"]
description: "本文提出进化测试时扩展（EvoScale），通过迭代选择与变异以及强化学习驱动的自我进化，使小型语言模型在软件工程任务中以极低的采样成本达到与 100B+ 参数模型相当的性能。"
---

> **Summary:** 本文提出进化测试时扩展（EvoScale），通过迭代选择与变异以及强化学习驱动的自我进化，使小型语言模型在软件工程任务中以极低的采样成本达到与 100B+ 参数模型相当的性能。 

> **Keywords:** LLM, Test Time Scaling, Sampling, Reasoning, RLHF

**Authors:** Guangtao Zeng, Maohao Shen, Delin Chen, Zhenting Qi, Subhro Das, Dan Gutfreund, David Cox, Gregory Wornell, Wei Lu, Zhang-Wei Hong, Chuang Gan

**Institution(s):** Singapore University of Technology and Design, MIT, UMass Amherst, Harvard, MIT-IBM Watson AI Lab, IBM Research


## Problem Background

大型语言模型（LLM）在标准化编码基准上表现良好，但在真实世界软件工程任务（如解决 GitHub 问题，SWE-Bench）中表现不佳，尤其是参数少于 100B 的小型模型。
这些任务往往涉及多文件、多文档的复杂推理，小型模型在零样本设置下准确率低于 10%，即使经过监督微调（SFT）也仅达 30%。
传统方法依赖昂贵的高质量数据进行微调，而测试时扩展（Test-Time Scaling）虽有效，但因需要大量采样和昂贵评分机制而效率低下。

## Method

*   **核心思想:** 提出进化测试时扩展（Evolutionary Test-Time Scaling, EvoScale），将生成过程视为进化过程，通过迭代的选择和变异逐步优化输出分布，减少找到正确解决方案所需的样本数量。
*   **具体实现:**
    *   **进化迭代:** 在每次迭代中，模型生成一批候选补丁（Patches），通过评分函数（如奖励模型或单元测试）选择得分最高的补丁作为条件提示（Conditional Prompt），指导下一轮生成。早期迭代注重探索（Exploration），后期注重利用（Exploitation）。
    *   **自我进化（Self-Evolve）:** 通过强化学习（RL）训练模型，使其在推理时无需外部验证器，基于自身生成的补丁进行自我改进。RL 使用基于潜能的奖励整形（Potential-Based Reward Shaping），通过计算当前补丁与前一补丁的得分差异，确保每次迭代的得分单调提升。
    *   **两阶段微调:** 包括经典监督微调（Classical SFT）和变异监督微调（Mutation SFT）。前者训练模型基于问题和代码上下文生成补丁，后者训练模型基于先前生成的补丁进行改进，确保模型具备变异能力。
*   **关键优势:** 不依赖大量外部数据或验证器，通过内部优化减少采样和计算成本，适用于资源受限场景。

## Experiment

*   **有效性:** 在 SWE-Bench-Verified 数据集上，Satori-SWE-32B 在贪婪解码下达到 35.8% 准确率，优于所有小型模型；在 Best@50 下达到 41.6%，与参数超过 100B 的模型（如 Llama3-SWE-RL-70B 的 Best@500）性能相当，但采样成本仅为后者的 1/10。
*   **样本效率:** EvoScale 通过迭代进化显著减少采样需求，运行时间（约 16.6 秒）远低于传统方法（如单元测试选择的 92.8 秒）。
*   **实验设置合理性:** 实验涵盖不同采样预算（N=5,10,15,20,25,50）和多种验证方式（奖励模型、单元测试、自我进化），并通过消融研究验证了 RL 和 SFT 的贡献，以及采样温度对性能的影响。
*   **局限性:** 实验主要基于无代理（Agentless）管道，未探索代理式（Agentic）环境中的表现。

## Further Thoughts

EvoScale 将进化算法引入测试时扩展的思路非常有启发性，是否可以将其泛化到其他领域（如数学推理或自然语言生成），通过迭代优化逐步逼近最优答案？
此外，自我进化机制是否可以通过更复杂的奖励设计或多模型协作进一步增强，例如动态调整迭代次数或引入外部知识以突破基础模型能力限制？