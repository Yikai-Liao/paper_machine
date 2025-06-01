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
institution: ["Singapore University of Technology and Design", "Massachusetts Institute of Technology", "University of Massachusetts Amherst", "Harvard University", "MIT-IBM Watson AI Lab"]
description: "本文提出进化测试时扩展（EvoScale）方法，通过进化算法和强化学习实现样本高效的测试时优化，使小型语言模型在软件工程任务上接近大型模型性能。"
---

> **Summary:** 本文提出进化测试时扩展（EvoScale）方法，通过进化算法和强化学习实现样本高效的测试时优化，使小型语言模型在软件工程任务上接近大型模型性能。 

> **Keywords:** LLM, Test Time Scaling, Sampling, Reasoning, RLHF

**Authors:** Guangtao Zeng, Maohao Shen, Delin Chen, Zhenting Qi, Subhro Das, Dan Gutfreund, David Cox, Gregory Wornell, Wei Lu, Zhang-Wei Hong, Chuang Gan

**Institution(s):** Singapore University of Technology and Design, Massachusetts Institute of Technology, University of Massachusetts Amherst, Harvard University, MIT-IBM Watson AI Lab


## Problem Background

语言模型（LMs）在标准化编码基准（如 HumanEval）上表现良好，但在真实世界软件工程（SWE）任务（如解决 GitHub 问题）中表现不佳，尤其是参数少于 100B 的小型模型，零样本准确率低于 10%，即使经过监督微调（SFT）也仅达 30%。
现有方法依赖昂贵的高质量数据 SFT 或测试时扩展（test-time scaling），后者因生成和评分成本高而效率低下，特别是在 SWE 任务中。
论文旨在解决如何以样本高效的方式提升小型模型在 SWE 任务上的性能，减少测试时样本需求，同时接近大型模型的准确率。

## Method

*   **核心思想**：提出进化测试时扩展（Evolutionary Test-Time Scaling, EvoScale），将生成过程视为进化过程，通过迭代的选择和变异优化输出分布，减少找到正确解决方案所需的样本数。
*   **具体实现**：
    *   **进化迭代**：将样本预算分摊到多个迭代中，每轮生成一小批代码补丁（patches），通过评分函数（如奖励模型）选择得分最高的补丁作为条件提示，指导下一轮生成，早期探索，后期利用。
    *   **变异操作**：使用语言模型本身作为变异算子，基于先前补丁生成语法和语义有效的改进补丁，避免随机噪声破坏代码结构。
    *   **两阶段监督微调（SFT）**：首先进行经典 SFT，基于问题描述和代码上下文训练模型生成补丁；随后进行变异 SFT，基于先前生成的补丁训练模型学习如何改进输出。
    *   **自进化强化学习（RL）**：通过 RL 训练模型自我改进，使用基于潜能的奖励整形（potential-based reward shaping），基于迭代间得分差异优化输出质量，确保单调改进，无需推理时依赖外部评分模型。
*   **关键创新**：结合进化算法与语言模型生成，通过 RL 内化奖励机制，降低测试时计算成本，同时保持样本效率。

## Experiment

*   **性能提升**：在 SWE-Bench-Verified 数据集上，Satori-SWE-32B 模型在贪婪解码下准确率达 35.8%，优于所有小型模型；在 Best@50 设置下准确率达 41.6%，与参数超 100B 的模型（如 Llama3-SWE-RL-70B 的 Best@500）相当，样本需求仅为后者的 1/10。
*   **样本效率**：EvoScale 的自进化方法在少量样本（20-50 个）下表现优异，相比传统测试时扩展方法（如奖励模型选择或单元测试选择），性能提升更稳定，运行时间仅为单元测试选择的 1/6。
*   **实验设置**：实验对比了多种测试时扩展方法，分析了 SFT 和 RL 模型在不同迭代下的表现，通过消融研究（如奖励模型与字符串匹配对比、采样温度影响）验证方法有效性，设置全面且合理。
*   **局限性**：实验主要基于无代理（agentless）管道设置，未涉及与运行时环境交互的代理设置，可能限制方法在复杂场景中的适用性。

## Further Thoughts

EvoScale 的进化算法与语言模型结合的思路启发了我，是否可以将这种迭代优化策略应用于其他生成任务（如数学推理或文本创作），通过多轮选择和变异逐步逼近最优解？
此外，RL 训练模型自进化的方法提示我们，是否可以通过内化奖励机制减少对外部评分模型的依赖，特别是在资源受限的边缘设备上运行模型时？
最后，样本效率的关注让我思考，如何在其他领域（如图像生成）中设计类似的进化策略，以有限计算资源实现高质量输出？