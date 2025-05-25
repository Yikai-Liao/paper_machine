---
title: "Understanding Prompt Tuning and In-Context Learning via Meta-Learning"
pubDatetime: 2025-05-22T17:58:53+00:00
slug: "2025-05-prompt-tuning-meta-learning"
type: "arxiv"
id: "2505.17010"
score: 0.5975548976147216
author: "grok-3-latest"
authors: ["Tim Genewein", "Kevin Wenliang Li", "Jordi Grau-Moya", "Anian Ruoss", "Laurent Orseau", "Marcus Hutter"]
tags: ["LLM", "In-Context Learning", "Prompt Tuning", "Meta-Learning", "Bayesian Inference"]
institution: ["Google DeepMind"]
description: "本文通过元学习和贝叶斯推断视角，揭示提示优化和上下文学习的理论基础与局限性，并通过实验验证软提示在适配目标任务中的优越性及无法处理多模态分布的限制。"
---

> **Summary:** 本文通过元学习和贝叶斯推断视角，揭示提示优化和上下文学习的理论基础与局限性，并通过实验验证软提示在适配目标任务中的优越性及无法处理多模态分布的限制。 

> **Keywords:** LLM, In-Context Learning, Prompt Tuning, Meta-Learning, Bayesian Inference

**Authors:** Tim Genewein, Kevin Wenliang Li, Jordi Grau-Moya, Anian Ruoss, Laurent Orseau, Marcus Hutter

**Institution(s):** Google DeepMind


## Problem Background

大型语言模型（LLMs）或前沿模型通过提示（Prompting）适配目标任务的能力令人印象深刻，但对提示优化（Prompt Tuning）和上下文学习（In-Context Learning）的概念性理解不足。
论文从元学习（Meta-Learning）和贝叶斯推断（Bayesian Inference）的视角出发，探讨在何种条件下提示优化能使预训练模型达到贝叶斯最优性能，以及其理论局限性（如无法处理多模态目标分布或全新任务），从而揭示提示优化与权重调整（Weight Tuning）的适用场景差异。

## Method

*   **理论框架：贝叶斯序列预测器**：将预训练模型视为通过元学习形成的贝叶斯序列预测器，其核心特性是快速上下文适配（In-Context Adaptation）。提示优化被理解为对预测器的条件化（Conditioning），通过前缀（Prefix）调整模型内部状态以适配目标任务。
*   **前缀调整方法**：研究了多种前缀优化技术，包括硬提示搜索（HardPT，使用真实 token 进行搜索）、单纯形前缀（SimplexPT，基于概率向量）、实数前缀（RealPT，基于实值向量）和软提示（SoftPT，基于嵌入空间的实值向量）。软提示允许输入超出 token 字母表的离分布向量（Off-Distribution Inputs），从而更灵活地操控模型内部激活状态。
*   **权重调整对比**：作为对比，测试了多种权重调整方法，包括嵌入层调整（EmbedWT，仅调整输入嵌入层参数）、解嵌入层调整（UnembedWT，仅调整输出解嵌入层参数）、嵌入与解嵌入联合调整（Un+EmbedWT）、全权重调整（FullWT，调整所有模型参数）和低秩适配（LoRAWT，通过低秩矩阵适配注意力层）。
*   **实验任务设计**：通过硬币翻转序列任务（Coin-Flip Sequences）控制数据分布，设计了三种任务分布：随机硬币（Random Coins，作为预训练分布）、单一硬币（Single Coin，作为符合预训练分布的目标任务）和双硬币混合（Two-Coin Mixture，作为多模态目标任务）。实验在预训练和未训练的 LSTM 和 Transformer 模型上进行，评估不同调整方法的预测性能（以累计遗憾即 Excess Log Loss 衡量）。

## Experiment

*   **有效性**：在单一硬币任务中，软提示（SoftPT）使预训练的 Transformer 和 LSTM 达到贝叶斯最优性能（累计遗憾接近零），显著优于其他前缀调整方法（如硬提示 HardPT），表明软提示能有效操控模型内部状态以适配目标任务。
*   **局限性**：在双硬币混合任务中，所有前缀调整方法均无法达到贝叶斯最优性能（SoftPT 遗憾值约为 2.74，而目标贝叶斯预测器为 0.69），验证了理论预测的提示优化局限性（无法处理多模态目标分布）；而权重调整方法（如 FullWT 和 LoRAWT）能够成功适配，遗憾值接近目标贝叶斯预测器。
*   **软提示机制优势**：软提示通过离分布输入操控模型内部激活，即使在未训练的 Transformer 上也能表现出较好的上下文学习能力（例如在 Random Coins 任务中，SoftPT 遗憾值接近贝叶斯最优），表明其可能挖掘了模型架构的固有算法能力。
*   **实验设置合理性**：实验通过简化的硬币翻转任务控制数据统计特性，便于与精确贝叶斯预测器对比，聚焦提示优化的基本机制；测试了不同前缀长度（L=6 和 L=25），确认结果不受长度限制影响；同时在预训练和未训练网络上进行实验，全面评估了方法适用性。
*   **性能权衡**：软提示在保持预训练模型性能的同时实现了高效适配，而权重调整虽效果更优，但会永久改变模型，可能导致预训练分布上的性能下降。

## Further Thoughts

贝叶斯视角为上下文学习和提示优化提供了一个统一理论框架，启发未来研究可将更多模型能力（如强化学习）纳入元学习框架；软提示通过离分布输入操控模型内部状态的能力表明，提示设计可探索更多非传统输入形式（如嵌入空间操作）以提升适配效果；提示优化与权重调整的权衡提示可结合两种方法（如先提示优化后权重微调）实现更高效的模型适配。