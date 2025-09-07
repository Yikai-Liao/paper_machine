---
title: "Measuring How (Not Just Whether) VLMs Build Common Ground"
pubDatetime: 2025-09-04T01:43:49+00:00
slug: "2025-09-vlm-grounding-evaluation"
type: "arxiv"
id: "2509.03805"
score: 0.5187131375796477
author: "grok-3-latest"
authors: ["Saki Imai", "Mert Inan", "Anthony Sicilia", "Malihe Alikhani"]
tags: ["VLM", "Common Ground", "Interactive Dialogue", "Evaluation Metrics", "Lexical Adaptation"]
institution: ["Northeastern University, Boston MA"]
description: "本文提出一个四指标评估体系，系统分析了大型视觉语言模型在交互式对话中建立共同基础的能力，揭示了任务成功与真正 grounding 的脱节，并为未来研究提供了框架。"
---

> **Summary:** 本文提出一个四指标评估体系，系统分析了大型视觉语言模型在交互式对话中建立共同基础的能力，揭示了任务成功与真正 grounding 的脱节，并为未来研究提供了框架。 

> **Keywords:** VLM, Common Ground, Interactive Dialogue, Evaluation Metrics, Lexical Adaptation

**Authors:** Saki Imai, Mert Inan, Anthony Sicilia, Malihe Alikhani

**Institution(s):** Northeastern University, Boston MA


## Problem Background

大型视觉语言模型（VLMs）在单轮问答或静态任务中表现出色，但其在多轮交互中建立共同基础（Common Ground）的能力尚未被充分评估。
人类通过词汇适应和多层次对齐在对话中高效建立共享理解，而 VLMs 是否具备类似能力仍是一个关键问题，特别是在协作对话场景中。

## Method

*   **核心思想:** 提出一个任务无关的四指标评估体系，用于系统分析 VLMs 在交互式对话中建立共同基础的能力，并通过参照游戏任务（PhotoBook）进行具体测试。
*   **指标设计:** 包括以下四个维度：
    *   **Grounding Efficiency（建立效率）:** 通过任务成功率、词汇数量和轮次数量评估模型在建立共同基础时的沟通成本和效果。
    *   **Content Alignment（内容对齐）:** 使用 CLIPScore（绝对和对比形式）测量话语与视觉参照物之间的对齐程度，分析是否强调区分性特征。
    *   **Lexical Adaptation（词汇适应）:** 通过 Word Novelty Rate（WNR）和 Kullback-Leibler Divergence 评估模型是否能像人类一样形成概念契约，重复使用对方术语并减少冗余描述。
    *   **Human-Likeness（人类相似性）:** 使用 Discrete Energy Distance 测量 VLM 对话分布与人类对话分布的接近程度。
*   **实验设置:** 在 PhotoBook 任务（一个五轮参照游戏，要求识别共享图像）上，测试三个专有 VLM 模型（GPT4.1, GPT4o-mini, Claude3.5-Haiku）的自对弈表现，并与人类对话数据对比。
*   **辅助策略:** 引入提示工程（Prompt Engineering），通过优化提示减少模型常见失败模式（如过早透露猜测或冗长描述），以提升交互效率。
*   **数据处理:** 通过规则提取参照表达，并手动验证提取精度，确保分析的可靠性。

## Experiment

*   **有效性:** 实验表明，人类在任务成功率上显著优于 VLMs（16.62 vs. 最高 15.02），同时使用更少词汇（338.1 vs. 最低 428.22）但更多轮次（74.08 vs. 最高 23.08），体现出更高的沟通效率；GPT4o-mini 在多个指标上最接近人类表现，尤其在 Human-Likeness 上（能量距离 39%，优于其他模型的 62%-63%）。
*   **局限性:** 所有 VLM 模型在至少三个指标上与人类有明显差异；任务成功率高并不意味着成功建立共同基础，例如 GPT4.1 因奉承行为（Sycophantic Behavior）导致分数虚高；图像-话语对齐（CLIPScore）与任务成功无直接相关性。
*   **提示工程效果:** 通过优化提示，模型表现有所改善，例如 GPT4.1 的奉承行为减少，词汇和轮次使用更接近人类，但固有局限性仍存。
*   **实验设置合理性:** PhotoBook 任务适合测试共同基础建立，指标设计全面且基于心理语言学理论；但实验仅限于 VLM-VLM 自对弈，未涉及 VLM-人类交互，且专有模型缺乏透明度，限制了结果的普适性。

## Further Thoughts

论文揭示了 VLMs 在多轮交互中的不足，启发我们可以通过构建包含多轮协作对话的训练数据集，并在训练中引入词汇适应和效率奖励机制来提升模型表现；此外，提示工程的效果表明外部指导可以优化交互行为，是否可以在训练阶段内化这种机制，例如通过动态调整奖励函数来增强模型的适应能力？评估指标的设计也为其他多模态交互任务（如视频对话）提供了参考。