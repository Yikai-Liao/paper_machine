---
title: "Text2Grad: Reinforcement Learning from Natural Language Feedback"
pubDatetime: 2025-05-28T13:23:49+00:00
slug: "2025-05-text2grad-feedback"
type: "arxiv"
id: "2505.22338"
score: 0.6083851182783925
author: "grok-3-latest"
authors: ["Hanyang Wang", "Lu Wang", "Chaoyun Zhang", "Tianjun Mao", "Si Qin", "Qingwei Lin", "Saravan Rajmohan", "Dongmei Zhang"]
tags: ["LLM", "Reinforcement Learning", "Natural Language Feedback", "Policy Optimization", "Token-Level Reward"]
institution: ["University of Chicago", "Microsoft", "Fudan University"]
description: "TEXT2GRAD 提出了一种将自然语言反馈转化为 token 级别梯度信号的框架，通过细粒度策略优化显著提升语言模型性能和学习效率。"
---

> **Summary:** TEXT2GRAD 提出了一种将自然语言反馈转化为 token 级别梯度信号的框架，通过细粒度策略优化显著提升语言模型性能和学习效率。 

> **Keywords:** LLM, Reinforcement Learning, Natural Language Feedback, Policy Optimization, Token-Level Reward

**Authors:** Hanyang Wang, Lu Wang, Chaoyun Zhang, Tianjun Mao, Si Qin, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang

**Institution(s):** University of Chicago, Microsoft, Fudan University


## Problem Background

传统强化学习从人类反馈（RLHF）方法将复杂的自然语言反馈简化为单一标量奖励，导致信息丢失，难以精确归因具体错误位置，学习效率低且缺乏可解释性；同时，现有保留自然语言反馈的方法未将其融入模型参数更新，导致反馈无法被模型内化，需反复纠正相同错误。

## Method

* **核心思想**：将自由形式的自然语言反馈转化为 token 级别的梯度信号（称为自然语言梯度，NL-Gradient），直接用于模型策略优化，实现细粒度、针对性的参数更新。
* **具体实现**：
  * **双重反馈标注（Dual-Feedback Annotation）**：利用强大语言模型（如 GPT-4o）生成自然语言批评和结构化的 span 级别奖励映射，将反馈内容与输出中的具体 token 段对齐，并通过多步提示策略（如链式思维，Chain-of-Thought）增强标注质量；对于每个 token，分配伪奖励值（+1 表示正面，-1 表示负面，0 表示中性）。
  * **奖励模型训练（Reward Model Training）**：训练一个双头奖励模型，同时生成自然语言批评和 span 级别的奖励分布；该模型以条件语言生成的方式工作，通过最大似然估计优化，确保反馈既具有可解释性又能转化为数值信号。
  * **自然语言梯度策略优化（NL-Gradient Policy Optimization）**：基于 token 级别的伪奖励信号，计算细粒度的优势估计（advantage estimation），并通过改进的 PPO（Proximal Policy Optimization）算法更新模型策略；具体而言，利用 NL-Gradient 定义为每个 token 的梯度贡献，结合重要性比率和熵正则化，确保更新稳定且针对性强。
* **关键创新**：通过 token 级别的奖励信号，解决传统 RLHF 中标量奖励的信用分配问题，同时将自然语言反馈直接融入训练循环，而非仅用于推理时调整。

## Experiment

* **有效性**：TEXT2GRAD 在摘要生成（SLF5K）、代码生成（KodCode）和开放域问答（UltraFeedback）任务中均显著优于标量奖励 RLHF（如 PPO）和基于提示的反思基线；例如，在 SLF5K 上，ROUGE-L 得分提升 3.3 个百分点，BLEU 提升 25.3%；在 KodCode 上，pass@1 准确率平均提升 3.6-5.8 个百分点。
* **收敛速度**：TEXT2GRAD 展现出更高的样本效率和更快收敛速度，例如在 SLF5K 上，仅需 75% 的训练步骤即可达到最优性能，而 PPO 需要 97%。
* **实验设置合理性**：实验覆盖多种任务类型，数据集规模较大（如 UltraFeedback 包含 64K 提示），评价指标多样（ROUGE, BLEU, BERTScore, pass@1 等），且与人类标注的对齐率较高（82%-94%），表明实验设计全面，结果可信。
* **局限性**：奖励模型质量对性能有较大影响，token 级别奖励计算引入额外开销，可能影响大规模部署。

## Further Thoughts

TEXT2GRAD 将自然语言反馈转化为梯度的思想非常具有启发性，未来是否可以将其扩展到多模态任务（如图像描述生成），通过对齐视觉区域与文本反馈实现细粒度优化？此外，是否可以通过设计更高效的奖励模型或动态调整 token 级别奖励权重，减少计算开销并提升上下文相关性？