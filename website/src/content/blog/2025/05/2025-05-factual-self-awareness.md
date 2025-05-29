---
title: "Factual Self-Awareness in Language Models: Representation, Robustness, and Scaling"
pubDatetime: 2025-05-27T16:24:02+00:00
slug: "2025-05-factual-self-awareness"
type: "arxiv"
id: "2505.21399"
score: 0.6614821846048627
author: "grok-3-latest"
authors: ["Hovhannes Tamoyan", "Subhabrata Dutta", "Iryna Gurevych"]
tags: ["LLM", "Self-Awareness", "Representation", "Scaling", "Hallucination"]
institution: ["Technical University of Darmstadt", "Ubiquitous Knowledge Processing Lab (UKP Lab)", "Hessian Center for AI (hessian.AI)"]
description: "本文揭示了大型语言模型在生成时通过内部线性表示编码事实自我意识信号，并通过实验验证其存在、鲁棒性和缩放行为，为解决幻觉问题提供了生成前干预的新视角。"
---

> **Summary:** 本文揭示了大型语言模型在生成时通过内部线性表示编码事实自我意识信号，并通过实验验证其存在、鲁棒性和缩放行为，为解决幻觉问题提供了生成前干预的新视角。 

> **Keywords:** LLM, Self-Awareness, Representation, Scaling, Hallucination

**Authors:** Hovhannes Tamoyan, Subhabrata Dutta, Iryna Gurevych

**Institution(s):** Technical University of Darmstadt, Ubiquitous Knowledge Processing Lab (UKP Lab), Hessian Center for AI (hessian.AI)


## Problem Background

大型语言模型（LLMs）在生成内容时常出现事实性错误（即幻觉问题），这是其广泛应用中的核心挑战。
本文关注模型是否在生成时就具备内在的‘事实自我意识’（Factual Self-Awareness），即能否在生成前区分出‘已知’（Known）和‘遗忘’（Forgotten）的事实关联，从而为解决幻觉问题提供比生成后事实核查更直接的切入点。

## Method

*   **核心思想:** 探索语言模型在生成时是否通过内部表示线性编码了事实自我意识信号，即模型能否在生成前就‘知道’自己是否能正确回忆某个事实。
*   **数据集构建:** 构造一个事实回忆数据集，涵盖足球运动员、电影、城市和歌曲四个类别，基于实体-关系-属性三元组设计输入模板，并根据模型输出概率分布（Logit Distribution）将事实关联标注为‘已知’或‘遗忘’（若目标 token 在 top-k 预测中占主导则为‘已知’，在 bottom-l 中占主导则为‘遗忘’）。
*   **线性探针（Linear Probe）:** 在 Transformer 的残差流（Residual Stream）中提取最终 token 的表示，训练线性分类器预测‘已知’或‘遗忘’标签，以检测内部是否线性编码了自我意识信号。
*   **稀疏自编码器（Sparse Autoencoder, SAE）:** 作为对比方法，无监督分解模型表示，验证线性探针结果的一致性。
*   **上下文扰动实验:** 通过修改输入格式（如添加引号、转为问句、加入少样本提示或无关语句），测试自我意识信号对上下文变化的鲁棒性。
*   **缩放实验:** 在不同模型规模（Gemma 2 2B/9B, Pythia 70M/1.4B/6.9B/12B）和训练阶段（Pythia 1.4B 多个检查点）上，分析自我意识信号的出现和演变规律。

## Experiment

*   **有效性:** 实验表明语言模型内部确实线性编码了事实自我意识信号。Gemma 2 2B 在测试集上表现最佳，准确率提升显著（∆=0.311），Pythia 12B 在其系列中表现最优（∆=0.120），但与 Gemma 2 仍有差距。
*   **鲁棒性:** 上下文扰动实验显示信号对表面变化（如引号）较为鲁棒，测试准确率仅小幅下降（如从 0.820 降至 0.802），但对语义结构变化（如问句形式）较敏感（准确率降至 0.756），提示信号与深层语义关联更紧密。
*   **缩放行为:** 模型规模越大，自我意识信号越强，但提升非线性（如 Gemma 2 9B 的 ∆=0.265 低于 2B）；训练早期信号快速出现并饱和，表明此能力是模型学习的基础特性。
*   **实验设置:** 数据集覆盖多个实体类型，输入模板优化避免伪相关性，实验设计考虑上下文扰动和模型规模，较为全面；但数据集覆盖有限，未探讨信号因果机制，存在一定局限。

## Further Thoughts

本文揭示了语言模型在生成前的内部表示中编码了事实正确性的‘元知识’，这启发我们可以在推理阶段设计干预机制，通过检测自我意识信号阻止错误生成，而无需外部知识库或后处理；此外，这种线性探针与稀疏自编码器的结合方法可扩展至检测其他内部信号（如情感或意图），为模型可解释性研究提供新工具。