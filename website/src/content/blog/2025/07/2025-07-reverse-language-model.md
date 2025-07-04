---
title: "LEDOM: An Open and Fundamental Reverse Language Model"
pubDatetime: 2025-07-02T03:52:00+00:00
slug: "2025-07-reverse-language-model"
type: "arxiv"
id: "2507.01335"
score: 0.6124783601353607
author: "grok-3-latest"
authors: ["Xunjian Yin", "Sitao Cheng", "Yuxi Xie", "Xinyu Hu", "Li Lin", "Xinyi Wang", "Liangming Pan", "William Yang Wang", "Xiaojun Wan"]
tags: ["LLM", "Reverse Modeling", "Posterior Evaluation", "Reasoning", "Sampling"]
institution: ["Peking University", "University of California, Santa Barbara", "University of Arizona", "National University of Singapore"]
description: "本文提出 L EDOM，首个大规模反向训练自回归语言模型，并通过 Reverse Reward 机制利用其后验评估能力显著提升前向模型在数学推理任务中的生成质量。"
---

> **Summary:** 本文提出 L EDOM，首个大规模反向训练自回归语言模型，并通过 Reverse Reward 机制利用其后验评估能力显著提升前向模型在数学推理任务中的生成质量。 

> **Keywords:** LLM, Reverse Modeling, Posterior Evaluation, Reasoning, Sampling

**Authors:** Xunjian Yin, Sitao Cheng, Yuxi Xie, Xinyu Hu, Li Lin, Xinyi Wang, Liangming Pan, William Yang Wang, Xiaojun Wan

**Institution(s):** Peking University, University of California, Santa Barbara, University of Arizona, National University of Singapore


## Problem Background

传统语言模型（Forward Language Models, FLMs）采用自左向右的训练方式，虽然在实践中成功，但可能限制了对语言中某些依赖关系和推理模式的捕捉能力。
作者质疑这种方向性假设，探索反向训练的语言模型（Reverse Language Model, RLM）是否能作为基础模型适用于通用任务，并解决前向模型在溯因推理或后验评估任务中的不足，揭示不同的推理路径或‘世界模型’。

## Method

*   **反向自回归预训练**：提出 L EDOM，一个纯粹反向训练的自回归语言模型，训练于 4350 亿 token，规模为 2B 和 7B 参数，与传统 FLM 不同，L EDOM 预测前一个 token（从右向左建模），依赖‘未来’上下文预测‘过去’，以捕捉不同的语义依赖和推理模式。
*   **训练设置与对比**：L EDOM 使用与 FLM 相同的 Transformer 解码器架构、tokenizer 和训练数据（包括通用文本、数学数据和代码数据，总计 4350 亿 token），确保公平对比，训练采用 AdamW 优化器和余弦学习率调度。
*   **反向奖励（Reverse Reward）机制**：提出一种创新应用，利用 L EDOM 的后验评估能力，通过计算输入序列在给定 FLM 生成输出下的反向概率，对 FLM 生成的候选输出进行重新排序（Best-of-N）或引导多步推理（基于束搜索的逐步解码），结合前向和反向概率（通过可调参数 λ 平衡）计算综合奖励分数，以提升生成质量。
*   **关键特点**：反向训练不改变模型架构，仅调整训练目标，Reverse Reward 机制则在推理阶段引入反向模型的评估信号，与前向模型形成互补。

## Experiment

*   **基础性能**：在多个基准测试（如 GSM8K, HellaSwag, HumanEval 等）上，L EDOM 表现与同规模 FLM 相比有差距，尤其在代码生成（HumanEval Pass@1 仅 1.22% vs FLM-7B 的 13.41%）和世界知识任务（如 TriviaQA）上较弱，但在常识推理任务（如 BoolQ）上接近 FLM，显示一定通用性。
*   **Reverse Reward 效果**：在数学推理任务（如 GSM8K, MATH-500）上，Reverse Reward 显著提升 FLM 性能，例如 QwenMath 在 GSM8K 上从 95.6%（贪婪解码）提升至 96.1%，在 MATH-500 上从 78.0% 提升至 80.8%，DeepSeekMath 在 AIME 2024 上从 10.0% 提升至 13.3%。
*   **实验设置合理性**：实验覆盖多种模型（DeepSeekMath, QwenMath, OpenMath2）和解码策略（贪婪解码、随机 Best-of-N、Reverse Reward 的 Best-of-N 和 Beam Search），测试了不同采样规模（N=1 到 64），设置全面，数据可信。
*   **局限与分析**：L EDOM 训练收敛较慢，损失较高，可能限制独立性能；Reverse Reward 在多步推理任务中效果更显著，但计算成本随采样规模增加而上升，需权衡。

## Further Thoughts

反向训练揭示了语言模型方向性偏见，启发是否可以通过混合前向和反向训练构建更对称的语言理解系统；Reverse Reward 的后验评估机制提示不同范式模型（前向 vs 反向）的协作潜力，是否可扩展至其他异构模型（如编码器模型）与自回归模型的结合；反向模型在溯因推理和逆向关系（如‘reversal curse’）上的优势，是否可应用于知识图谱补全或因果推理任务？