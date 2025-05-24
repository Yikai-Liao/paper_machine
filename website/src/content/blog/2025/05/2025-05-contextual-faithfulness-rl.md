---
title: "Teaching Large Language Models to Maintain Contextual Faithfulness via Synthetic Tasks and Reinforcement Learning"
pubDatetime: 2025-05-22T10:10:07+00:00
slug: "2025-05-contextual-faithfulness-rl"
type: "arxiv"
id: "2505.16483"
score: 0.771282035486954
author: "grok-3-latest"
authors: ["Shuzheng Si", "Haozhe Zhao", "Cheng Gao", "Yuzhuo Bai", "Zhitong Wang", "Bofei Gao", "Kangyang Luo", "Wenhao Li", "Yufei Huang", "Gang Chen", "Fanchao Qi", "Minjia Zhang", "Baobao Chang", "Maosong Sun"]
tags: ["LLM", "Faithfulness", "Synthetic Data", "Reinforcement Learning", "Contextual Generation"]
institution: ["Tsinghua University", "Peking University", "DeepLang AI", "University of Illinois Urbana-Champaign"]
description: "本文提出 CANOE 框架，通过合成短篇 QA 数据和 Dual-GRPO 强化学习方法，在无需人工标注的情况下显著提升了大型语言模型在短篇和长篇生成任务中的上下文忠实性。"
---

> **Summary:** 本文提出 CANOE 框架，通过合成短篇 QA 数据和 Dual-GRPO 强化学习方法，在无需人工标注的情况下显著提升了大型语言模型在短篇和长篇生成任务中的上下文忠实性。 

> **Keywords:** LLM, Faithfulness, Synthetic Data, Reinforcement Learning, Contextual Generation

**Authors:** Shuzheng Si, Haozhe Zhao, Cheng Gao, Yuzhuo Bai, Zhitong Wang, Bofei Gao, Kangyang Luo, Wenhao Li, Yufei Huang, Gang Chen, Fanchao Qi, Minjia Zhang, Baobao Chang, Maosong Sun

**Institution(s):** Tsinghua University, Peking University, DeepLang AI, University of Illinois Urbana-Champaign


## Problem Background

大型语言模型（LLMs）在基于上下文生成文本时常出现‘忠实性幻觉’（faithfulness hallucinations），即生成的响应偏离输入上下文，损害了模型在信息检索、问答和摘要等任务中的可信度。
论文指出三大挑战：(1) 单纯扩大模型参数规模无法有效提升忠实性，甚至可能因内部知识冲突加剧问题；(2) 现有方法难以在不同下游任务（如短篇和长篇生成）上持续提升忠实性；(3) 用于提升忠实性的训练数据难以扩展，尤其是长篇生成任务的数据通常需要人工标注，成本高且不可持续。
目标是开发一种无需人工标注的系统性方法，提升 LLMs 在短篇和长篇生成任务中的上下文忠实性。

## Method

*   **核心思想:** 通过合成易于验证的短篇问答（QA）数据，并利用强化学习（RL）方法同时优化短篇和长篇生成的上下文忠实性，避免人工标注的高成本。
*   **数据合成:** 从知识库（如 Wikidata）提取三元组（head-relation-tail），利用 GPT-4o 合成上下文和问题，确保数据正确性和易验证性；设计四种任务类型（直观上下文、需推理上下文、不一致上下文、反事实上下文）以增加数据复杂性和多样性，防止模型依赖内部知识。
*   **强化学习方法 Dual-GRPO:** 基于 GRPO（一种无需人工标注偏好数据的 RL 方法），提出 Dual-GRPO，通过系统提示要求模型生成推理过程、长篇回答和短篇回答；设计三种规则奖励：
    *   **Accuracy Reward:** 针对短篇回答，直接评估是否与真实答案匹配（通过精确匹配，EM），奖励值为 1（匹配）或 0（不匹配），确保短篇生成的忠实性。
    *   **Proxy Reward:** 针对长篇回答，间接评估忠实性，将长篇回答作为新上下文输入模型，检查是否能生成正确短篇答案，奖励值为 1（正确）或 0（错误），以此推断长篇回答是否忠实于上下文。
    *   **Format Reward:** 针对整体输出格式，检查是否符合预定义结构（如包含 <think>, <long_answer>, <short_answer> 标签），奖励值为 1（符合）或 0（不符合），增强输出一致性。
*   **训练过程:** 使用合成数据训练模型，生成多组候选答案，通过综合奖励计算优势值，指导策略更新，避免过优化短篇生成导致长篇能力下降。
*   **关键点:** 不修改模型架构，仅通过后训练（post-training）和规则奖励调整模型行为，兼顾短篇和长篇任务的忠实性提升。

## Experiment

*   **有效性:** CANOE 在11个下游任务上显著提升了忠实性，例如在 LLaMA-3-Instruct-8B 上平均 EM 分数提升了 22.6%，在 Qwen-2.5-7B 上提升了 19.0%，甚至超越了最先进的模型如 GPT-4o 和 OpenAI o1。
*   **全面性:** 实验覆盖短篇问答（ConFiQA, FaithEval 等）、长篇摘要（XSum）、文本简化（WikiLarge）和长篇问答（CLAPNQ）等多种任务，数据集包括反事实上下文、真实世界 RAG 场景等，验证了方法的普适性。
*   **合理性:** 消融研究表明 Dual-GRPO 和数据合成策略的结合是关键，单独使用 GRPO 会导致长篇生成能力下降；不同类型合成数据（如反事实上下文）的加入增强了模型对复杂上下文的适应能力。
*   **额外收益:** CANOE 不仅提升忠实性，还改善了长篇生成的响应质量（QualityScore 提升明显），增强了推理能力（如在 ConFiQA 的多跳推理任务中表现优异），并缓解了模型对错误输出的过度自信（perplexity 分析显示较低信心）。
*   **开销与局限:** 主要开销在于数据合成和 RL 训练，但无需人工标注降低了成本；实验未直接合成长篇数据，未来可进一步探索多轮对话数据合成和长篇直接优化。

## Further Thoughts

CANOE 通过短篇 QA 数据的规则验证能力间接提升长篇生成的忠实性，这一思路启发我们可以在其他难以直接评估的生成任务（如多轮对话或创意写作）中尝试类似策略，利用易验证的子任务数据设计代理奖励；此外，四种合成数据类型（尤其是反事实上下文）迫使模型依赖上下文而非内部知识，未来可探索更多复杂数据类型或结合少量人工标注数据，进一步提升模型对知识冲突的鲁棒性；Dual-GRPO 的规则奖励设计也提示我们可以尝试更多创新奖励机制，如基于语义一致性的奖励，来优化 RL 训练效果。