---
title: "Teaching Large Language Models to Maintain Contextual Faithfulness via Synthetic Tasks and Reinforcement Learning"
pubDatetime: 2025-05-22T10:10:07+00:00
slug: "2025-05-contextual-faithfulness-canoe"
type: "arxiv"
id: "2505.16483"
score: 0.771282035486954
author: "grok-3-latest"
authors: ["Shuzheng Si", "Haozhe Zhao", "Cheng Gao", "Yuzhuo Bai", "Zhitong Wang", "Bofei Gao", "Kangyang Luo", "Wenhao Li", "Yufei Huang", "Gang Chen", "Fanchao Qi", "Minjia Zhang", "Baobao Chang", "Maosong Sun"]
tags: ["LLM", "Contextual Faithfulness", "Reinforcement Learning", "Synthetic Data", "Question Answering"]
institution: ["Tsinghua University", "Peking University", "University of Illinois Urbana-Champaign", "DeepLang AI"]
description: "本文提出 CANOE 框架，通过合成短文本 QA 数据和 Dual-GRPO 强化学习方法，在无需人工标注的情况下显著提升了大型语言模型在短文本和长文本生成任务中的上下文忠实性。"
---

> **Summary:** 本文提出 CANOE 框架，通过合成短文本 QA 数据和 Dual-GRPO 强化学习方法，在无需人工标注的情况下显著提升了大型语言模型在短文本和长文本生成任务中的上下文忠实性。 

> **Keywords:** LLM, Contextual Faithfulness, Reinforcement Learning, Synthetic Data, Question Answering

**Authors:** Shuzheng Si, Haozhe Zhao, Cheng Gao, Yuzhuo Bai, Zhitong Wang, Bofei Gao, Kangyang Luo, Wenhao Li, Yufei Huang, Gang Chen, Fanchao Qi, Minjia Zhang, Baobao Chang, Maosong Sun

**Institution(s):** Tsinghua University, Peking University, University of Illinois Urbana-Champaign, DeepLang AI


## Problem Background

大型语言模型（LLMs）在基于上下文生成文本时，常常会产生不忠实于输入上下文的内容（即忠实性幻觉），这在信息检索、问答和摘要等任务中损害了模型的可信度。
论文指出三大挑战：(1) 单纯增加模型参数规模无法有效提升忠实性，甚至可能因内部知识冲突加剧问题；(2) 现有方法难以在不同下游任务（如短文本和长文本生成）上持续提升忠实性；(3) 用于提升忠实性的训练数据难以扩展，尤其长文本任务中人工标注成本高昂。因此，研究目标是开发一种无需人工标注的系统化后训练方法，提升 LLMs 在多种任务中的上下文忠实性。

## Method

*   **核心思想:** 通过合成高质量短文本问答（QA）数据和强化学习（RL），在无需人工标注的情况下提升 LLMs 的上下文忠实性，同时优化短文本和长文本生成任务。
*   **数据合成:** 从 Wikidata 提取知识三元组（head-relation-tail），利用 GPT-4o 合成短文本 QA 数据，并设计四种任务类型以增加复杂性和多样性：
    *   **直接上下文（Straightforward Context）**：上下文直接包含答案，要求模型准确提取信息。
    *   **推理需求上下文（Reasoning-required Context）**：包含多跳推理路径，要求模型进行复杂推理。
    *   **不一致上下文（Inconsistent Context）**：包含多个随机排序的上下文，模拟噪声场景，要求模型识别相关信息。
    *   **反事实上下文（Counterfactual Context）**：包含与常识矛盾的陈述，防止模型依赖内部知识。
*   **强化学习方法 Dual-GRPO:** 基于 GRPO 提出 Dual-GRPO，通过规则奖励同时优化短文本和长文本生成：
    *   **生成流程**：模型先输出推理过程，再生成详细的长文本答案，最后是简洁的短文本答案。
    *   **奖励设计**：包括三种规则奖励：
        *   **准确性奖励（Accuracy Reward）**：评估短文本答案与真实答案的匹配度（使用精确匹配 EM）。
        *   **代理奖励（Proxy Reward）**：将长文本答案作为新上下文输入模型，评估是否能引导生成正确短文本答案，间接衡量长文本忠实性。
        *   **格式奖励（Format Reward）**：确保输出符合预定义结构（如使用特定标签），提升一致性。
    *   **优势**：无需人工标注偏好数据，避免过优化短文本任务，平衡两种生成任务的性能。
*   **关键点:** 方法不依赖人工标注，通过规则奖励和合成数据实现高效训练，同时避免了传统 RL 中奖励模型训练的复杂性。

## Experiment

*   **有效性:** CANOE 在 11 个下游任务上显著提升了忠实性，例如在 LLaMA-3-Instruct-8B 上平均 EM 分数提升了 22.6%，在 Qwen-2.5-7B 上提升了 19.0%，整体表现超越最先进的模型如 GPT-4o 和 OpenAI o1。
*   **全面性与合理性:** 实验覆盖短文本 QA（ConFiQA、FiQA 等）、长文本生成（XSum、WikiLarge 等）和 RAG 场景（FollowRAG），数据集多样且任务类型广泛，体现了方法的适用性；对比基线（SFT、Context-DPO、SCOPE）显示 CANOE 避免了过优化某一任务，同时提升了长文本生成质量（如 QualityScore）。
*   **消融实验:** 验证了 Dual-GRPO 和数据合成策略的有效性，例如去除 Dual-GRPO 会导致短文本过优化，长文本生成质量下降；多样性任务设计（如反事实上下文）有助于防止模型依赖内部知识。
*   **额外优势:** CANOE 还缓解了过自信偏差（perplexity 分析），提升了推理能力（ConFiQA 多跳推理任务），并在中文数据集上展现了多语言迁移能力。
*   **局限性:** 论文未详细讨论 RL 训练和数据合成的计算开销，可能在资源需求上存在实际应用瓶颈。

## Further Thoughts

CANOE 的代理奖励机制为评估难以量化的任务（如长文本忠实性）提供了创新思路，未来可扩展到其他领域，如情感一致性或风格一致性，通过间接指标设计奖励；此外，合成数据中反事实上下文的设计启发我们可以通过构造冲突场景，强制模型依赖上下文而非内部知识，这种方法可应用于解决知识冲突或偏见问题；最后，规则奖励的灵活性表明 RL 可以在无需人工标注的情况下结合更多自监督信号，未来可探索与其他无监督学习方法的结合。