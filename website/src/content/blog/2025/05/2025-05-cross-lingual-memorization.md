---
title: "OWL: Probing Cross-Lingual Recall of Memorized Texts via World Literature"
pubDatetime: 2025-05-28T23:57:03+00:00
slug: "2025-05-cross-lingual-memorization"
type: "arxiv"
id: "2505.22945"
score: 0.8276778700146306
author: "grok-3-latest"
authors: ["Alisha Srivastava", "Emir Kaan Korukluoglu", "Minh Nhat Le", "Duyen Tran", "Chau Minh Pham", "Marzena Karpinska", "Mohit Iyyer"]
tags: ["LLM", "Cross-Lingual Transfer", "Memorization", "Multilingual Data", "Probing Tasks"]
institution: ["UMass Amherst", "University of Maryland, College Park", "Microsoft"]
description: "本文通过 OWL 数据集和三种探查任务，揭示了大型语言模型在多语言和跨语言环境中显著的记忆能力，证明其知识转移能力即使在未见翻译的低资源语言中依然存在。"
---

> **Summary:** 本文通过 OWL 数据集和三种探查任务，揭示了大型语言模型在多语言和跨语言环境中显著的记忆能力，证明其知识转移能力即使在未见翻译的低资源语言中依然存在。 

> **Keywords:** LLM, Cross-Lingual Transfer, Memorization, Multilingual Data, Probing Tasks

**Authors:** Alisha Srivastava, Emir Kaan Korukluoglu, Minh Nhat Le, Duyen Tran, Chau Minh Pham, Marzena Karpinska, Mohit Iyyer

**Institution(s):** UMass Amherst, University of Maryland, College Park, Microsoft


## Problem Background

大型语言模型（LLMs）在预训练过程中能够记忆并回忆训练数据中的文本内容，尤其在英语文本上已有广泛研究，但其在非英语语言中的记忆能力以及跨语言知识回忆能力（即在一种语言中学习的内容是否能在另一种语言中被回忆）仍未被充分探索。
这一问题至关重要，因为跨语言记忆能力不仅揭示了模型的多语言泛化能力，还涉及潜在的版权和数据隐私风险，例如模型可能无意中泄露受保护的内容。

## Method

*   **核心思想：** 通过构建多语言对齐数据集和设计多种探查任务，系统性地评估 LLMs 在不同语言和条件下的记忆能力，探索其跨语言知识转移能力。
*   **数据集构建：** 创建 OWL 数据集，包含 31,540 个对齐文学段落，来自 20 本英文书籍，涵盖英文原文、官方翻译（西班牙语、土耳其语、越南语）以及新翻译的六种低资源语言（塞索托语、约鲁巴语、迈蒂利语、马达加斯加语、茨瓦纳语、塔希提语），确保内容一致性并测试未见过语言的跨语言能力。
*   **探查任务：** 设计三种任务评估记忆能力：
    *   **直接探查（Direct Probing）：** 要求模型根据段落内容识别书籍标题和作者，测试对元数据的回忆能力。
    *   **名称填空（Name Cloze）：** 将段落中的人物名称用 [MASK] 替换，要求模型预测被掩盖的名称，测试对具体内容的精确回忆。
    *   **前缀探查（Prefix Probing）：** 给定段落前半部分，要求模型生成后半部分，测试对完整内容的再现能力。
*   **实验变量与扰动：** 引入扰动如打乱单词顺序（Shuffling）、掩盖人物名称（Masking）等，测试回忆鲁棒性；同时探索跨模态（文本 vs 音频）和模型量化（4-bit 和 8-bit）的影响。
*   **模型选择：** 测试多种模型，包括 GPT-4o、LLaMA 系列、Qwen 系列等，覆盖不同规模和开源/闭源模型，确保结果广泛适用性。

## Experiment

*   **有效性：** 实验表明 LLMs 具备显著的跨语言回忆能力，例如 GPT-4o 在直接探查任务中对英文段落准确率达 92.3%，对官方翻译为 83.4%，对未见过的新翻译低资源语言段落也能达到 69.4%；名称填空任务准确率较低（英文 38.6%，新翻译 6.3%），但仍高于随机猜测，显示跨语言记忆能力。
*   **对比分析：** 相较于英文，模型在官方翻译和未见翻译上的表现有所下降，但仍保持较高水平，尤其在直接探查任务中，表明模型可能通过共享多语言表示实现知识转移。
*   **鲁棒性测试：** 扰动（如打乱单词顺序）对直接探查准确率影响较小（英文下降约 3-8%，官方翻译下降约 6-7%），表明模型回忆不完全依赖语序，可能依赖关键词或语义模式。
*   **跨模态与量化：** 跨模态实验显示 GPT-4o-Audio 在音频输入下直接探查准确率仍高达 75.5%，表明记忆能力可跨模态转移；量化实验中，LLaMA-3.1-70B 在 8-bit 量化下性能下降显著（直接探查准确率下降高达 25%），而 4-bit 影响较小，与部分先前研究矛盾。
*   **实验设置合理性：** 实验设置全面，涵盖多种语言、任务、模型和扰动条件，确保结果多样性和代表性；但低资源语言翻译依赖 Microsoft Translator，可能引入翻译质量问题，影响结果下限估计；数据集基于畅销书籍，可能存在流行度偏差。

## Further Thoughts

本文启发了对 LLMs 跨语言知识表示本质的思考：模型是否形成了语言无关的语义表示？未来可通过可视化或解释性方法揭示其机制；此外，跨语言记忆可能加剧数据污染和隐私风险，是否可设计‘遗忘机制’或‘数据隔离’策略限制特定语言回忆能力？同时，这种能力在低资源语言中的应用潜力值得探索，例如用于教育或内容生成工具，通过高资源语言知识库弥补数据不足。