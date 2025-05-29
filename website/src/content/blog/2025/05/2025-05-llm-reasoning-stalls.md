---
title: "Large Language Models' Reasoning Stalls: An Investigation into the Capabilities of Frontier Models"
pubDatetime: 2025-05-26T08:34:07+00:00
slug: "2025-05-llm-reasoning-stalls"
type: "arxiv"
id: "2505.19676"
score: 0.7944029697401623
author: "grok-3-latest"
authors: ["Lachlan McGinness", "Peter Baumgartner"]
tags: ["LLM", "Reasoning", "Prompt Engineering", "Automated Theorem Proving", "In-Context Learning"]
institution: ["Australian National University", "CSIRO", "Data61"]
description: "本文通过 PRONTOQA 基准测试揭示了大型语言模型（LLMs）推理能力在2023年12月至2024年8月间的停滞，并证明提示工程在近期改进中的关键作用。"
---

> **Summary:** 本文通过 PRONTOQA 基准测试揭示了大型语言模型（LLMs）推理能力在2023年12月至2024年8月间的停滞，并证明提示工程在近期改进中的关键作用。 

> **Keywords:** LLM, Reasoning, Prompt Engineering, Automated Theorem Proving, In-Context Learning

**Authors:** Lachlan McGinness, Peter Baumgartner

**Institution(s):** Australian National University, CSIRO, Data61


## Problem Background

大型语言模型（LLMs）在逻辑推理能力上的表现备受关注，但其推理能力是否真正随着时间和模型更新而提升仍存疑问。
论文聚焦于评估 LLMs 是否能通过上下文学习掌握自动化定理证明（ATP）策略，并探讨其推理过程的正确性与最终答案准确性之间的关系，解决当前评估多集中于结果而非过程的局限性。

## Method

*   **研究设计:** 本研究采用经验性方法，通过 PRONTOQA 基准测试数据集评估2023年12月和2024年8月的前沿 LLMs（如 GPT-3.5 Turbo、GPT-4、GPT-4o、Gemini-Pro、Claude 3 Opus、Llama3.1 405B）的逻辑推理能力。
*   **推理策略:** 通过上下文学习（in-context learning），教导模型使用三种 ATP 推理策略：
    *   自底向上（Bottom-Up，forward chaining）：从基本事实和规则开始，逐步推导出结论，直至回答查询。
    *   自顶向下（Top-Down，backward chaining）：从查询开始，使用规则递归推导出子目标，直至子目标可被事实证明或证伪。
    *   魔法集变换（Magic Set Transformation）：首先通过自顶向下探索确定与查询相关的规则和事实集，缩小搜索空间后进行自底向上推理。
*   **提示技术:** 设计了多种提示条件，包括普通提示（Normal）、零样本思维链（Zero-shot CoT）、单样本思维链（One-shot CoT）以及基于 ATP 策略的提示，以对比不同提示对推理表现的影响。
*   **评估方式:** 使用 SpaCy 工具（一个开源自然语言处理工具）自动化解析模型输出，评估推理过程是否包含所有必要步骤，以及是否忠实于指定的推理策略；同时记录模型准确性和令牌计数（token count）以分析行为模式。
*   **实验条件:** 测试涵盖不同推理步骤复杂度（1-hop, 2-hop, 3-hop）和最难的‘False Ontology with Distractors’条件，确保评估的全面性。

## Experiment

*   **推理能力停滞:** 实验结果表明，LLMs 的推理能力在2023年12月至2024年8月间几乎没有显著提升；GPT-4 在多个条件下表现优于2024年新模型（如 GPT-4o 和 Claude 3 Opus）。
*   **提示工程影响:** 通过令牌计数发现，2024年模型在普通提示下生成更多令牌，表明其可能被训练或内置隐藏提示以自动使用思维链（CoT）推理，但这种改进仅在零样本条件下明显，其他条件下的表现与 GPT-4 相近。
*   **策略表现:** 模型在使用自底向上（Bottom-Up）推理策略时准确性和过程忠实度最高，而魔法集变换（Magic Set）因复杂度较高，表现较差。
*   **相关性分析:** 准确性与推理过程正确性之间存在弱正相关，表明正确推理不一定导致正确答案。
*   **实验设置合理性:** 实验设计较为全面，涵盖多种提示技术和推理复杂度，并使用 PRONTOQA 数据集避免数据污染问题；但由于通过不同 API 访问模型，未评估计算成本和延迟，存在一定局限性。

## Further Thoughts

论文揭示了 LLMs 推理能力可能已接近瓶颈，单纯增加模型规模或训练数据不足以突破，启发我们探索神经符号方法（neuro-symbolic approaches），结合 LLMs 的语言理解与 ATP 的严谨逻辑推理，以提升推理可靠性；此外，隐藏提示对模型表现的影响提示我们需设计更透明的模型架构，避免限制新提示技术的研究；最后，强调推理过程评估的重要性，为未来设计更全面的 LLMs 基准提供了新思路。