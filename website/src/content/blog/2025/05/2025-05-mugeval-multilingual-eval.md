---
title: "MUG-Eval: A Proxy Evaluation Framework for Multilingual Generation Capabilities in Any Language"
pubDatetime: 2025-05-20T14:14:00+00:00
slug: "2025-05-mugeval-multilingual-eval"
type: "arxiv"
id: "2505.14395"
score: 0.5385678651917176
author: "grok-3-latest"
authors: ["Seyoung Song", "Seogyeong Jeong", "Eunsu Kim", "Jiho Jin", "Dongkwan Kim", "Jamin Shin", "Alice Oh"]
tags: ["LLM", "Multilingual Evaluation", "Generation Capability", "Conversational Task", "Language Agnostic"]
institution: ["KAIST", "Trillion Labs"]
description: "M U G-Eval 提出了一种语言无关的评估框架，通过对话任务成功率作为代理指标，高效评估大型语言模型的多语言生成能力，解决了资源依赖和可扩展性问题。"
---

> **Summary:** M U G-Eval 提出了一种语言无关的评估框架，通过对话任务成功率作为代理指标，高效评估大型语言模型的多语言生成能力，解决了资源依赖和可扩展性问题。 

> **Keywords:** LLM, Multilingual Evaluation, Generation Capability, Conversational Task, Language Agnostic

**Authors:** Seyoung Song, Seogyeong Jeong, Eunsu Kim, Jiho Jin, Dongkwan Kim, Jamin Shin, Alice Oh

**Institution(s):** KAIST, Trillion Labs


## Problem Background

大型语言模型（LLMs）在多语言生成能力评估上面临显著挑战，尤其是在低资源语言中，由于缺乏专门的自然语言处理工具、参考语料库和基准数据集，传统评估方法（如 BLEU、ROUGE）依赖人工标注数据，难以扩展到多种语言；同时，近期使用 LLMs-as-judges 的方法在低资源语言中的可靠性存疑，因此需要一个通用的、资源高效的评估框架。

## Method

*   **核心思想**：提出 M U G-Eval 框架，通过将现有基准转化为对话任务，间接评估大型语言模型的多语言生成能力，以任务成功率作为生成能力的代理指标，避免直接评估文本质量。
*   **任务设计**：设计三种对话任务，要求两个模型实例在目标语言中进行信息交互：
    *   **Easy Twenty Questions**：基于字词猜测游戏，评估推理和策略性提问能力，任务中一个模型（提问者）通过最多 20 个是/否问题猜测隐藏词，另一个模型（回答者）仅回复“是”、“否”或“可能”。
    *   **MCQ Conversation**：基于阅读理解数据集 Belebele 改编为对话任务，评估多轮指令跟随能力，提问者通过最多 10 个问题确定正确答案，回答者根据隐藏段落回复。
    *   **Code Reconstruction**：基于代码生成任务，评估编程能力，一个模型（描述者）用目标语言描述代码，另一个模型（重建者）根据描述重建代码并通过单元测试。
*   **评估机制**：使用算法方法（如字符串匹配、代码测试）计算任务完成率，避免依赖语言特定工具或 LLMs-as-judges，保持语言无关性。
*   **实现细节**：任务提示明确要求使用目标语言，回答格式受限（如仅允许“是/否/可能”），并使用 GlotLID 工具确保语言一致性。

## Experiment

*   **有效性**：M U G-Eval 在 30 种语言（高、中、低资源各 10 种）上评估了 8 个模型（包括 Llama、Qwen、GPT-4o 等），任务成功率与现有基准（如 Belebele、Global-MMLU）高度相关（Pearson 相关系数 r=0.75），表明评估结果可靠。
*   **区分能力**：三种任务表现出不同的难度和饱和效应，Code Reconstruction 较简单（准确率接近 0.9），MCQ Conversation 次之（约 0.8），Easy Twenty Questions 最难（低资源语言中接近 0），增强了框架对模型和语言的区分能力。
*   **跨语言表现**：高资源和中资源语言性能差距较小，但中资源与低资源语言差距显著，反映模型在低资源语言上的局限性；实验覆盖语言家族和书写系统多样性，设置全面合理。
*   **成本与效率**：框架不依赖人工标注或语言特定工具，资源高效，当前支持 2102 种语言，具有极高的可扩展性。
*   **局限性**：仅基于任务完成率，无法评估生成文本的风格或文化适应性，可能对某些应用场景不够全面。

## Further Thoughts

M U G-Eval 的代理评估思路启发我们可以通过任务成功率间接评估其他能力，如情感表达或上下文适应，而不仅仅是语言生成；自沟通范式的对话任务设计可以扩展到模拟更复杂的人类交互场景，如多方协作或辩论；此外，语言无关性设计为跨模态评估（如图像或音频生成）提供了借鉴，或许可以通过引入文化特定任务进一步测试模型的跨文化适应能力。