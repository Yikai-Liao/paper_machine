---
title: "Counterfactual Simulatability of LLM Explanations for Generation Tasks"
pubDatetime: 2025-05-27T20:29:50+00:00
slug: "2025-05-counterfactual-simulatability-generation"
type: "arxiv"
id: "2505.21740"
score: 0.49605544308536886
author: "grok-3-latest"
authors: ["Marvin Limpijankit", "Yanda Chen", "Melanie Subbiah", "Nicholas Deas", "Kathleen McKeown"]
tags: ["LLM", "Explanation", "Generation Task", "Evaluation Framework", "Mental Model"]
institution: ["Columbia University"]
description: "本文提出一个针对生成任务的反事实可模拟性评估框架，揭示了大型语言模型解释在技能型任务（如新闻摘要）中表现优于知识型任务（如医疗建议）的差异。"
---

> **Summary:** 本文提出一个针对生成任务的反事实可模拟性评估框架，揭示了大型语言模型解释在技能型任务（如新闻摘要）中表现优于知识型任务（如医疗建议）的差异。 

> **Keywords:** LLM, Explanation, Generation Task, Evaluation Framework, Mental Model

**Authors:** Marvin Limpijankit, Yanda Chen, Melanie Subbiah, Nicholas Deas, Kathleen McKeown

**Institution(s):** Columbia University


## Problem Background

大型语言模型（LLMs）在生成任务中的输出空间庞大且行为难以预测，尤其在高风险领域（如医疗），用户需要通过模型解释来理解其行为以降低风险。
现有研究主要关注分类任务的解释评估，而生成任务由于输出多样性，解释的可靠性和实用性评估仍是一个空白。
论文提出‘反事实可模拟性’（Counterfactual Simulatability）作为评估标准，旨在解决 LLMs 解释是否能帮助用户形成与模型行为一致的心理模型（Mental Model）的问题。

## Method

*   **框架目标:** 提出一个评估 LLMs 在生成任务中解释能力的框架，衡量解释是否能帮助用户准确推断模型在反事实输入上的输出。
*   **具体步骤:**
    *   使用 LLM（如 GPT-4 Turbo）生成解释和相关反事实输入（Counterfactuals）。
    *   将解释分解为原子单元（Atomic Units），作为用户心理模型的代理，用于简化生成任务中复杂的输出空间评估。
    *   定义并测量反事实可模拟性（Simulatability，是否能基于解释推断输出）、泛化性（Generality，解释适用的反事实输入多样性）和精确性（Precision，用户推断与实际输出的匹配度）。
    *   通过人类标注和自动化评估（使用 LLM 辅助标注）验证框架效果。
*   **任务应用:** 在新闻摘要（CNN/DM 数据集，技能型任务）和医疗建议（Taiwan e-Hospital 数据集，知识型任务）上测试框架。
*   **解释生成方式:** 采用两种提示策略生成解释：思维链（Chain-of-Thought，强调推理过程）和后验解释（Post-hoc，事后总结决策依据），以对比不同策略对解释效果的影响。
*   **任务差异处理:** 针对技能型任务，解释设计为高层次、抽象的描述以提高泛化性；针对知识型任务，解释聚焦于具体输入细节和建议以提高精确性。

## Experiment

*   **效果对比:** 实验结果表明，LLMs 的解释在新闻摘要任务中表现较好，泛化性和精确性较高（例如，Chain-of-Thought 解释的 Precision 达到 0.81-0.93，Generality 约为 0.52-0.67），用户能较准确预测模型行为；而在医疗建议任务中，解释效果较差（Precision 仅为 0.46-0.66，Generality 约为 0.19-0.26），表明模型在知识型任务中难以提供可靠解释。
*   **评估方式:** 实验结合人类评估和自动化评估（使用 GPT-4 Turbo 标注），自动化评估与人类标注一致性较高（Cohen’s Kappa 约为 0.61-0.65），但医疗建议任务中解释解析和反事实生成错误较多。
*   **设置合理性:** 实验设置较为全面，涵盖了多个模型（GPT-4, Claude 3.7 Sonnet, Llama 3）、不同任务类型和解释方法，数据量和反事实生成数量（3-10 个）设计合理，但框架在知识型任务中的局限性（如反事实生成困难）表明其适用性有待改进。
*   **结论支持:** 实验数据支持了论文假设，即 LLMs 解释在技能型任务中更有效，而在知识型任务中存在显著挑战。

## Further Thoughts

论文揭示了任务类型对 LLMs 解释效果的影响，技能型任务（如新闻摘要）由于依赖输入结构，解释更易泛化，而知识型任务（如医疗建议）依赖模型内部知识，解释泛化性差但更具体。这启发我们可以在设计解释系统时根据任务类型调整策略，例如为知识型任务引入领域知识增强解释的可靠性。此外，原子单元作为心理模型代理的思路值得进一步探索，或许可以通过更精细的语义分析或用户反馈来优化原子单元的提取和评估方式。