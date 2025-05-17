---
title: "Pre-Act: Multi-Step Planning and Reasoning Improves Acting in LLM Agents"
pubDatetime: 2025-05-15T05:17:47+00:00
slug: "2025-05-pre-act-planning"
type: "arxiv"
id: "2505.09970"
score: 0.6805857904114396
author: "grok-3-latest"
authors: ["Mrinal Rawat", "Ambuje Gupta", "Rushil Goomer", "Alessandro Di Bari", "Neha Gupta", "Roberto Pieraccini"]
tags: ["LLM", "Reasoning", "Multi-Step Planning", "Fine-Tuning", "Agentic Systems"]
institution: ["Uniphore"]
description: "本文提出 Pre-Act 方法，通过多步骤规划和详细推理显著提升 LLM 代理在复杂任务中的表现，并通过微调策略使小模型媲美大模型，同时设计了两级评估框架全面衡量代理性能。"
---

> **Summary:** 本文提出 Pre-Act 方法，通过多步骤规划和详细推理显著提升 LLM 代理在复杂任务中的表现，并通过微调策略使小模型媲美大模型，同时设计了两级评估框架全面衡量代理性能。 

> **Keywords:** LLM, Reasoning, Multi-Step Planning, Fine-Tuning, Agentic Systems

**Authors:** Mrinal Rawat, Ambuje Gupta, Rushil Goomer, Alessandro Di Bari, Neha Gupta, Roberto Pieraccini

**Institution(s):** Uniphore


## Problem Background

当前大型语言模型（LLM）驱动的代理系统在处理复杂任务时存在局限性，特别是在需要多步骤规划和长期决策的场景中，基于 ReAct 的框架仅关注当前行动的推理，缺乏整体任务的结构化规划，导致效率低下；此外，高级推理能力多局限于大型专有模型（如 GPT-4），而较小模型由于资源限制难以胜任复杂任务，亟需一种方法提升代理性能并降低对大模型的依赖。

## Method

* **核心思想**：提出 Pre-Act 方法，作为 ReAct 的增强版本，通过生成多步骤执行计划和详细推理，提升 LLM 代理在复杂任务中的表现，同时通过课程学习和微调策略让较小模型也能具备强大代理能力。
* **多步骤规划**：针对用户输入，Pre-Act 生成一个包含多个步骤的结构化计划，每个步骤明确指定行动和详细推理，计划在执行过程中根据观察结果动态调整，直到最终答案生成。
* **上下文累积与动态调整**：将之前的行动和观察结果作为上下文，融入后续步骤的推理中，确保计划的连贯性和适应性，尤其在步骤结果偏离预期或失败时能够动态调整策略。
* **课程学习与微调**：采用两阶段微调策略，第一阶段在 Glaive 数据集上使用 ReAct 方法初步训练模型，第二阶段在专有数据集上使用 Pre-Act 方法进一步优化，并通过 LoRA 技术减少参数更新以避免灾难性遗忘。
* **数据集适配**：对 Glaive 和专有数据集进行改造，确保支持 Pre-Act 的多步骤规划需求，专有数据集中加入专家标注的详细推理内容，提升训练质量。
* **关键创新**：通过结构化规划和动态调整提升代理性能，同时针对小模型的微调策略降低了延迟和成本，适用于实际应用场景。

## Experiment

* **有效性**：Pre-Act 在多个预训练模型上显著优于 ReAct，例如在 Almita 数据集上平均提升了 70% 的行动召回率（Action Recall）；微调后的 Llama 3.1 70B 模型在 Almita 数据集上比 GPT-4 提升了 69.5% 的行动准确率。
* **端到端表现**：在 Almita 数据集的五个复杂用例上，微调后的 Llama 3.1 70B 模型平均目标完成率（Goal Completion Rate）达到 0.82，远高于 GPT-4 的 ReAct（0.32）和 Pre-Act（0.64）表现。
* **实验设置合理性**：实验覆盖了多个模型（预训练和微调）、多个数据集（域内和域外），并通过两级评估框架（逐轮和端到端）全面衡量代理性能；端到端评估引入模拟环境和里程碑依赖图，贴近真实应用场景。
* **不足与开销**：Glaive 数据集上缺乏 Pre-Act 标注，无法直接对比；微调和多步骤规划可能增加一定的计算开销，但通过 LoRA 技术有效控制了参数更新规模。

## Further Thoughts

多步骤规划的概念可以扩展到自动驾驶或机器人控制等需要长期决策的领域；课程学习和微调策略为资源受限场景下小模型赋能提供了实用范例，值得进一步探索如何通过高质量数据和分阶段训练提升性能；两级评估框架（逐轮和端到端）为复杂代理系统评估提供了全面视角，未来可尝试标准化并应用于更多 LLM 代理任务。