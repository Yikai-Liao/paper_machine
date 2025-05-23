---
title: "Prior Prompt Engineering for Reinforcement Fine-Tuning"
pubDatetime: 2025-05-20T10:05:11+00:00
slug: "2025-05-prior-prompt-rft"
type: "arxiv"
id: "2505.14157"
score: 0.7205778449768043
author: "grok-3-latest"
authors: ["Pittawat Taveekitworachai", "Potsawee Manakul", "Sarana Nutanong", "Kunat Pipatanakul"]
tags: ["LLM", "Prompt Engineering", "Reinforcement Learning", "Reasoning", "Behavior Shaping"]
institution: ["SCB 10X R&D, SCBX Group, Thailand", "Vidyasirimedhi Institute of Science and Technology, Thailand"]
description: "本文提出前期提示工程（pPE）方法，通过在强化微调（RFT）中设计不同提示引导语言模型内化多样化行为并显著提升性能，揭示了提示设计在训练阶段的重要潜力。"
---

> **Summary:** 本文提出前期提示工程（pPE）方法，通过在强化微调（RFT）中设计不同提示引导语言模型内化多样化行为并显著提升性能，揭示了提示设计在训练阶段的重要潜力。 

> **Keywords:** LLM, Prompt Engineering, Reinforcement Learning, Reasoning, Behavior Shaping

**Authors:** Pittawat Taveekitworachai, Potsawee Manakul, Sarana Nutanong, Kunat Pipatanakul

**Institution(s):** SCB 10X R&D, SCBX Group, Thailand, Vidyasirimedhi Institute of Science and Technology, Thailand


## Problem Background

强化微调（Reinforcement Fine-Tuning, RFT）通过奖励信号激励语言模型（LLM）在推理时展现特定行为（如逐步推理），但现有研究主要关注算法、奖励设计和数据选择，而忽略了训练过程中用于引导行为的前期提示（Prior Prompt）设计的重要性。
论文提出核心问题：不同的前期提示工程（Prior Prompt Engineering, pPE）方法是否能在 RFT 过程中引导模型内化不同的行为模式，并带来性能提升？这一研究填补了 RFT 中提示设计的研究空白，探索了提示在训练阶段对模型行为的塑造潜力。

## Method

*   **核心思想：** 将推理时提示工程（Inference-Time Prompt Engineering, iPE）的成功策略转化为前期提示工程（pPE），在 RFT 过程中通过不同提示设计引导语言模型内化特定行为模式。
*   **具体实现：** 选取五种代表性 iPE 策略并映射为 pPE 方法：
    *   **推理（Reasoning）**：基于链式思维（Chain-of-Thought, CoT），使用 <think> 标签，引导模型逐步推理。
    *   **规划（Planning）**：基于计划与解决（Plan-and-Solve, PS），使用 <plan> 标签，要求模型先制定计划再执行。
    *   **代码推理（Code-Based Reasoning）**：基于程序思维（Program-of-Thought, PoT），使用 <code> 标签，鼓励通过代码解决问题。
    *   **知识回忆（Knowledge Recall）**：基于生成知识提示，使用 <knowledge> 标签，引导模型回忆相关知识后再回答。
    *   **空示例利用（Null-Example Utilization）**：基于空示例提示（Null-Shot Prompting），使用 <examples> 标签，鼓励模型生成虚构示例辅助推理。
*   **训练过程：** 使用 Qwen2.5-7B 作为基础模型，训练数据限于数学领域（STILLv3 数据集），采用 Group Relative Policy Optimization (GRPO) 算法优化模型行为，奖励函数由准确性（Accuracy）和格式（Format）两部分组成，确保模型输出既正确又符合预期结构。
*   **关键点：** pPE 方法仅通过修改前期提示即可影响训练轨迹和最终行为，无需改变模型架构或核心算法，是一种低成本、高灵活性的行为塑造方式。

## Experiment

*   **有效性：** 所有 pPE 训练的模型在数学推理（AIME2024, AMC12, MATH-500）、编码（HumanEval+）和问答（GPQA-Diamond）等基准数据集上的性能均优于对应 iPE 基线和基础模型，其中空示例利用（<examples>）方法取得最高平均性能提升（+6.98 分），超越常用推理方法（<think>，+6.37 分）。
*   **差异性：** 不同 pPE 方法在任务上的表现不一，例如代码推理方法在编码任务（HumanEval+）上提升最显著（78.00%），而知识回忆方法在问答任务（GPQA）上表现较弱（21.72%）；这表明 pPE 的影响并非简单与任务领域对齐。
*   **行为分析：** 定性分析显示 pPE 方法引发了不同的行为模式，例如 <plan> 方法生成的模型倾向于先列出计划步骤，<code> 方法更倾向于代码推理；响应长度分析表明 <examples> 方法在高性能的同时生成最短响应，效率最高。
*   **泛化性：** 在不同模型规模（Qwen2.5-3B）和家族（Llama 3.1-8B, Qwen2.5-Coder-7B）上的实验表明，pPE 在较强模型上表现稳定，而较弱模型可能出现奖励欺骗或无法内化目标行为。
*   **实验设置合理性：** 实验涵盖了领域内和领域外任务，结合定量和定性分析，数据支持了 pPE 方法的有效性，但也揭示了方法在不同任务和模型上的不一致性，提示需进一步研究其机制。

## Further Thoughts

pPE 作为 RFT 的新维度，通过简单提示设计即可实现模型行为的多样化定制，启发我们思考如何将提示工程与强化学习更紧密结合；动态 pPE 的概念（根据任务类型或难度选择提示）可能进一步提升性能；此外，通过可验证奖励间接激励中间行为的训练方式（如计划生成或代码合成）无需直接监督信号即可提取有用行为，这一思路可扩展至更多领域，如对话风格定制或安全对齐训练。