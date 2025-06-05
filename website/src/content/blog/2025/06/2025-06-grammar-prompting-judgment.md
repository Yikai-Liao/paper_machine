---
title: "Explain-then-Process: Using Grammar Prompting to Enhance Grammatical Acceptability Judgments"
pubDatetime: 2025-06-02T22:42:33+00:00
slug: "2025-06-grammar-prompting-judgment"
type: "arxiv"
id: "2506.02302"
score: 0.7591893139379627
author: "grok-3-latest"
authors: ["Russell Scheinberg", "Ameeta Agrawal", "Amber Shore", "So Young Lee"]
tags: ["LLM", "Grammar Prompting", "Acceptability Judgment", "Multilingual", "Reasoning"]
institution: ["Portland State University", "Miami University"]
description: "本文提出‘explain-then-process’语法提示方法，通过模型自生成的语法解释显著提升多语言语法判断任务准确率，尤其对小型模型效果显著。"
---

> **Summary:** 本文提出‘explain-then-process’语法提示方法，通过模型自生成的语法解释显著提升多语言语法判断任务准确率，尤其对小型模型效果显著。 

> **Keywords:** LLM, Grammar Prompting, Acceptability Judgment, Multilingual, Reasoning

**Authors:** Russell Scheinberg, Ameeta Agrawal, Amber Shore, So Young Lee

**Institution(s):** Portland State University, Miami University


## Problem Background

大型语言模型（LLMs）在语言生成和功能性任务上表现出色，但当需要明确判断句子的语法可接受性（grammatical acceptability judgments）时，常常无法系统性地应用其隐含的语法知识，尤其在多语言场景中存在显著缺陷。
作者区分了形式性能力（formal competence，即区分语法与非语法形式）和功能性能力（functional competence，即实际语言使用），指出 LLMs 在形式性能力上的不足，并旨在通过新的提示方法弥合‘知道规则’与‘应用规则’之间的差距。

## Method

*   **核心思想：** 提出‘explain-then-process’（解释-处理）范式，通过‘grammar prompting’（语法提示）利用模型自身的元语言能力生成语法解释，并将其作为上下文反馈给目标模型，指导其在语法判断任务中应用规则。
*   **具体步骤：**
    *   **解释阶段（Explain）：** 首先让一个大型语言模型（LLM，如 GPT-4o 或 Claude 3.5 Sonnet）针对特定语法现象生成简洁的解释，解释可分为初学者导向（简明易懂）和专家导向（技术性强）。
    *   **处理阶段（Process）：** 将生成的语法解释作为额外上下文输入给目标模型（可以是 LLM 或小型语言模型 SLM），随后目标模型基于此解释判断一对最小对（minimal pair）句子中哪个更符合语法。
    *   **结合链式思维（CoT）：** 进一步测试语法提示与链式思维提示结合的效果，即在处理阶段要求模型逐步推理后再给出判断。
*   **设计细节：** 语法解释通过精心设计的指令模板生成，确保覆盖关键规则和语法细节；解释生成后可重复使用，成本低廉；方法语言无关，适用于多语言场景。
*   **创新点：** 通过显性化模型的隐性语法知识，引导其将规则应用于具体任务，尤其对资源受限的小型模型（SLM）有显著提升。

## Experiment

*   **有效性：** 在英语（BLiMP）、中文（SLING）和俄语（RuBLiMP）三个基准数据集上，语法提示（GP）显著提升了模型的语法判断准确率，例如 GPT-4o 在 BLiMP 上从基准 78.0% 提升至 GP+CoT 条件下的 92.7%，小型模型 GPT-3.5 从 67.9% 提升至 78.4%；在中文和俄语数据集上也有类似提升，尤其对 SLM 效果更明显。
*   **对比分析：** 相比基准提示（Base）和单独链式思维（CoT），语法提示在大多数语法现象上表现更好，尤其对简单分布约束（如英语负极性项 NPI licensing）接近完美准确率；对复杂现象（如岛屿效应 island effects）提升较小但仍显著；控制条件（无关解释）和教科书条件（多规则混合）验证了提升来源于针对性语法解释。
*   **实验设置合理性：** 实验覆盖多个模型（包括 GPT-4o、Claude 3.5 Sonnet、GPT-3.5、Llama 3.3 9B 等），聚焦三个语言的挑战性语法现象（初始准确率较低），提示条件多样（初学者/专家解释、控制条件等），句子呈现顺序随机化以避免偏见，设计全面合理。
*   **成本与局限：** 语法提示额外计算成本极低，仅需生成一次解释并重复使用；局限包括未测试 SLM 生成解释的能力，以及对低资源语言效果未验证。

## Further Thoughts

‘explain-then-process’范式启发我们利用模型的元语言能力作为自引导机制，通过显性化隐性知识提升任务表现，这种思路不仅限于语法判断，还可能扩展到逻辑推理或代码调试等领域；此外，语法提示对小型模型（SLM）的显著提升提示了一种低成本部署策略，即用一个大型模型生成解释服务多个小型模型，尤其在多语言场景中具有潜力；针对复杂任务效果有限的问题，未来可探索分层提示策略，根据任务复杂度设计不同深度的解释或结合示例。