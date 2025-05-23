---
title: "Tokenization Constraints in LLMs: A Study of Symbolic and Arithmetic Reasoning Limits"
pubDatetime: 2025-05-20T10:32:30+00:00
slug: "2025-05-tokenization-reasoning-limits"
type: "arxiv"
id: "2505.14178"
score: 0.6876189705302532
author: "grok-3-latest"
authors: ["Xiang Zhang", "Juntai Cao", "Jiaqi Wei", "Yiwei Xu", "Chenyu You"]
tags: ["LLM", "Tokenization", "Reasoning", "Chain of Thought", "Symbolic Computation"]
institution: ["University of British Columbia", "Zhejiang University", "Cisco", "Stony Brook University"]
description: "本文通过理论和实证研究揭示了分词作为大型语言模型符号推理瓶颈的关键作用，并证明原子对齐分词与链式思维提示的结合可显著提升模型在算术和符号任务中的性能。"
---

> **Summary:** 本文通过理论和实证研究揭示了分词作为大型语言模型符号推理瓶颈的关键作用，并证明原子对齐分词与链式思维提示的结合可显著提升模型在算术和符号任务中的性能。 

> **Keywords:** LLM, Tokenization, Reasoning, Chain of Thought, Symbolic Computation

**Authors:** Xiang Zhang, Juntai Cao, Jiaqi Wei, Yiwei Xu, Chenyu You

**Institution(s):** University of British Columbia, Zhejiang University, Cisco, Stony Brook University


## Problem Background

大型语言模型（LLMs）在符号推理和算术任务中的性能受到分词（Tokenization）方案的根本性制约。
尽管链式思维（Chain-of-Thought, CoT）提示通过外部化中间步骤显著提升了 Transformer 模型的计算能力，但现代分词方法（如字节对编码 BPE）通过合并或模糊原子推理单位，破坏了模型对符号操作的逻辑对齐，导致推理泛化能力受限。
论文旨在揭示分词如何成为 LLMs 推理能力的瓶颈，并探索通过优化分词策略释放模型潜在计算能力。

## Method

*   **理论框架**：引入‘Token Awareness’概念，形式化分析分词粒度如何破坏逻辑对齐，并探讨分词对 CoT 推理表达性的限制，包括信息隐藏（语义模糊）和表达瓶颈（无法外部化复杂状态）。
*   **输入设计**：设计两类输入格式以隔离分词影响——‘原子对齐输入’（确保分词边界与任务所需单位如字符或数字对齐）和‘合并分词输入’（故意合并符号单位以模糊结构），用于对比测试。
*   **实验框架**：提出模型无关的评估方法，通过控制分词格式（纯字符串、空格分隔、逗号分隔、精确项目分词）测试 LLMs 在算术计数、排序和反转等符号任务上的表现，同时对比不同输入长度和模型。
*   **CoT 变体**：对比无 CoT、普通 CoT 和监督 CoT（Supervised CoT，提供明确步骤模板）在不同分词条件下的效果，以验证 CoT 与分词的协同作用。
*   **核心目标**：通过控制变量量化分词对推理性能的影响，并探索最优分词与提示策略的组合。

## Experiment

*   **有效性**：实验表明分词格式对性能影响显著，原子对齐输入（如精确项目分词）相比合并分词输入（如纯 BPE 分词）可提升高达 70% 的准确率，尤其在 CoT 条件下；例如，GPT-4o-mini 在长度 30-40 的计数任务中，使用精确项目分词和 CoT 时准确率达 70.8%，而纯 BPE 分词仅为 2.7%。
*   **优越性**：CoT 提示（尤其是监督 CoT）在长序列任务中显著提升性能，与原子对齐分词结合时效果最佳，证明了两者的协同作用；不同类型 token（如数字 vs 字母）也表现出性能差异，数字任务准确率普遍更高。
*   **合理性**：实验设置全面，覆盖多种任务（计数、排序、反转）、输入长度、模型（GPT-4o-mini、Claude 3.5 Sonnet、Qwen Turbo、OpenAI o1）和分词类型，确保结果普适性；通过对比无 CoT 和 CoT 条件，验证了 CoT 在模拟递归计算中的必要性。
*   **局限性**：未测试极长上下文（数百 token）或更多开源模型（如 LLaMA），部分由于预算限制，但作者认为结果对主流模型具有普适性。

## Further Thoughts

分词不仅是预处理步骤，而是影响模型计算能力的核心组件，未来可设计自适应分词器以匹配任务需求；‘Token Awareness’概念为评估分词影响提供了新视角，可用于改进分词策略；不同类型 token（如数字 vs 字母）对推理性能的差异提示嵌入信息密度和注意力机制的作用，值得深入研究；监督 CoT 通过结构化提示显著提升性能，启发推理时可通过任务特定提示进一步挖掘模型潜力。