---
title: "Enhancing Code Generation via Bidirectional Comment-Level Mutual Grounding"
pubDatetime: 2025-05-12T17:20:30+00:00
slug: "2025-05-interactive-code-grounding"
type: "arxiv"
id: "2505.07768"
score: 0.5739306934802183
author: "grok-3-latest"
authors: ["Yifeng Di", "Tianyi Zhang"]
tags: ["LLM", "Code Generation", "Interactive Feedback", "Code Refinement", "Mutual Grounding"]
institution: ["Purdue University"]
description: "本文提出PING方法，通过内联注释作为双向沟通媒介，显著提升了大型语言模型的代码生成准确率和开发者体验，展示了人机协作在代码生成中的潜力。"
---

> **Summary:** 本文提出PING方法，通过内联注释作为双向沟通媒介，显著提升了大型语言模型的代码生成准确率和开发者体验，展示了人机协作在代码生成中的潜力。 

> **Keywords:** LLM, Code Generation, Interactive Feedback, Code Refinement, Mutual Grounding

**Authors:** Yifeng Di, Tianyi Zhang

**Institution(s):** Purdue University


## Problem Background

大型语言模型（LLMs）在代码生成中表现出色，但生成的代码常存在功能性错误，尤其是在处理复杂或未见过的编程任务时，导致开发者在检查和修复代码时效率低下，并降低了对LLM工具的信任。
论文旨在解决LLM生成的代码与开发者意图不一致的问题，通过引入双向沟通机制，建立开发者与模型之间的共同理解（Mutual Grounding），从而提升代码质量和开发体验。

## Method

*   **核心思想:** 提出一种名为Programming with Interactive Grounding (PING)的交互式方法，利用代码内联注释（Inline Comments）作为开发者与模型之间的沟通媒介，促进双向理解和代码精炼。
*   **具体步骤:**
    *   **注释生成（Comment Generation）:** 使用经过微调的CodeBERT模型，解析代码的抽象语法树（AST），为每个代码语句生成细粒度的内联注释，帮助开发者快速理解代码行为。针对复杂语句（如if、for）采用定制化分割策略，确保注释准确性；若代码不可编译，则按行分割生成注释。
    *   **人工审查（Human Review）:** 开发者阅读生成的注释，发现错误后直接编辑对应注释，描述期望的代码行为，而无需直接修改代码。这种方式降低了开发者的认知负担，尤其对不熟悉特定API的开发者更为友好。
    *   **代码精炼（Code Refinement）:** 基于编辑后的注释和上下文，使用微调后的DeepSeek Coder（6.7B）模型重新生成出错语句及其后续代码段，而非整个代码片段，从而提高精炼效率。微调数据集来自Stack数据集，通过筛选高注释比例的代码片段，并采用下一token预测目标进行优化。
*   **技术细节:** 注释生成模型基于CodeBERT，微调数据集包含28万代码-注释对；代码精炼模型基于DeepSeek Coder，微调时使用Adam优化器和交叉熵损失，确保模型适应细粒度精炼任务。
*   **关键优势:** 通过细粒度反馈和局部代码再生，避免了传统方法中整体代码重生成的低效问题，同时增强了开发者对模型输出的控制感。

## Experiment

*   **有效性:** 在模拟用户研究中，PING显著提升了多个LLM的代码生成准确率。例如，在HumanEval数据集上，code-davinci-002的pass@1从46.3%提升至63.4%（增幅17.1%），InCoder从16.5%提升至29.9%；在MBPP数据集上，类似提升也显著。多轮迭代反馈进一步提高准确率，如InCoder在三轮后达到34.1%。
*   **对比优越性:** 与八种基线方法（如ReAct、Self-Debug、CodeChain）相比，PING在HumanEval和MBPP上的pass@1分别达到65.9%和71.1%，表现最佳，同时任务完成时间最短（如HumanEval上为35.2秒，优于其他方法的37.8-64.5秒）。
*   **用户研究:** 在真实用户研究中（12名参与者），PING的任务成功率比GitHub Copilot和Multi-Turn Program Synthesis分别高16.7%和58.3%，任务完成时间分别缩短10.5%和22.9%。参与者对PING的信心和满意度提升20%，认知负荷（如挫折感、努力程度）也显著低于基线。
*   **实验设置合理性:** 实验覆盖多种模型、基准数据集（HumanEval、MBPP及其增强版）和任务难度（简单到困难），并结合模拟与真实用户研究，设置较为全面。但模拟用户研究由作者提供反馈，可能存在主观偏见；真实用户研究样本量较小（12人），可能影响结果普适性。
*   **额外观察:** 微调代码精炼模型对准确率有额外提升（如InCoder在HumanEval上提升3.7%）；不同注释生成模型（如CodeBERT vs Seq2Seq）对结果影响较小（约2-3%），表明PING的核心优势在于交互反馈而非注释模型本身。

## Further Thoughts

论文提出的双向共同理解（Mutual Grounding）理念非常具有启发性，不仅限于代码生成，还可能应用于其他需要人机协作的领域（如文本生成或数据分析），通过细粒度反馈机制提升模型对用户意图的理解。
此外，细粒度反馈（Statement-Level Feedback）与局部代码再生的结合是一个创新点，未来可以探索将其与自动化错误检测结合，进一步减少人工干预。
最后，注释呈现方式对复杂代码可读性的影响值得关注，未来可研究按需显示或分组注释的优化策略，以平衡理解与简洁性。