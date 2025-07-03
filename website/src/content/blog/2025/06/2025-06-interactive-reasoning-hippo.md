---
title: "Interactive Reasoning: Visualizing and Controlling Chain-of-Thought Reasoning in Large Language Models"
pubDatetime: 2025-06-30T10:00:43+00:00
slug: "2025-06-interactive-reasoning-hippo"
type: "arxiv"
id: "2506.23678"
score: 0.604340882007431
author: "grok-3-latest"
authors: ["Rock Yuren Pang", "Chu Li", "K. J. Kevin Feng", "Weijia Shi", "Shangbin Feng", "Yulia Tsvetkov", "Jeffrey Heer", "Katharina Reinecke"]
tags: ["LLM", "Reasoning", "User Interaction", "Test Time Scaling", "Visualization"]
institution: ["University of Washington"]
description: "本文提出交互式推理设计，通过Hippo系统可视化和控制大型语言模型的思维链，显著提升用户对推理过程的理解和控制感，为AI决策支持开辟了新范式。"
---

> **Summary:** 本文提出交互式推理设计，通过Hippo系统可视化和控制大型语言模型的思维链，显著提升用户对推理过程的理解和控制感，为AI决策支持开辟了新范式。 

> **Keywords:** LLM, Reasoning, User Interaction, Test Time Scaling, Visualization

**Authors:** Rock Yuren Pang, Chu Li, K. J. Kevin Feng, Weijia Shi, Shangbin Feng, Yulia Tsvetkov, Jeffrey Heer, Katharina Reinecke

**Institution(s):** University of Washington


## Problem Background

大型语言模型（LLMs）通过生成思维链（Chain-of-Thought, CoT）内容提升输出质量，但这些推理步骤冗长且缺乏结构化组织，用户难以快速理解；此外，用户无法直接干预推理过程，尤其在高风险决策场景（如伦理、金融、医疗）中，模型推理可能与用户价值观或优先级不一致，导致输出不符合需求。
因此，论文旨在解决如何让用户更有效地理解和控制LLM推理过程的问题，以提升用户体验和决策支持效果。

## Method

*   **核心理念：交互式推理（Interactive Reasoning）**：将LLM的复杂推理链转化为交互式树状层次结构，使用户能够直观理解并直接控制推理过程。
*   **具体实现步骤：**
    *   **结构化推理文本**：利用LLM（如GPT-4o）通过少样本提示（few-shot prompting）将冗长推理文本分解为主题和子主题的层次结构，使用XML-like标签（如<topic>和<branch>）标注，确保文本按逻辑组织。
    *   **交互式树状界面**：在Hippo原型系统中，推理节点以预序遍历（深度优先）方式逐步生成，用户可实时观察模型推理过程，并对节点执行添加、编辑、删除或重新生成操作，降低认知负荷。
    *   **用户反馈机制**：通过‘Clarify’操作，利用LLM分类能力识别需要用户澄清的节点（如涉及不确定性或个人偏好），暂停推理生成并提示用户输入，同时避免重复询问以减少用户疲劳。
    *   **推理与输出链接**：通过自然语言推理（NLI）任务，使用零样本提示方法，将最终输出句子与推理节点关联，增强用户对输出来源的可追溯性。
*   **技术支持**：系统后端基于GPT-4o和DeepSeek-R1模型，前端使用Next.js和Tailwind CSS实现，确保高效性和低成本（GPT-4o成本为$2.50/百万token）。
*   **设计目标**：强调用户直接操控（DG1）、认知参与假设（DG2）、图形化减少信息负荷（DG3）、及时干预（DG4）和输出溯源（DG5），平衡自动化与人类干预。

## Experiment

*   **有效性**：在用户研究中（16名参与者），Hippo系统在控制感（p=0.003）、推理理解（sense-making, p=0.004）、信息布局（p=0.009）和假设觉察（p=0.012）方面显著优于可编辑基线系统（线性文本推理），用户对最终决策的信心也有提升（p=0.049）。
*   **合理性**：实验设置合理，任务基于日常伦理困境（如财务管理和友情冲突），贴近高风险决策场景；参与者背景多样（学生到专业人士），条件平衡，减少学习效应干扰。
*   **局限性**：对最终输出满意度和洞察获取无显著差异（p>0.1），可能因基线输出已较‘体面’；部分用户对频繁交互感到疲倦，显示交互频率与用户体验的权衡问题；此外，样本量较小（N=16）且参与者对LLM较熟悉，可能影响结果普适性。
*   **额外观察**：案例研究（信息搜索和金融规划）进一步验证了Hippo在支持复杂决策和深度逻辑审查中的潜力，但也揭示了用户对交互深度需求的差异。

## Further Thoughts

论文启发我们重新定义AI系统的输出目标——不仅仅是最终结果，而是支持用户决策的推理过程；这一理念可扩展到其他领域，如推荐系统或自动驾驶，通过可视化和交互增强用户信任。此外，自适应交互设计（根据任务复杂度和用户偏好调整推理展示和干预频率）是一个值得探索的方向，例如为简单任务提供简略推理，为复杂决策提供深度交互，平衡效率与参与感。