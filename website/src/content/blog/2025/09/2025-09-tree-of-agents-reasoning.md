---
title: "Tree of Agents: Improving Long-Context Capabilities of Large Language Models through Multi-Perspective Reasoning"
pubDatetime: 2025-09-08T08:34:02+00:00
slug: "2025-09-tree-of-agents-reasoning"
type: "arxiv"
id: "2509.06436"
score: 0.7131449427105805
author: "grok-3-latest"
authors: ["Song Yu", "Xiaofei Xu", "Ke Deng", "Li Li", "Lin Tian"]
tags: ["LLM", "Long Context", "Multi-Agent", "Reasoning", "Efficiency"]
institution: ["Southwest University", "Murdoch University", "RMIT University", "University of Technology Sydney"]
description: "本文提出 Tree of Agents (TOA) 框架，通过多代理协作和树状结构的多视角推理，显著提升大型语言模型在长上下文任务中的性能，同时通过缓存和剪枝策略优化计算效率。"
---

> **Summary:** 本文提出 Tree of Agents (TOA) 框架，通过多代理协作和树状结构的多视角推理，显著提升大型语言模型在长上下文任务中的性能，同时通过缓存和剪枝策略优化计算效率。 

> **Keywords:** LLM, Long Context, Multi-Agent, Reasoning, Efficiency

**Authors:** Song Yu, Xiaofei Xu, Ke Deng, Li Li, Lin Tian

**Institution(s):** Southwest University, Murdoch University, RMIT University, University of Technology Sydney


## Problem Background

大型语言模型（LLMs）在处理长上下文任务时面临显著挑战，尤其是‘lost in the middle’问题，即模型往往忽略长输入中间部分的信息，导致推理能力下降。
现有方法如输入压缩可能丢失关键信息，而扩展上下文窗口则会导致注意力分散和计算成本激增，论文旨在解决如何有效提升长上下文理解能力，同时避免信息丢失和高计算开销。

## Method

*   **核心思想:** 提出 Tree of Agents (TOA) 框架，通过多代理协作和树状结构的多路径推理，实现对长上下文的多视角理解，缓解位置偏见和幻觉问题。
*   **具体实现:** 
    *   **文本分割与代理分配:** 将长文档分割成多个小块（chunks），每个小块分配给一个独立代理（agent），每个代理基于其分配的小块生成初始认知状态（包括证据和答案）。
    *   **多视角推理:** 在第二阶段，代理通过树状结构探索不同的小块阅读顺序，形成多路径推理，确保从不同角度理解文档内容，避免固定顺序带来的偏见。
    *   **信息交换与更新:** 代理间交换认知状态，动态请求访问其他代理的小块，基于新信息更新自身认知，形成更全面的理解。
    *   **优化策略:** 引入前缀哈希缓存（prefix-hash caching）以重用中间状态，减少重复计算；采用自适应剪枝（adaptive pruning）策略，提前终止无效推理路径，提升计算效率。
    *   **共识形成:** 最后通过两级投票机制（代理内聚合和跨代理多数投票）合成最终答案，确保结果的鲁棒性。
*   **关键点:** 该方法无需修改底层模型结构，属于即插即用（plug-and-play）方案，且通过多视角推理显著提升了对长上下文的全局理解能力。

## Experiment

*   **有效性:** TOA 在多个长上下文任务（如 DetectiveQA, NovelQA, Needle-in-a-Haystack）上表现出色，基于 LLaMA3.1-8B 模型的准确率分别达到 54.3% 和 45.0%，显著优于基线方法（如 COA 的 25.3% 和 LONGAGENT 的 48.7%），且 none-rate 极低（仅 1.7% 和 4.3%），表明其避免幻觉的能力较强。
*   **优越性:** 相比其他多代理方法（如 COA 和 LONGAGENT），TOA 在‘lost in the middle’问题上表现更稳定，尤其在多针（multi-needle）任务中平均得分提升超过 100%；与大型商业模型（如 Gemini 1.5-pro）相比，TOA 使用更小的模型即可达到接近甚至超越的表现，凸显架构创新的价值。
*   **实验设置合理性:** 实验覆盖了不同数据集、模型（LLaMA3.1-8B 和 DeepSeek-V3）、输入长度和代理数量的影响，设置较为全面；同时测试了缓存和剪枝策略对效率的提升效果，API 调用量减少了 50.8%。
*   **局限性:** 尽管有优化策略，TOA 的计算开销仍高于简单基线（如 COA），特别是在大规模部署时，推理速度是一个瓶颈。

## Further Thoughts

TOA 的多视角推理机制启发我们可以在其他 NLP 任务中探索类似的多代理协作模式，例如对话系统或多文档总结，通过不同代理从不同角度分析输入，提升全局理解能力；此外，是否可以通过强化学习动态优化代理的路径选择，或者结合图神经网络建模代理间的复杂依赖关系，进一步提升推理效率和效果？