---
title: "Software Development Life Cycle Perspective: A Survey of Benchmarks for CodeLLMs and Agents"
pubDatetime: 2025-05-08T14:27:45+00:00
slug: "2025-05-code-llm-benchmarks"
type: "arxiv"
id: "2505.05283"
score: 0.38957150506676486
author: "grok-3-latest"
authors: ["Kaixin Wang", "Tianlin Li", "Xiaoyu Zhang", "Chong Wang", "Weisong Sun", "Yang Liu", "Bin Shi"]
tags: ["LLM", "Code Generation", "Software Engineering", "Benchmarking", "SDLC"]
institution: ["Xi’an Jiaotong University", "Nanyang Technological University"]
description: "本文通过系统分析181个CodeLLMs和代理基准测试，揭示了SDLC各阶段评估的不平衡性，并为未来基准测试设计提供了全面指导。"
---

> **Summary:** 本文通过系统分析181个CodeLLMs和代理基准测试，揭示了SDLC各阶段评估的不平衡性，并为未来基准测试设计提供了全面指导。 

> **Keywords:** LLM, Code Generation, Software Engineering, Benchmarking, SDLC

**Authors:** Kaixin Wang, Tianlin Li, Xiaoyu Zhang, Chong Wang, Weisong Sun, Yang Liu, Bin Shi

**Institution(s):** Xi’an Jiaotong University, Nanyang Technological University


## Problem Background

代码大语言模型（CodeLLMs）和代理在软件工程中的应用日益广泛，展现出比传统方法更强的复杂任务处理能力和跨任务泛化能力。然而，现有基准测试（Benchmarks）缺乏系统性综述，无法全面评估这些模型在软件开发生命周期（SDLC）各阶段的表现，尤其是在需求工程和软件设计等早期阶段的覆盖不足，难以反映模型在真实场景中的应用潜力。

## Method

* **核心方法：** 本文通过系统文献综述和分类分析，对181个CodeLLMs和代理的基准测试进行全面评估，覆盖461篇相关论文。
* **文献收集：** 采用关键词总结、学术数据库搜索（如IEEE Xplore、ACM Digital Library）、手动筛选和雪球式扩展四步策略，确保研究覆盖面广，最终收集461篇高质量文献。
* **分类框架：** 以软件开发生命周期（SDLC）的五个阶段（需求工程、软件设计、软件开发、测试、软件维护）为框架，将基准测试按阶段和具体任务（如代码生成、漏洞检测）进行细致分类，梳理其应用场景和评估目标。
* **统计分析：** 从使用频率、发布年份、编程语言分布和SDLC阶段分布等维度对基准测试进行量化分析，揭示研究热点和空白领域。
* **问题与方向：** 基于分析结果，总结当前基准测试的局限性（如任务覆盖不均、现实场景不足），并提出未来研究方向（如跨阶段评估、多模态基准测试）。
* **特点：** 方法注重系统性和多维度分析，不仅关注具体基准测试的内容，还从宏观视角探讨研究趋势和差距，为未来基准测试设计提供指导。

## Experiment

* **分析结果：** 通过对181个基准测试的统计，发现约60%的基准测试集中在软件开发阶段，而需求工程和软件设计阶段分别仅占5%和3%，显示出明显的不平衡；Python是最常用的编程语言（支持100个基准测试），反映了语言流行度对研究的影响；HumanEval是使用频率最高的基准测试（62次），表明代码生成任务是研究热点；2022年后基准测试数量激增，2024年达到顶峰，显示领域快速发展。
* **全面性与合理性：** 分析覆盖SDLC全阶段，统计数据详实，揭示了研究偏见（如早期阶段忽视）和空白领域，逻辑严谨，为未来研究提供了可靠依据。
* **局限性：** 本文未深入评估具体基准测试的质量（如数据可靠性、评估指标有效性），仅停留在数量和分布层面，缺乏对模型在这些基准测试上表现的直接对比或验证。

## Further Thoughts

论文提出的跨阶段评估理念启发了我，未来可以设计‘端到端’基准测试，模拟从需求分析到维护的全流程任务，评估模型在阶段间协作和信息传递中的表现；此外，多模态基准测试的潜力值得探索，例如从UI设计草图生成代码，或从视频教程提取编程逻辑，以更贴近实际开发场景；最后，人机协作评估是一个被忽视但重要的方向，可以设计基准测试评估模型在与人类开发者交互时的效率提升和意图理解能力。