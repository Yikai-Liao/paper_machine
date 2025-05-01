---
title: "SeriesBench: A Benchmark for Narrative-Driven Drama Series Understanding"
pubDatetime: 2025-05-01T15:52:52Z
slug: "2025-04-seriesbench-narrative"
type: "arxiv"
id: "2504.21435"
score: 0.7965481891409583
author: "grok-3-latest"
authors: ["Chenkai Zhang", "Yiming Lei", "Zeming Liu", "Haitao Leng", "ShaoGuo Liu", "Tingting Gao", "Qingjie Liu", "Yunhong Wang"]
tags: ["MLLM", "Video Understanding", "Narrative Reasoning", "Benchmark Design", "Multi-Modal"]
institution: ["State Key Laboratory of Virtual Reality Technology and Systems, Beihang University", "School of Computer Science and Engineering, Beihang University", "Hangzhou Innovation Institute, Beihang University", "Kuaishou Technology"]
description: "本文提出 SeriesBench 基准和 PC-DCoT 框架，首次针对叙事驱动系列剧设计全面评估体系，并通过双链推理显著提升多模态大语言模型的叙事理解能力。"
---

> **Summary:** 本文提出 SeriesBench 基准和 PC-DCoT 框架，首次针对叙事驱动系列剧设计全面评估体系，并通过双链推理显著提升多模态大语言模型的叙事理解能力。 

> **Keywords:** MLLM, Video Understanding, Narrative Reasoning, Benchmark Design, Multi-Modal
> **Recommendation Score:** 0.7965481891409583

**Authors:** Chenkai Zhang, Yiming Lei, Zeming Liu, Haitao Leng, ShaoGuo Liu, Tingting Gao, Qingjie Liu, Yunhong Wang
**Institution(s):** State Key Laboratory of Virtual Reality Technology and Systems, Beihang University, School of Computer Science and Engineering, Beihang University, Hangzhou Innovation Institute, Beihang University, Kuaishou Technology

## Problem Background

随着多模态大语言模型（MLLMs）的快速发展，视频理解能力成为研究热点，但现有基准主要聚焦于独立视频，关注视觉元素（如动作、物体状态），忽略了现代视频中常见的复杂叙事结构和跨视频系列的角色发展，尤其是在剧情驱动的系列剧中，模型在深层叙事理解和角色关系分析上表现不足，这可能限制其在系列推荐、交互媒体和视频摘要等领域的应用。

## Method

* **SeriesBench 基准构建**：
  - 收集了 105 个叙事驱动系列剧（共 1072 个视频），覆盖多种类型（如日常生活、动漫、历史剧等）。
  - 设计了 5 大任务维度（视觉、剧本、音频、增强、综合理解），细分为 28 个子任务，涵盖多模态内容。
  - 采用长跨度叙事标注方法（long-span narrative annotation），由 30 多名专业标注员标注关键事件和角色发展，并通过全信息转换技术（full-information transformation）生成多样化问题类型（如多选、判断、开放式问题）。
* **PC-DCoT 框架（Plot & Character Dual Chain of Thought）**：
  - 受人类刷剧行为启发，提出一种叙事推理框架，通过构建‘剧情事件链’（Plot Event Chain）和‘角色时间链’（Character Temporal Chain）增强模型对系列剧叙事结构的理解。
  - 具体步骤：1）从视频和问题中提取关键事件和角色；2）利用视频片段模型识别相关场景，分别构建事件链（描述连续事件）和角色链（追踪角色行为）；3）基于时间轴对齐事件和角色，进行双链合成推理，生成综合答案。
  - 关键创新在于将叙事理解分解为事件和角色两个维度，并通过时间对齐实现结构化推理，而非单纯依赖模型的通用推理能力。

## Experiment

* **模型表现不足**：在 SeriesBench 上测试了 10 个领先的视频 MLLMs（如 InternVL2, Qwen2-VL, GPT-4o），结果显示即使是 SOTA 模型（如 GPT-4o，整体准确率 62.8%）也远低于人类水平（95.8%），特别是在细粒度视觉分析和深层叙事理解任务上。
* **PC-DCoT 提升显著**：应用 PC-DCoT 后，模型性能显著提升，例如 InternVL2 从 59.2% 提升到 73.3%（提升 14.1%），GPT-4o 从 62.8% 提升到 76.2%（提升 13.4%），表明该框架在叙事理解任务上的有效性。
* **实验设置合理性**：数据集按 8:1:1 划分为训练、验证和测试集，任务类型多样（多选、判断、开放式），评估指标包括准确率、BLEU-2、METEOR 和 BERTScore F1，覆盖词法和语义层面；此外，实验还包括多集任务分析、模态影响分析和消融研究，验证了双链设计的重要性。
* **局限性**：尽管性能提升明显，模型与人类水平仍有差距，尤其在复杂因果推理和多角色剧情分析上，表明叙事理解仍需进一步研究。

## Further Thoughts

PC-DCoT 的双链推理框架启发了我，人类在观看系列剧时同时关注剧情发展和角色关系，这种‘双轨思维’可以通过结构化方式引入模型，或许可以扩展到更多维度（如情感链、主题链）以捕捉更复杂的叙事元素；此外，SeriesBench 的长跨度标注方法也提示我们，是否可以利用类似方法构建其他领域的长序列理解基准（如长文档、长音频），推动模型在长上下文推理上的能力。