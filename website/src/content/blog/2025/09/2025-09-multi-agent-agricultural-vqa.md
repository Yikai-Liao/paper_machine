---
title: "Dynamic Orchestration of Multi-Agent System for Real-World Multi-Image Agricultural VQA"
pubDatetime: 2025-09-29T06:52:10+00:00
slug: "2025-09-multi-agent-agricultural-vqa"
type: "arxiv"
id: "2509.24350"
score: 0.8001492376650549
author: "grok-3-latest"
authors: ["Yan Ke", "Xin Yu", "Heming Du", "Scott Chapman", "Helen Huang"]
tags: ["Multimodal Models", "Multi-Agent Systems", "Retrieval Augmented Generation", "Visual Question Answering"]
institution: ["The University of Queensland"]
description: "本文提出一个自反思和自改进的多智能体框架，通过检索者、反思者、回答者和改进者的协作，显著提升了多图像农业视觉问答的性能和可靠性。"
---

> **Summary:** 本文提出一个自反思和自改进的多智能体框架，通过检索者、反思者、回答者和改进者的协作，显著提升了多图像农业视觉问答的性能和可靠性。 

> **Keywords:** Multimodal Models, Multi-Agent Systems, Retrieval Augmented Generation, Visual Question Answering

**Authors:** Yan Ke, Xin Yu, Heming Du, Scott Chapman, Helen Huang

**Institution(s):** The University of Queensland


## Problem Background

农业视觉问答（Agricultural Visual Question Answering, VQA）在为农民和研究人员提供准确、及时的知识方面至关重要，但现有方法多局限于纯文本或单一图像输入，无法应对现实农业场景中常见的多图像输入需求，这些输入往往涉及跨空间尺度、生长阶段的互补视图。此外，现有系统缺乏对动态农业环境（如作物生长阶段、疾病症状变化）的适应能力，以及对答案质量的系统性控制，导致其在证据不完整或查询模糊时表现不佳。

## Method

* **核心思想**：提出一个自反思和自改进的多智能体框架，通过四个角色（Retriever、Reflector、Answerer、Improver）的协作，实现上下文丰富、反思推理、答案起草和迭代改进，以适应动态农业场景并提升答案可靠性。
* **Retriever（检索者）**：根据用户查询动态制定搜索关键词，调用不同检索工具（如天气数据、农业文献），并在反思者反馈下迭代优化检索结果，确保获取与农业场景相关的可靠信息。
* **Reflector（反思者）**：评估检索信息的质量（如主题相关性、事实一致性、数据时效性），并检查是否符合特定农业上下文（如作物类型、生长阶段），若不足则触发查询重写和重新检索，确保只有高质量信息进入推理阶段。
* **Answerer（回答者）**：两个回答者并行基于验证后的证据生成候选答案，并通过交叉检查减少个体偏差，填补推理空白，形成综合性初步答案，特别适用于农业问题中多重解释的场景。
* **Improver（改进者）**：对初步答案进行质量控制，评估其完整性、指令遵循性和与多图像证据的一致性，确保答案综合所有图像的互补信息而非偏向单一视图，若不达标则指导回答者进一步修订。
* **协作流程**：通过迭代循环实现自反思和自改进，例如检索阶段的质量评估触发重新检索，答案阶段的验证减少幻觉，确保最终输出对证据的忠实性和实用性。

## Experiment

* **有效性**：在 AgMMU 数据集上，框架在疾病识别、害虫识别等农业 VQA 类别中的平均准确率达到 90.78%，显著优于大多数基准模型（如 GPT-4o 89%、Qwen2.5-VL-7B 81.32%），尤其在需要精细视觉推理的类别中表现突出。
* **多图像场景表现**：在不同图像数量（1 到 ≥4 张）的测试中，框架准确率均超过 90%，在单图像和三图像类别中取得最佳成绩，在 ≥4 张图像类别中接近顶级表现（95% vs Gemini-1.5-Pro 的 100%），验证了其整合跨图像证据的能力。
* **实验设置合理性**：AgMMU 数据集包含 116k 条用户-专家对话，覆盖多种农业任务和图像数量，充分反映真实场景复杂性；基准模型包括专有和开源多模态模型，比较全面；但未讨论计算成本和实时性，可能限制实际应用。
* **结论**：实验表明框架在多图像农业 VQA 任务中实现了显著提升，尤其在动态上下文适应和多模态推理方面优于现有方法。

## Further Thoughts

多智能体协作框架的角色分工和动态迭代机制不仅适用于农业 VQA，还可能扩展到其他多模态推理领域，如医疗诊断或教育问答，是否可以通过强化学习或用户反馈进一步优化自反思机制？此外，多图像整合的思路启发了对信息冲突或冗余的处理策略，例如在视频分析或多传感器数据融合中，如何设计类似的质量控制机制以确保信息综合利用？