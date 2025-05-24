---
title: "Advancing the Scientific Method with Large Language Models: From Hypothesis to Discovery"
pubDatetime: 2025-05-22T10:05:48+00:00
slug: "2025-05-llm-scientific-discovery"
type: "arxiv"
id: "2505.16477"
score: 0.573464978501005
author: "grok-3-latest"
authors: ["Yanbo Zhang", "Sumeer A. Khan", "Adnan Mahmud", "Huck Yang", "Alexander Lavin", "Michael Levin", "Jeremy Frey", "Jared Dunnmon", "James Evans", "Alan Bundy", "Saso Dzeroski", "Jesper Tegner", "Hector Zenil"]
tags: ["LLM", "Scientific Discovery", "Hypothesis Generation", "Automation", "Reasoning"]
institution: ["Allen Discovery Center at Tufts University", "Living Systems Lab, KAUST", "Intelligent Infrastructure Team, Network Rail", "Department for AI in Society, Science, and Technology, Zuse Institute Berlin", "NVIDIA Research", "Biological and Environmental Science and Engineering Division, KAUST", "SDAIA-KAUST Center of Excellence in Data Science and Artificial Intelligence", "Pasteur Labs", "Wyss Institute for Biologically Inspired Engineering, Harvard University", "Department of Chemistry, University of Southampton", "Department of Biomedical Data Science, Stanford University", "Knowledge Lab, Department of Sociology, University of Chicago", "Santa Fe Institute", "School of Informatics, The University of Edinburgh", "Department of Knowledge Technologies, Jožef Stefan Institute", "Unit of Computational Medicine, Karolinska Institutet", "Computer, Electrical and Mathematical Sciences and Engineering Division, KAUST", "Science for Life Laboratory", "Algorithmic Dynamics Lab, King’s College London", "King’s Institute for Artificial Intelligence", "The Alan Turing Institute", "Oxford Immune Algorithmics", "Cancer Interest Group, Francis Crick Institute"]
description: "本文系统综述了大型语言模型（LLMs）在科学方法中的应用现状与潜力，分析了其局限性，并提出通过深度整合和人类监督将其转变为科学发现的创造引擎。"
---

> **Summary:** 本文系统综述了大型语言模型（LLMs）在科学方法中的应用现状与潜力，分析了其局限性，并提出通过深度整合和人类监督将其转变为科学发现的创造引擎。 

> **Keywords:** LLM, Scientific Discovery, Hypothesis Generation, Automation, Reasoning

**Authors:** Yanbo Zhang, Sumeer A. Khan, Adnan Mahmud, Huck Yang, Alexander Lavin, Michael Levin, Jeremy Frey, Jared Dunnmon, James Evans, Alan Bundy, Saso Dzeroski, Jesper Tegner, Hector Zenil

**Institution(s):** Allen Discovery Center at Tufts University, Living Systems Lab, KAUST, Intelligent Infrastructure Team, Network Rail, Department for AI in Society, Science, and Technology, Zuse Institute Berlin, NVIDIA Research, Biological and Environmental Science and Engineering Division, KAUST, SDAIA-KAUST Center of Excellence in Data Science and Artificial Intelligence, Pasteur Labs, Wyss Institute for Biologically Inspired Engineering, Harvard University, Department of Chemistry, University of Southampton, Department of Biomedical Data Science, Stanford University, Knowledge Lab, Department of Sociology, University of Chicago, Santa Fe Institute, School of Informatics, The University of Edinburgh, Department of Knowledge Technologies, Jožef Stefan Institute, Unit of Computational Medicine, Karolinska Institutet, Computer, Electrical and Mathematical Sciences and Engineering Division, KAUST, Science for Life Laboratory, Algorithmic Dynamics Lab, King’s College London, King’s Institute for Artificial Intelligence, The Alan Turing Institute, Oxford Immune Algorithmics, Cancer Interest Group, Francis Crick Institute


## Problem Background

大型语言模型（LLMs）正在通过提升生产力和支持科学方法的各个阶段（如文献综述、实验设计、数据分析）来改变科学研究。然而，其在基础科学发现（即发现新原理或科学定律）中的作用仍受限于幻觉、推理能力不足和可解释性问题。论文探讨如何将 LLMs 从技术工具转变为‘创造引擎’，以支持科学发现的全过程，并在人类科学目标的指导下实现深度整合。

## Method

* **观察阶段**：利用 LLMs 进行数据标注、分类和信息提取，特别是在自然语言处理和社会科学领域，通过通用模型（如 GPT-4）和领域特定模型（如 BioGPT）处理多模态数据，支持科学家从复杂数据中提取关键信息。
* **假设生成阶段**：通过提示工程（如 Chain-of-Thought, CoT）和检索增强生成（RAG）方法，LLMs 基于现有文献和观察数据提出新假设，利用其存储的压缩知识和互联网信息检索能力，加速文献综述和跨学科知识整合。
* **实验阶段**：LLMs 通过工具调用、代码生成和实验规划（如 ReAct 方法）支持实验设计和执行，例如生成结构化输出（如 JSON 格式）调用外部函数，或直接编写代码实现复杂实验控制，同时通过规划能力分解任务并动态调整计划。
* **自动化与循环**：通过 LLM 代理（agents）和多代理系统实现科学发现的自动化循环，包括假设-实验-观察的迭代过程，利用反馈机制（如 Reflexion）和多轮验证提高结果可靠性，同时支持大规模实验扩展和知识积累。
* **基础模型应用**：讨论了基础模型（Foundation Models）在科学领域的应用，如 ChemBERT、scGPT 等，通过大规模预训练和领域特定微调支持跨任务的科学应用，处理多模态数据并实现零样本预测和生成任务。

## Experiment

* **有效性**：论文引用了大量现有研究结果，表明 LLMs 在特定任务上效果显著，例如在文本标注任务中优于人类（如 ChatGPT 相较于众包工人），领域特定模型（如 BioGPT、ChemBERT）在专业领域任务中表现优于通用模型，自动化实验设计工具（如 CRISPR-GPT）显著提高了基因编辑实验效率。
* **局限性**：由于是综述性论文，未提供新的实验数据，实验设置的全面性和合理性依赖于引用的研究，论文指出 LLMs 在开放性任务和基础科学发现中的应用仍需进一步验证，尤其是在解决幻觉和推理缺陷方面。
* **合理性与全面性**：引用的研究覆盖了多个科学领域（如生物学、化学、物理学）和科学方法阶段（观察、假设、实验），但缺乏对 LLMs 在不同领域应用效果的直接对比分析，未来需要更系统的实验设计来评估其在基础科学中的潜力。

## Further Thoughts

论文提出将 LLMs 的幻觉作为创造力来源，用于生成新颖假设并通过验证机制筛选有价值的想法，这一反直觉视角启发我们重新审视 AI 的‘缺陷’在科学创新中的潜在价值；此外，‘算法置信度’的概念为评估 AI 在科学任务中的可靠性提供了新思路，未来可探索如何结合多维度指标（如模型、任务、工具使用）构建预测性信任框架；最后，论文提到的‘答案生成新问题’的对称性机制（如 AlphaGo）若能应用于 LLMs，可能显著提升其在科学发现中的自主探索能力，值得进一步研究。