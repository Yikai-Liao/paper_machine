---
title: "Stylometry recognizes human and LLM-generated texts in short samples"
pubDatetime: 2025-07-01T15:08:53+00:00
slug: "2025-07-stylometry-llm-detection"
type: "arxiv"
id: "2507.00838"
score: 0.5383821233976122
author: "grok-3-latest"
authors: ["Karol Przystalski", "Jan K. Argasiński", "Iwona Grabska-Gradzińska", "Jeremi Ochab"]
tags: ["LLM", "Stylometry", "Text Classification", "Feature Extraction", "Ethical AI"]
institution: ["Exadel", "Sano - Centre for Computational Medicine", "Faculty of Physics, Astronomy and Applied Computer Science, Jagiellonian University", "Mark Kac Centre for Complex Systems Research, Jagiellonian University"]
description: "本文通过文体学和机器学习方法，成功区分人类和大型语言模型生成的短文本，并利用特征解释性分析揭示其风格差异，为AI伦理使用和内容追踪提供了技术支持。"
---

> **Summary:** 本文通过文体学和机器学习方法，成功区分人类和大型语言模型生成的短文本，并利用特征解释性分析揭示其风格差异，为AI伦理使用和内容追踪提供了技术支持。 

> **Keywords:** LLM, Stylometry, Text Classification, Feature Extraction, Ethical AI

**Authors:** Karol Przystalski, Jan K. Argasiński, Iwona Grabska-Gradzińska, Jeremi Ochab

**Institution(s):** Exadel, Sano - Centre for Computational Medicine, Faculty of Physics, Astronomy and Applied Computer Science, Jagiellonian University, Mark Kac Centre for Complex Systems Research, Jagiellonian University


## Problem Background

随着大型语言模型（LLMs）生成文本的能力日益接近人类水平，区分人类撰写的文本与机器生成的文本变得至关重要。
这一问题涉及模型归属、知识产权保护以及人工智能的伦理使用，例如防止误信息传播或恶意内容生成。
论文旨在探索是否可以通过文体学（Stylometry）方法，基于短文本样本（仅10句话）有效区分人类和LLM生成的文本。

## Method

*   **核心思想：** 利用文体学分析文本的语言风格特征，通过机器学习模型区分人类和LLM生成的文本。
*   **数据集构建：** 基于Wikipedia条目，筛选出2439个符合条件的术语描述（至少1100字符、10句话、无参考文献），并通过多种文本摘要方法（T5, BART, Gensim, Sumy）和LLM（GPT-3.5, GPT-4, LLaMa 2/3, Orca, Falcon）生成对应文本，形成对比数据集。
*   **特征提取：** 使用两种文体学工具提取特征：
    *   StyloMetrix：提供195个特征，覆盖详细的语法形式（时态、情态动词等）、词汇形式（代词、标点等）、句法形式（问句、修辞等）以及文本统计（如类型-词条比）。
    *   CLARIN-PL的文体学流水线：基于n-gram频率特征，包括词条（1-3元）、词性标注、依存关系双元和形态标注等。
*   **分类模型：** 采用树形模型进行分类：
    *   决策树：使用默认参数（如Gini不纯度、最小样本分割为2），用于初步测试。
    *   LightGBM（LGBM）：优化参数（如最大深度5、学习率0.5、启用bagging），用于多分类和二分类任务，通过10折交叉验证评估性能。
*   **解释性分析：** 利用SHAP（Shapley Additive Explanations）方法分析特征重要性，揭示区分人类和LLM文本的关键文体特征，如专有名词频率、标点使用和语法标准化程度。
*   **关键点：** 方法不依赖于特定模型输出，仅通过风格特征实现区分，且注重特征的可解释性，便于理解LLM与人类文本的差异。

## Experiment

*   **有效性：** 在多分类任务中（7类：Wikipedia + 6个LLM），LGBM分类器基于频率特征的Matthews相关系数（MCC）高达0.87，基于StyloMetrix特征为0.72，显示出显著的分类能力；在二分类任务中，准确率在0.79到1.00之间，Wikipedia与GPT-4的二分类准确率高达0.98，表明区分效果极佳。
*   **特征洞察：** SHAP分析揭示Wikipedia文本通常包含更多专有名词和日期，而LLM文本在语法结构上更标准化，常滥用某些词汇（如‘significant’），这些差异是分类成功的关键。
*   **实验设置：** 数据集平衡，采用10折交叉验证，确保结果稳健；但文本类型局限于百科全书式风格（Wikipedia条目），可能限制对其他文本类型的泛化性；此外，实验仅针对英语文本，多语言适用性待验证。
*   **对比分析：** 相较于简单的决策树分类器，LGBM在多分类和二分类任务中均表现出更优的性能，尤其在特征数量较多时（如频率特征达3000个），分类效果更显著。

## Further Thoughts

文体学作为一种‘指纹’技术，不仅适用于传统作者归属，还能适应AI生成内容的检测，这为未来AI治理和内容溯源提供了新思路；LLM文本的语法标准化特性可能成为其‘弱点’，是否可以通过对抗性训练（如故意引入语法多样性）规避检测；特征解释性（SHAP）揭示的细微差异是否可用于开发轻量级实时检测工具；跨语言文体学特征（如句法复杂度而非具体词汇）是否能提升模型泛化能力，值得进一步探索。