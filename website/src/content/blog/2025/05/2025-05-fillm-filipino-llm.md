---
title: "FiLLM -- A Filipino-optimized Large Language Model based on Southeast Asia Large Language Model (SEALLM)"
pubDatetime: 2025-05-25T06:36:26+00:00
slug: "2025-05-fillm-filipino-llm"
type: "arxiv"
id: "2505.18995"
score: 0.6907401481656071
author: "grok-3-latest"
authors: ["Carlos Jude G. Maminta", "Isaiah Job Enriquez", "Michael B. Dela Fuente", "Deandre Nigel Nuñez"]
tags: ["LLM", "Low-Resource Language", "Fine-Tuning", "NLP Tasks", "Language Adaptation"]
institution: ["Polytechnic University of the Philippines"]
description: "本文提出FiLLM，一个基于SeaLLM-7B 2.5并通过LoRA微调的菲律宾语优化大型语言模型，在资源受限环境下实现了对关键NLP任务的有效支持，尽管与基准模型相比性能仍有差距。"
---

> **Summary:** 本文提出FiLLM，一个基于SeaLLM-7B 2.5并通过LoRA微调的菲律宾语优化大型语言模型，在资源受限环境下实现了对关键NLP任务的有效支持，尽管与基准模型相比性能仍有差距。 

> **Keywords:** LLM, Low-Resource Language, Fine-Tuning, NLP Tasks, Language Adaptation

**Authors:** Carlos Jude G. Maminta, Isaiah Job Enriquez, Michael B. Dela Fuente, Deandre Nigel Nuñez

**Institution(s):** Polytechnic University of the Philippines


## Problem Background

菲律宾作为一个拥有超过175种语言的语言多样性国家，针对菲律宾语（Filipino）的自然语言处理（NLP）和大型语言模型（LLM）开发受到标注数据集稀缺和计算资源不足的限制，导致本地化AI应用发展滞后。
本研究旨在开发一个针对菲律宾语优化的高效语言模型FiLLM，以支持关键NLP任务（如命名实体识别、词性标注、依存句法分析和文本摘要），从而填补菲律宾语NLP研究的空白。

## Method

*   **核心思想:** 基于东南亚大型语言模型SeaLLM-7B 2.5，通过低秩适应（Low-Rank Adaptation, LoRA）微调方法，开发一个针对菲律宾语优化的模型FiLLM，以在资源受限环境下实现高效的语言处理能力。
*   **具体实现:** 
    *   使用SeaLLM-7B 2.5作为预训练基础模型，冻结其原始权重（W），以保留预训练知识。
    *   引入低秩参数矩阵（LoRA A和LoRA B），通过对这些参数的微调，使模型适应菲律宾语的语言特性和特定NLP任务，而不影响原始模型的权重。
    *   训练数据包括多个菲律宾语数据集（如Filipino Hatespeech Dataset、Filipino Dengue Dataset、TLUnified-NER等），覆盖命名实体识别（NER）、词性标注（POS）、依存句法分析和文本摘要等任务。
    *   使用Transformers、Datasets和PyTorch等工具进行模型训练，注重参数效率和内存优化。
*   **关键优势:** LoRA方法显著降低了计算和内存需求，适合低资源环境，同时通过任务特定的微调提升了模型对菲律宾语的适应性。

## Experiment

*   **性能表现:** FiLLM在命名实体识别（NER）和词性标注（POS）任务上表现较好，F1分数均为0.89，显示出较强的能力；但在依存句法分析任务上表现较弱，F1分数为0.73，表明处理复杂句法关系的能力有待提升；在文本摘要任务中，模型在高压缩率下仍能保留关键信息，但性能因文本复杂性而异。
*   **对比分析:** 与基准模型CalamanCy相比，FiLLM在所有任务上表现稍逊（NER: 0.89 vs. 0.90；POS: 0.89 vs. 0.97；依存句法分析: 0.73 vs. 0.97），且t检验显示性能差异具有统计显著性（p=0.03<0.05）。
*   **实验设置:** 实验覆盖了多个NLP任务和菲律宾语数据集，采用80-20的训练-测试分割，设置较为全面合理；但未详细说明数据集的标注质量和分布均衡性，可能对结果可靠性产生影响。
*   **资源开销:** LoRA微调方法有效降低了计算和内存需求，适合低资源环境，但性能提升空间有限。

## Further Thoughts

本研究启发了我对低资源语言模型开发的思考：LoRA微调方法在资源受限环境下的潜力巨大，但如何选择合适的预训练模型（如SeaLLM是否完全适配菲律宾语）值得进一步探索；此外，数据集质量和多样性对性能影响显著，未来可通过数据增强或跨语言迁移学习弥补数据不足；同时，是否可以设计更轻量级的模型架构或任务特定的优化策略，以进一步降低计算成本并提升性能？