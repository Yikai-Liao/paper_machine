---
title: "Chat to Chip: Large Language Model Based Design of Arbitrarily Shaped Metasurfaces"
pubDatetime: 2025-09-29T02:24:57+00:00
slug: "2025-09-llm-metasurface-design"
type: "arxiv"
id: "2509.24196"
score: 0.6385491862998569
author: "grok-3-latest"
authors: ["Huanshu Zhang", "Sawyer D. Campbell", "Lei Kang", "Douglas H. Werner"]
tags: ["LLM", "Metasurface Design", "Inverse Design", "Fine-Tuning", "Optical Prediction"]
institution: ["The Pennsylvania State University"]
description: "本文提出基于大型语言模型的‘chat-to-chip’工作流程，通过自然语言交互和微调实现任意形状超表面的快速正向预测和逆向设计，显著降低纳米光子学设计的计算成本和技术门槛。"
---

> **Summary:** 本文提出基于大型语言模型的‘chat-to-chip’工作流程，通过自然语言交互和微调实现任意形状超表面的快速正向预测和逆向设计，显著降低纳米光子学设计的计算成本和技术门槛。 

> **Keywords:** LLM, Metasurface Design, Inverse Design, Fine-Tuning, Optical Prediction

**Authors:** Huanshu Zhang, Sawyer D. Campbell, Lei Kang, Douglas H. Werner

**Institution(s):** The Pennsylvania State University


## Problem Background

传统超表面设计依赖全波电磁仿真，计算成本极高，限制了复杂几何形状和大规模设计的探索；尽管数据驱动的深度神经网络（DNN）方法提高了效率，但每次新任务仍需重新设计网络和调整超参数，技术门槛较高；本文提出利用大型语言模型（LLM）通过自然语言交互和简单微调，加速任意形状超表面的正向预测和逆向设计，旨在降低纳米光子学研究者的设计负担。

## Method

* **核心思想**：将超表面设计问题转化为自然语言序列预测任务，利用预训练的大型语言模型（LLM）通过微调实现快速正向预测和逆向设计，无需重新设计网络结构。
* **数据准备**：生成包含45,790个任意形状超表面设计的数据集，通过随机4×4控制点网格、四重旋转对称性、插值和二值化生成几何形状，并使用Lumerical FDTD仿真其传输光谱（波长范围1050-1600nm）。
* **正向预测**：将控制点网格编码为自然语言提示（如‘4×4网格为[[...]]，传输光谱是多少？’），目标输出为31点光谱值；使用Meta-Llama-3.1-8B-Instruct模型，通过LoRA（Low-Rank Adaptation）进行参数高效微调，优化交叉熵损失以提升预测精度。
* **逆向设计**：将目标光谱编码为提示（如‘产生以下光谱的网格是什么？’），利用LLM的随机性生成多样化的控制点网格，解决逆向设计的多对一问题。
* **技术实现**：模型在单张NVIDIA RTX 2080 Ti GPU上训练，微调8个epoch，使用AdamW优化器，内存占用约10GB，推理时间为秒级。
* **关键创新**：通过自然语言交互降低技术门槛，1D token-wise LLM成功处理2D几何设计问题，且无需视觉模型支持。

## Experiment

* **正向预测效果**：微调后的Llama-3.1-8B模型在测试集（9,158个样本）上的均方误差（MSE）为3.4×10⁻³，与文献中定制DNN相当，预测时间约2秒，比全波仿真快60倍；微调epoch数（5-20）对精度影响较小，表明方法鲁棒性强。
* **逆向设计效果**：LLM生成的网格对应的光谱与目标高度吻合（MSE低至10⁻⁷级别），且几何形状多样，优于传统串联网络，避免了模式崩塌问题。
* **基准测试**：对11个开源LLM（参数规模0.1B-72B）测试发现，中型模型（7-9B）如Gemma在精度和成本间平衡最佳，参数规模增加到一定程度后收益递减（如Qwen-2.5从7B到72B仅降低1.2×10⁻³ MSE，但推理时间增至35秒）。
* **实验设置评价**：数据集规模较大，仿真条件明确，涵盖多种任意形状超表面，设计较为全面；但未深入探讨不同材料或入射条件下的泛化性，可能存在局限。
* **总体结论**：方法在速度、精度和易用性上显著优于传统仿真和定制DNN，验证了‘chat-to-chip’工作流程的实用性。

## Further Thoughts

论文展示了LLM在非语言领域（如物理建模）的潜力，启发我们将其他物理问题（如热传导、流体力学）编码为语言任务，利用LLM加速多物理场设计；LLM的随机性在逆向设计中解决多对一问题的思路，可通过控制采样策略进一步优化多样性和精度；此外，基准测试表明模型架构和分词器设计可能比参数规模更关键，提示我们在模型选择时应注重任务适配性。