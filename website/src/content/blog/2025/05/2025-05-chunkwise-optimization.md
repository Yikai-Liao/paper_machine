---
title: "Training Long-Context LLMs Efficiently via Chunk-wise Optimization"
pubDatetime: 2025-05-22T14:11:34+00:00
slug: "2025-05-chunkwise-optimization"
type: "arxiv"
id: "2505.16710"
score: 0.7919562643185737
author: "grok-3-latest"
authors: ["Wenhao Li", "Yuxin Zhang", "Gen Luo", "Daohai Yu", "Rongrong Ji"]
tags: ["LLM", "Long Context", "Training Efficiency", "Gradient Checkpointing", "Sparse Optimization"]
institution: ["Key Laboratory of Multimedia Trusted Perception and Efficient Computing, Xiamen University", "OpenGVLab, Shanghai AI Laboratory"]
description: "本文提出 SeCO 和 SpaCO 两种训练范式，通过序列分块优化和稀疏反向传播显著降低长上下文 LLM 训练的内存和计算开销，在单 GPU 上实现 16K 令牌微调，同时维持性能。"
---

> **Summary:** 本文提出 SeCO 和 SpaCO 两种训练范式，通过序列分块优化和稀疏反向传播显著降低长上下文 LLM 训练的内存和计算开销，在单 GPU 上实现 16K 令牌微调，同时维持性能。 

> **Keywords:** LLM, Long Context, Training Efficiency, Gradient Checkpointing, Sparse Optimization

**Authors:** Wenhao Li, Yuxin Zhang, Gen Luo, Daohai Yu, Rongrong Ji

**Institution(s):** Key Laboratory of Multimedia Trusted Perception and Efficient Computing, Xiamen University, OpenGVLab, Shanghai AI Laboratory


## Problem Background

长上下文大型语言模型（LLMs）在处理长文档时表现出色，但在训练和微调过程中面临显著的资源挑战：注意力机制的二次方复杂度导致训练时间随序列长度急剧增加，前向激活存储需求随序列长度线性增长，使得在消费级硬件（如单张 RTX 3090 GPU）上微调 8B 模型时，序列长度受限于 1K 令牌；现有方法如 LongLoRA 通过注意力近似降低计算开销，但牺牲了梯度精度且内存节省有限。

## Method

* **Sequential Chunk-wise Optimization (SeCO):**  
  - **核心思想：** 将长输入序列分割成小块（chunks），对每个 chunk 独立构建计算图并执行局部反向传播，确保在任意时刻仅存储一个 chunk 的前向激活。  
  - **实现方式：** 在序列维度上应用梯度检查点（gradient checkpointing），前向传播时顺序计算并存储每个 chunk 的 KV 缓存（作为检查点），反向传播时按逆序逐个重建 chunk 的计算图并累积梯度。  
  - **优势：** 内存占用与序列长度解耦，实现了数量级的内存节省，同时保持精确梯度计算，无需修改模型架构。  
* **Sparse Chunk-wise Optimization (SpaCO):**  
  - **核心思想：** 在 SeCO 基础上，通过稀疏化反向传播降低计算开销，在每次训练迭代中随机选择一小部分 chunk 进行梯度计算。  
  - **实现方式：** 利用 Transformer 架构中梯度链长度受限于模型层数的特性，随机采样固定数量的 chunk（如 8 个），并引入补偿因子（compensation factor）对梯度进行缩放，确保无偏估计；补偿因子基于稀疏采样概率（k/t，其中 k 为总 chunk 数，t 为采样数），对每个梯度链按长度累积缩放。  
  - **优势：** 计算成本与序列长度解耦，训练时间随序列长度增长趋于线性，接近推理时间，同时维持内存效率。

## Experiment

* **内存效率：** SeCO 和 SpaCO 在单张 RTX 3090 GPU 上将 8B 模型的微调序列长度从 1K 扩展至 16K，相比标准梯度检查点方法节省 4 倍内存，相比朴素并行训练节省 16 倍内存。  
* **时间效率：** SeCO 引入约 30% 的额外计算开销（因反向传播时重新计算），但优于 DeepSpeed ZeRO3 offload（后者因 GPU-CPU 通信慢 10 倍）；SpaCO 通过稀疏化显著降低计算开销，训练时间随序列长度增长呈线性趋势，接近推理时间，在相同设置下比 SeCO 快 3 倍。  
* **性能表现：** SeCO 计算精确梯度，性能与基线（模型并行+梯度检查点）几乎一致；SpaCO 因稀疏化引入梯度估计方差，但在稀疏比 1/8 时语言建模误差仅比精确梯度训练高 0.1，通过学习率调整（如 1e-3）可接近基线性能。  
* **实验设置合理性：** 实验基于 LLaMA3-8B 模型和 PG19 数据集（1000 样本，截断至 16K 令牌），对比了 DeepSpeed、标准梯度检查点和朴素并行训练，全面评估了时间、内存和性能三个维度，结果支持方法的有效性。

## Further Thoughts

本文在序列维度上应用梯度检查点的创新启发我们可以在数据维度上探索更多优化空间，例如在多模态模型中对图像或音频数据分块处理；SpaCO 的稀疏化与无偏估计结合表明，通过架构特性和数学推导可以在降低计算成本的同时维持训练效果，这种思路可推广至其他深度学习任务如图神经网络训练；此外，方法作为轻量级包装器降低了长上下文模型训练的硬件门槛，提示我们在算法设计中应更多考虑消费级硬件的可及性，推动 AI 民主化。