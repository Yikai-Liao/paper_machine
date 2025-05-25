---
title: "Training Long-Context LLMs Efficiently via Chunk-wise Optimization"
pubDatetime: 2025-05-22T14:11:34+00:00
slug: "2025-05-long-context-chunk-optimization"
type: "arxiv"
id: "2505.16710"
score: 0.7919562643185737
author: "grok-3-latest"
authors: ["Wenhao Li", "Yuxin Zhang", "Gen Luo", "Daohai Yu", "Rongrong Ji"]
tags: ["LLM", "Long Context", "Gradient Checkpointing", "Sparse Training", "Memory Efficiency"]
institution: ["Key Laboratory of Multimedia Trusted Perception and Efficient Computing, Ministry of Education of China, Xiamen University", "OpenGVLab, Shanghai AI Laboratory"]
description: "本文提出 SeCO 和 SpaCO 两种高效训练长上下文 LLMs 的方法，通过序列维度梯度检查点和稀疏反向传播显著降低内存与计算开销，使单 GPU 上序列长度扩展至 16K，同时保持训练效果。"
---

> **Summary:** 本文提出 SeCO 和 SpaCO 两种高效训练长上下文 LLMs 的方法，通过序列维度梯度检查点和稀疏反向传播显著降低内存与计算开销，使单 GPU 上序列长度扩展至 16K，同时保持训练效果。 

> **Keywords:** LLM, Long Context, Gradient Checkpointing, Sparse Training, Memory Efficiency

**Authors:** Wenhao Li, Yuxin Zhang, Gen Luo, Daohai Yu, Rongrong Ji

**Institution(s):** Key Laboratory of Multimedia Trusted Perception and Efficient Computing, Ministry of Education of China, Xiamen University, OpenGVLab, Shanghai AI Laboratory


## Problem Background

长上下文大型语言模型（LLMs）在处理长文档时表现出色，但在资源受限环境下训练和微调面临重大挑战：注意力机制的二次方复杂度导致训练时间随序列长度急剧增加，前向激活存储需求随序列长度线性增长，使得在单张 RTX 3090 GPU 上微调 8B 模型时序列长度受限于 1K 左右；现有方法如 LongLoRA 通过注意力近似减少计算开销，但牺牲梯度精度且内存节省有限，因此需要一种高效训练方法以支持长上下文模型在资源受限环境下的应用。

## Method

* **Sequential Chunk-wise Optimization (SeCO):**  
  - **核心思想:** 将长输入序列分割成小块（chunks），对每个 chunk 独立构建计算图并进行局部反向传播，通过沿序列维度应用梯度检查点（gradient checkpointing），确保任意时刻仅存储一个 chunk 的前向激活，使内存开销与序列长度无关。  
  - **实现细节:** 在前向传播时以推理模式计算并存储所有 chunk 的 KV 缓存作为检查点；在反向传播时按逆序逐个重建每个 chunk 的计算图，计算误差并累积梯度至模型参数和前序检查点；这种设计避免了传统层级检查点方法中内存随序列长度线性增长的问题，同时保持精确梯度计算。  
* **Sparse Chunk-wise Optimization (SpaCO):**  
  - **核心思想:** 在 SeCO 基础上通过稀疏化反向传播减少计算开销，即在每次训练迭代中随机选择一小部分 chunk 进行梯度计算，使训练时间随序列长度增长趋近推理时间。  
  - **实现细节:** 基于 Transformer 架构中梯度链长度受限于模型层数的特性，SpaCO 随机采样固定数量的 chunk（如 t=8）进行反向传播；为避免稀疏化导致的梯度估计偏差，引入补偿因子（compensation factor），根据稀疏采样概率（如 k/t，其中 k 为总 chunk 数，t 为采样数）对保留路径的梯度进行缩放，确保无偏估计；补偿因子在实现中通过修改反向传播梯度累积实现自动化缩放。  
* **共同特点:** 两种方法均作为轻量级训练包装器实现，无需修改模型架构，易于集成到现有框架，适用于资源受限环境下的长上下文模型微调。

## Experiment

* **内存效率:** SeCO 和 SpaCO 均显著降低内存占用，使单张 RTX 3090 GPU 上微调 LLaMA3-8B 模型的序列长度从 1K 扩展至 16K，相比朴素并行训练实现约 16 倍内存节省，相比标准梯度检查点方法也有 4 倍以上改进。  
* **时间效率:** SeCO 相比朴素并行训练增加约 30% 计算开销（源于反向传播重计算和频繁内核启动），但远优于 DeepSpeed ZeRO3 offload（后者速度慢 10 倍）；SpaCO 通过稀疏反向传播显著降低计算开销，最高比 SeCO 快 3 倍，且随序列长度增长训练时间接近推理时间。  
* **性能表现:** SeCO 计算精确梯度，与基线（模型并行结合梯度检查点）性能差距极小（语言建模误差差距在 0.02 以内）；SpaCO 因稀疏化引入梯度估计方差，性能略低于 SeCO，但在稀疏比率为 1/8 时误差增加不到 0.1，通过调整学习率和补偿因子上限（设为 2）可接近基线收敛性。  
* **实验设置合理性:** 实验基于 LLaMA3-8B 和 LoRA 微调，在单 GPU 上测试多种序列长度、chunk 大小和稀疏预算，与多个主流方法对比，设置全面且贴近资源受限场景；不足之处在于未验证多 GPU 环境下的扩展性。

## Further Thoughts

论文提出沿序列维度应用梯度检查点的思路启发我们思考是否可以结合其他维度（如批次维度）进一步优化内存效率；SpaCO 的稀疏反向传播和补偿因子机制为高效训练提供了新视角，是否可动态调整稀疏预算或基于梯度重要性设计补偿策略以平衡精度与效率；此外，SpaCO 使训练时间趋近推理时间的特性提示我们是否能结合推理优化技术（如 vLLM 的 chunk pre-filling）进一步缩小训练-推理差距；最后，方法在消费级硬件上的成功应用启发我们探索其在边缘设备或移动端训练中的潜力，是否可通过量化或混合精度进一步降低资源需求。