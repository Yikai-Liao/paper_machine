---
title: "FreeMesh: Boosting Mesh Generation with Coordinates Merging"
pubDatetime: 2025-05-19T13:52:57+00:00
slug: "2025-05-freemesh-coordinate-merging"
type: "arxiv"
id: "2505.13573"
score: 0.4814063348134726
author: "grok-3-latest"
authors: ["Jian Liu", "Haohan Weng", "Biwen Lei", "Xianghui Yang", "Zibo Zhao", "Zhuo Chen", "Song Guo", "Tao Han", "Chunchao Guo"]
tags: ["Mesh Generation", "Tokenization", "Compression", "Autoregressive Model", "Entropy"]
institution: ["Hong Kong University of Science and Technology", "Tencent Hunyuan", "South China University of Technology", "ShanghaiTech University"]
description: "本文提出 Per-Token-Mesh-Entropy (PTME) 指标和 Coordinate Merging 技术，为自回归网格生成提供理论评估框架和序列化优化方法，显著提升生成效率与质量。"
---

> **Summary:** 本文提出 Per-Token-Mesh-Entropy (PTME) 指标和 Coordinate Merging 技术，为自回归网格生成提供理论评估框架和序列化优化方法，显著提升生成效率与质量。 

> **Keywords:** Mesh Generation, Tokenization, Compression, Autoregressive Model, Entropy

**Authors:** Jian Liu, Haohan Weng, Biwen Lei, Xianghui Yang, Zibo Zhao, Zhuo Chen, Song Guo, Tao Han, Chunchao Guo

**Institution(s):** Hong Kong University of Science and Technology, Tencent Hunyuan, South China University of Technology, ShanghaiTech University


## Problem Background

在 3D 网格生成领域，自回归方法通过将网格数据序列化后进行生成已成为主流，但缺乏有效的理论指标来评估不同序列化方法（Tokenizer）的质量。
现有的评估依赖于耗时的模型训练和实验结果观察，存在随机性和资源浪费问题。
论文旨在解决如何在无训练情况下理论评估网格序列化方法的优劣，并优化序列化过程以提升生成效率和质量。

## Method

*   **核心思想:** 提出一个理论指标 Per-Token-Mesh-Entropy (PTME) 来评估网格序列化方法的质量，并通过 Coordinate Merging 技术优化序列化以降低 PTME 和压缩率。
*   **PTME 指标:** PTME 结合信息熵（Entropy）和压缩率（Compression Ratio），衡量序列的学习难易程度，较低的 PTME 表示序列更易于自回归模型学习，无需训练即可评估 Tokenizer 质量。
*   **Coordinate Merging 技术:** 借鉴自然语言处理中的 Byte-Pair Encoding (BPE) 算法，通过合并高频坐标模式减少序列冗余，具体分为两种实现：
    *   **Merge Coordinates (MC):** 基础方法，直接统计并合并相邻坐标对，构建新词汇表，但未考虑坐标轴间的空间关系，效果有限。
    *   **Rearrange & Merge Coordinates (RMC):** 改进方法，先对坐标序列进行重组（如按 x, y, z 轴分组处理），再应用 BPE 合并高频模式，显著降低序列长度和 PTME。
*   **实现细节:** 使用 SentencePiece 工具训练合并规则，针对不同基线 Tokenizer（如 MeshXL, MeshAnything V2, EdgeRunner）进行适配，词汇表大小设为 8192 以平衡压缩效果和复杂度。
*   **关键优势:** 方法为即插即用型，可无缝集成到现有自回归网格生成框架中，仅通过序列化优化即可提升生成能力。

## Experiment

*   **有效性验证:** PTME 指标与生成质量（如 Chamfer Distance, Hausdorff Distance）高度相关（Pearson 相关系数 r=0.965），证明其作为无训练评估工具的可靠性，优于传统 Perplexity 指标。
*   **方法提升:** Coordinate Merging 技术显著降低压缩率和 PTME，尤其 RMC 结合 EdgeRunner 实现 21.2% 压缩率，Chamfer Distance 从 0.198 降至 0.123，生成质量提升约 37.9%。
*   **对比分析:** 相比基础 MC 方法，RMC 通过坐标重组克服了跨轴合并的局限性，效果更优；与基线 Tokenizer 相比，RMC 增强版本在所有面数（500-4000 面）场景下均提升了拓扑质量和生成稳定性。
*   **实验设置合理性:** 实验基于 ShapeNetV2, Objaverse 等数据集，覆盖低到高多边形网格生成任务，控制模型架构一致，仅改变 Tokenizer 确保公平对比；但论文指出方法在高量化精度（如 1024 级）下效果可能减弱，提示适用范围。
*   **计算开销:** Coordinate Merging 训练过程轻量（CPU 环境下 1 小时内完成），推理时仅增加少量序列处理负担，整体高效。

## Further Thoughts

论文启发我思考信息论在结构化数据生成中的更广泛应用，PTME 指标可能扩展到图像或视频生成任务中作为无训练评估工具；此外，NLP 技术（如 BPE）向 3D 领域的迁移表明跨领域方法融合的潜力，未来是否可引入更多 NLP 优化（如注意力机制改进）以提升网格生成效率？同时，动态合并策略的探索可能解决高精度场景下的压缩难题，值得进一步研究。