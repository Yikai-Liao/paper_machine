---
title: "Benchmarking Universal Interatomic Potentials on Zeolite Structures"
pubDatetime: 2025-09-09T06:04:40+00:00
slug: "2025-09-universal-potentials-zeolite"
type: "arxiv"
id: "2509.07417"
score: 0.8493742464839567
author: "grok-3-latest"
authors: ["Shusuke Ito", "Koki Muraoka", "Akira Nakayama"]
tags: ["Machine Learning", "Interatomic Potential", "Zeolite Structure", "High-Throughput Screening", "Density Functional Theory"]
institution: ["The University of Tokyo"]
description: "本文通过对沸石结构的基准测试，揭示了通用机器学习原子间势（MLIPs）在几何和能量预测上的优越性与普适性，为高通量材料发现提供了实用工具。"
---

> **Summary:** 本文通过对沸石结构的基准测试，揭示了通用机器学习原子间势（MLIPs）在几何和能量预测上的优越性与普适性，为高通量材料发现提供了实用工具。 

> **Keywords:** Machine Learning, Interatomic Potential, Zeolite Structure, High-Throughput Screening, Density Functional Theory

**Authors:** Shusuke Ito, Koki Muraoka, Akira Nakayama

**Institution(s):** The University of Tokyo


## Problem Background

在材料科学中，密度泛函理论（DFT）计算大型结构（如沸石）时成本高昂，限制了高通量材料发现的应用，而传统经验性原子间势（IPs）缺乏普适性，无法适应多样化化学环境；近年来发展的通用原子间势（Universal IPs）和机器学习势（MLIPs）覆盖广泛元素，但其在沸石结构上的适用性尚待验证，因此本文旨在评估这些通用势是否能替代DFT或定制势用于高效材料筛选。

## Method

* **核心思想**：通过基准测试系统比较通用原子间势（包括分析势和机器学习势）在沸石结构上的表现，评估其几何和能量预测精度，以验证其在高通量材料发现中的潜力。
* **具体实现**：
  * **测试对象**：选取纯硅沸石、含铝硅酸盐沸石以及包含铜、钾和有机阳离子的复杂沸石结构作为测试集，数据来源于实验和国际沸石协会数据库。
  * **测试方法**：对比两类通用势——分析势（如GFN-FF, UFF, Dreiding）和预训练机器学习势（MLIPs，如CHGNet, ORB-v3, MatterSim, eSEN-30M-OAM, PFP-v7, EquiformerV2），并与定制势（如SLC, ClayFF, BSFF）和DFT计算（PBE泛函加D3色散校正）进行比较。
  * **计算工具**：使用GULP软件对分析势进行结构优化（采用Newton-Raphson优化器结合BFGS更新），使用ASE中的FIRE算法优化MLIPs结构，评估指标包括键长、键角的几何误差和相对能量的偏差（以实验数据或DFT结果为参考）。
* **关键点**：通过多维度测试（结构多样性和化学多样性），全面评估通用势的普适性和精度，揭示其在不同化学环境下的表现差异。

## Experiment

* **有效性**：通用分析势中GFN-FF表现最佳，但对高应变硅环和含客体阳离子的沸石结构预测不准确，结构畸变严重，能量偏差较大（RMSE较高）；通用MLIPs整体表现优异，几何和能量预测接近DFT结果，其中eSEN-30M-OAM模型在所有测试结构中表现最一致，RMSE最低（如对Cu/CHA为0.14 kJ/mol/atom）。
* **优越性**：与定制势相比，通用MLIPs在处理化学多样性（如含客体阳离子的沸石）时表现出更好的普适性，而定制势（如SLC）仅在纯硅沸石上表现最佳；相比DFT，MLIPs显著降低了计算成本，同时保持了接近DFT的精度。
* **合理性与局限**：实验设置全面，涵盖了纯硅沸石和复杂沸石结构，并以实验数据和DFT为参考，但MLIPs训练数据多基于DFT，可能继承其系统误差，且未充分测试化学键多样性（如键形成与断裂）。
* **数据支持**：表2和表4显示eSEN模型在能量预测上的RMSE显著低于其他势，图1和图4表明MLIPs在几何和能量分布上与DFT高度一致。

## Further Thoughts

通用MLIPs的成功依赖于DFT训练数据，但DFT存在系统误差，未来是否可以通过更高精度的量子化学方法（如CCSD(T)或改进的SCAN泛函）作为训练数据来源，进一步提升MLIPs精度？此外，是否可以扩展测试到更多反应性环境（如催化反应中间态），验证MLIPs在动态过程中的适用性？另一个方向是设计混合策略，将定制势的高精度与通用势的普适性结合，通过自适应选择势函数优化计算效率和精度。