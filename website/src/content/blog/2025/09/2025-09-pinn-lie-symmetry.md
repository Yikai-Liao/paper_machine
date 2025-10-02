---
title: "Enhancing PINN Performance Through Lie Symmetry Group"
pubDatetime: 2025-09-30T11:30:46+00:00
slug: "2025-09-pinn-lie-symmetry"
type: "arxiv"
id: "2509.26113"
score: 0.743425001521375
author: "grok-3-latest"
authors: ["Ali Haider Shah", "Naveed R. Butt", "Asif Ahmad", "Muhammad Omer Bin Saeed"]
tags: ["PINN", "Lie Symmetry", "PDE Solving", "Neural Network", "Adaptive Learning"]
institution: ["Ghulam Ishaq Khan Institute of Engineering Sciences and Technology"]
description: "本文通过将 Lie 对称群和自适应激活函数融入 PINNs 框架，显著提升了非线性偏微分方程求解的精度和效率。"
---

> **Summary:** 本文通过将 Lie 对称群和自适应激活函数融入 PINNs 框架，显著提升了非线性偏微分方程求解的精度和效率。 

> **Keywords:** PINN, Lie Symmetry, PDE Solving, Neural Network, Adaptive Learning

**Authors:** Ali Haider Shah, Naveed R. Butt, Asif Ahmad, Muhammad Omer Bin Saeed

**Institution(s):** Ghulam Ishaq Khan Institute of Engineering Sciences and Technology


## Problem Background

偏微分方程（PDEs）在科学计算中广泛用于描述物理、化学和生物学等领域的复杂现象，但传统数值方法求解成本高且对非线性问题处理困难。
物理信息神经网络（PINNs）作为一种数据驱动的深度学习方法虽有潜力，但在复杂 PDEs 求解中精度和效率仍需提升，特别是在非线性问题如 Burgers 方程上存在较大偏差。
本文旨在解决 PINNs 在求解 PDEs 时的准确性和计算效率问题。

## Method

*   **核心思想:** 将 Lie 对称群的数学结构融入物理信息神经网络（PINNs），利用 PDEs 的内在对称性约束神经网络训练，提升求解精度和效率。
*   **具体实现:** 
    *   **Lie 对称群嵌入:** 通过 Lie 群的无穷小变换（infinitesimal transformations）和无穷小生成子（infinitesimal generators），分析 PDEs 的对称性，并将对称性信息嵌入 PINNs 的损失函数，形成 modified Symmetry-based PINN (m-SPINN)。具体方法是对配点（collocation points）进行 Lie 群变换，计算额外的对称性残差项（L_symm），并将其加入总损失函数。
    *   **自适应优化:** 进一步提出 modified Adaptive Symmetry-based PINN (m-ASPINN)，通过引入自适应激活函数（adaptive activation function），在训练过程中动态调整激活函数参数（如引入超参数 α 并优化），以加速收敛并提升精度。
    *   **训练过程:** 使用 Adam 优化器对损失函数进行优化，确保初始条件、边界条件和 PDE 残差的平衡，同时通过对称性约束和自适应技术增强网络对物理规律的捕捉能力。
*   **关键点:** 该方法不改变 PINNs 的基本架构，而是通过损失函数改进和激活函数优化，利用数学对称性提升性能，同时保持计算效率。

## Experiment

*   **有效性:** 实验通过两个 Burgers 方程问题验证方法效果。传统 PINN (Case A) 结果与精确解偏差较大；引入 Lie 对称性后的 m-SPINN (Case B) 显著降低绝对误差；结合自适应激活函数的 m-ASPINN (Case C) 进一步提升精度，误差接近甚至优于部分传统数值方法（如 MCB-DQM）。
*   **全面性:** 实验设置合理，涵盖三种案例对比、多种激活函数（Tanh, GELU, Mish, Swish）、不同时间和空间点的数据，以及与多种数值方法（MCB-DQM, WA-DQM, BDF 等）的比较，数据点数量充足（初始条件 500 个，边界条件 500 个，配点 20000 个）。
*   **计算效率:** m-ASPINN 在处理大规模数据时表现出较高效率，例如 10000 个数据点的计算时间约为 71-124 秒，表明方法在实际应用中具有可行性。
*   **结论:** 实验结果表明，Lie 对称性和自适应技术的结合显著提升了 PINNs 的性能，尤其在非线性 PDEs 求解中效果突出。

## Further Thoughts

Lie 对称群与深度学习的结合为将抽象数学理论融入机器学习提供了新思路，是否可以探索其他数学不变性（如时空对称性）来改进神经网络性能？
自适应激活函数在提升收敛速度和精度方面的潜力，是否可推广至其他领域如图像处理或自然语言处理？
通过嵌入领域知识（如 PDEs 对称性）到损失函数，是否能在生物学或经济学等领域设计类似方法以提升模型对复杂系统的建模能力？