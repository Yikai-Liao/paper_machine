---
title: "Quantitative Evaluation of KIRETT Wearable Demonstrator for Rescue Operations"
pubDatetime: 2025-09-30T08:21:09+00:00
slug: "2025-09-kirett-wearable-rescue"
type: "arxiv"
id: "2509.25928"
score: 0.6993538230650251
author: "grok-3-latest"
authors: ["Mubaris Nadeem", "Johannes Zenkert", "Lisa Bender", "Christian Weber", "Madjid Fathi"]
tags: ["Wearable Technology", "Artificial Intelligence", "Internet of Things", "Knowledge Graph", "Rescue Operations"]
institution: ["University of Siegen"]
description: "本文通过对 KIRETT 可穿戴设备的用户研究，量化了救援人员对数字化工具的需求，揭示了设备在紧急救援中的潜力和改进方向，为医疗技术设计提供了重要参考。"
---

> **Summary:** 本文通过对 KIRETT 可穿戴设备的用户研究，量化了救援人员对数字化工具的需求，揭示了设备在紧急救援中的潜力和改进方向，为医疗技术设计提供了重要参考。 

> **Keywords:** Wearable Technology, Artificial Intelligence, Internet of Things, Knowledge Graph, Rescue Operations

**Authors:** Mubaris Nadeem, Johannes Zenkert, Lisa Bender, Christian Weber, Madjid Fathi

**Institution(s):** University of Siegen


## Problem Background

紧急救援服务在时间紧迫和信息不足的情况下，难以快速做出准确的医疗诊断和治疗决策，可能导致严重后果。
现代技术（如人工智能、物联网和可穿戴设备）被认为是解决这一问题的关键，KIRETT 项目旨在通过开发一款可穿戴设备，为救援人员提供实时生命体征监测、治疗建议和情境识别支持。

## Method

*   **研究设计**：在德国锡根的两家救援站（消防站和德国红十字会）开展了为期两天的用户研究，共有14名救援人员参与，测试 KIRETT 可穿戴设备在模拟救援场景中的表现。
*   **设备技术**：KIRETT 设备集成了现场可编程门阵列（FPGA）硬件，用于加速实时情境感知算法，通过蓝牙连接医疗设备以传输生命体征数据，并基于知识图谱生成治疗建议。
*   **评估方式**：采用定量与定性结合的方法，通过问卷调查收集参与者对设备硬件设计、功能需求和数字化需求的反馈，并通过个人访谈了解他们的使用体验和改进建议。
*   **数据分析**：使用在线调查工具（如 LimeSurvey）和分析软件（如 MAXQDA 和 Excel）对问卷结果进行统计分析，评估设备在实际救援场景中的适用性和用户接受度。
*   **核心目标**：以用户为中心，关注设备在紧急情况下的实用性、易用性和舒适性，通过直接反馈优化设计。

## Experiment

*   **数字化需求**：86%的参与者认为救援服务的数字化非常重要，100%认为人工智能和机器学习在救援中的应用重要或非常重要，显示出对现代技术的强烈需求。
*   **硬件反馈**：71%参与者希望设备更轻便紧凑，100%强调设备需易于清洁消毒，79%认为支持戴手套操作非常重要，当前 3D 打印外壳的重量和舒适性评价较低（FPGA 外壳50%认为非常不舒服）。
*   **功能需求**：实时生命体征监测（64%认为非常重要）和结构化报告功能（79%认为非常重要）被认为是关键特性，参与者还希望设备支持与电子健康记录（EHR）等系统的扩展通信。
*   **实验设置评价**：实验涵盖了硬件、软件和用户体验多个维度，设置较为全面，但样本量较小（仅14人）且局限于锡根地区，可能影响结果的普适性和代表性。

## Further Thoughts

KIRETT 项目展示了物联网、人工智能和知识图谱技术在紧急医疗场景中的潜力，启发我们思考如何通过跨学科协作进一步提升设备性能，例如结合5G网络提升数据传输速度；
同时，用户对卫生标准和易用性的高要求提示我们，未来设计需更广泛地考虑不同文化和环境下的用户需求差异；
此外，FPGA 加速实时算法的思路值得借鉴，但设备体积问题表明可以探索更小型化的硬件解决方案，如基于 ASIC 的定制芯片。