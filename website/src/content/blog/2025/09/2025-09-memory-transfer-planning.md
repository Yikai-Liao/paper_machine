---
title: "Memory Transfer Planning: LLM-driven Context-Aware Code Adaptation for Robot Manipulation"
pubDatetime: 2025-09-29T01:18:59+00:00
slug: "2025-09-memory-transfer-planning"
type: "arxiv"
id: "2509.24160"
score: 0.5557383056175371
author: "grok-3-latest"
authors: ["Tomoyuki Kagaya", "Subramanian Lakshmi", "Yuxuan Lou", "Thong Jing Yuan", "Jayashree Karlekar", "Sugiri Pranata", "Natsuki Murakami", "Akira Kinose", "Yang You"]
tags: ["LLM", "Memory Retrieval", "Context Adaptation", "Robot Manipulation", "Planning"]
institution: ["Panasonic Connect Co., Ltd., Japan", "Panasonic R&D Center, Singapore", "National University of Singapore, Singapore"]
description: "本文提出Memory Transfer Planning (MTP)框架，通过检索和上下文适配以往成功控制代码，提升大型语言模型在机器人操作中的跨环境适应性，显著提高任务成功率。"
---

> **Summary:** 本文提出Memory Transfer Planning (MTP)框架，通过检索和上下文适配以往成功控制代码，提升大型语言模型在机器人操作中的跨环境适应性，显著提高任务成功率。 

> **Keywords:** LLM, Memory Retrieval, Context Adaptation, Robot Manipulation, Planning

**Authors:** Tomoyuki Kagaya, Subramanian Lakshmi, Yuxuan Lou, Thong Jing Yuan, Jayashree Karlekar, Sugiri Pranata, Natsuki Murakami, Akira Kinose, Yang You

**Institution(s):** Panasonic Connect Co., Ltd., Japan, Panasonic R&D Center, Singapore, National University of Singapore, Singapore


## Problem Background

大型语言模型（LLMs）在机器人操作中的应用日益增多，但现有方法在适应新环境时面临挑战：许多系统依赖特定环境的策略训练或固定提示的单次代码生成，导致跨环境的可移植性差，需手动调整提示或重新训练。
论文的出发点是利用以往成功控制代码作为程序性知识（Procedural Knowledge），通过检索和适配这些知识，提升LLM在不同环境下的规划能力，解决跨环境适应性不足的关键问题。

## Method

*   **核心思想:** 提出Memory Transfer Planning (MTP)框架，通过存储和复用以往成功控制代码作为程序性知识，结合上下文感知适配，指导LLM在无需更新模型参数的情况下进行跨环境重新规划。
*   **具体步骤:**
    *   **代码生成（Code Generation）:** 利用LLM根据任务指令生成初始计划和可执行代码，将任务分解为子任务，调用低级程序（Low-level Model Programs, LMPs）生成机器人轨迹。
    *   **记忆检索（Memory Retrieval）:** 构建一个存储成功案例的记忆库，根据任务指令的文本嵌入余弦相似度，检索与当前任务最相关的成功代码。
    *   **记忆适配与重新规划（Memory Adaptation and Re-planning）:** 将检索到的代码通过上下文提示适配到目标环境（例如调整环境特定参数、坐标或初始化步骤），并将适配后的代码作为上下文输入到LLM中，生成新的规划代码。若首次重新规划失败，则迭代使用次相关代码进行尝试，最多支持多次试验。
*   **技术细节:** 记忆库仅存储规划层级（Planner-level）的代码，而非低级程序，以聚焦于任务相关的多样性输出；适配过程利用目标环境的示例提示，确保代码风格和参数与目标环境一致；整个过程不涉及模型微调，仅依赖LLM的上下文学习能力。
*   **优势:** 该方法避免了昂贵的模型训练成本，通过动态记忆检索和适配实现跨环境适应，同时保持代码生成质量。

## Experiment

*   **有效性:** MTP在多个环境中表现出显著提升：在RLBench上，成功率从基线VoxPoser的39.3%提升至64.4%；在CALVIN上，从52.0%提升至67.3%（单一指令）和59.3%（改写指令）；在真实机器人UR5上，总体成功率从30%提升至75%。
*   **适应性与鲁棒性:** MTP在不同任务和指令变化下均表现出更强的泛化能力，尤其在真实机器人任务中，利用模拟环境构建的记忆库成功迁移到现实场景，验证了跨域适应的可行性。
*   **消融分析:** 去掉记忆适配后，成功率显著下降（RLBench从64.4%降至49.3%，CALVIN从67.3%降至60.0%），证明适配步骤对泛化至关重要；不同记忆来源的对比显示，记忆库的质量和多样性（如RLBench相较CALVIN）对效果有较大影响。
*   **实验设置合理性:** 实验覆盖模拟（RLBench、CALVIN）和现实（UR5机器人）场景，任务类型多样（如抓取、推动、旋转等），评估指标（成功率）直接反映方法效果；对比方法包括基线代码生成（VoxPoser）、简单重试（Retry）和自反思（Self-reflection），设置全面且具有代表性。
*   **开销与局限:** 方法主要增加记忆检索和适配的计算成本，但未涉及模型训练，整体开销可控；论文未详细讨论记忆库规模对性能的影响及动态管理策略。

## Further Thoughts

MTP框架中‘程序性知识’的复用概念非常具有启发性，不仅限于机器人操作领域，也可应用于其他需要跨场景适应的任务，如自然语言处理中的任务迁移或软件开发中的代码复用；此外，记忆适配的过程让我思考是否可以通过引入多模态数据（如视觉输入或点云）进一步提升适配精度，或者设计动态记忆管理机制（如自动更新或剪枝记忆库）以应对长期部署中的扩展性挑战；另一个方向是探索记忆库中知识的多样性与质量对迁移效果的深层影响，或许可以通过主动学习策略优化记忆构建。