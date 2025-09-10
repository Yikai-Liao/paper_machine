---
title: "Proof2Silicon: Prompt Repair for Verified Code and Hardware Generation via Reinforcement Learning"
pubDatetime: 2025-09-07T23:04:15+00:00
slug: "2025-09-prompt-repair-hardware"
type: "arxiv"
id: "2509.06239"
score: 0.5131447364707137
author: "grok-3-latest"
authors: ["Deming Chen", "Manvi Jha", "Jiaxin Wan"]
tags: ["LLM", "Reinforcement Learning", "Formal Verification", "Hardware Synthesis", "Prompt Optimization"]
institution: ["University of Illinois Urbana-Champaign"]
description: "Proof2Silicon提出了一种端到端框架，通过强化学习优化提示生成形式验证通过的Dafny代码，并自动化转换为RTL硬件设计，显著提升验证成功率和硬件合成率。"
---

> **Summary:** Proof2Silicon提出了一种端到端框架，通过强化学习优化提示生成形式验证通过的Dafny代码，并自动化转换为RTL硬件设计，显著提升验证成功率和硬件合成率。 

> **Keywords:** LLM, Reinforcement Learning, Formal Verification, Hardware Synthesis, Prompt Optimization

**Authors:** Deming Chen, Manvi Jha, Jiaxin Wan

**Institution(s):** University of Illinois Urbana-Champaign


## Problem Background

大型语言模型（LLMs）在自动化代码生成方面表现出色，但生成的代码往往无法通过形式验证，这在硬件设计和安全关键领域（如航空航天、汽车安全系统）是不可接受的，因为错误可能导致灾难性后果。
传统解决方案如模型微调成本高昂且可能破坏通用能力，因此需要一种无需微调、通过优化提示生成可验证代码的方法，并将其进一步转化为可合成的硬件设计，填补自然语言描述到硬件实现之间的空白。

## Method

*   **核心框架：Proof2Silicon** - 这是一个端到端的合成框架，将自然语言描述转化为形式验证通过的代码，并最终生成硬件设计（RTL）。其核心步骤包括：
    *   **PREFACE框架（提示优化）**：基于强化学习（RL）的提示修复机制，利用一个小型语言模型（SLM）作为RL代理，迭代优化输入到冻结LLM的提示。SLM根据Dafny验证器的反馈（错误信息作为奖励信号）调整提示，指导LLM生成可通过形式验证的Dafny代码。RL问题被建模为马尔可夫决策过程（MDP），使用Proximal Policy Optimization（PPO）算法训练SLM。
    *   **代码转换流程**：验证通过的Dafny代码首先通过Dafny编译器转换为Python代码，随后通过自动化脚本去除Dafny特有依赖（如内部库），并利用NumPy替换数学操作以确保语义清晰。
    *   **硬件合成优化**：Python代码通过PyLog工具进一步转换为高层次合成（HLS）C代码，PyLog使用装饰器（decorators）进行硬件加速优化（如循环流水线、数组分区），确保代码适合FPGA硬件执行。
    *   **RTL生成**：最终使用Xilinx Vivado HLS工具将HLS C代码合成为RTL，验证其在FPGA上的结构兼容性和性能指标（如时序、资源利用）。
*   **关键创新**：整个流程无需微调LLM，依赖于模型无关的提示优化，且通过自动化转换链条实现从软件验证到硬件合成的无缝衔接。

## Experiment

*   **验证效果**：在100个DafnyBench任务的基准测试中，PREFACE的RL引导提示优化显著提升了Dafny代码的验证成功率，最高增幅达21%（如Gemini-2-Flash从20%提升至55%），且对多个LLM（如ChatGPT-4o、Qwen2.5-Coder-14B）均有效，体现了方法普适性。
*   **硬件合成效果**：Proof2Silicon的端到端硬件合成成功率最高达72.4%（Gemini-2-Flash无反馈模式），成功合成的设计在Xilinx Zynq-7000 SoC上表现出一致的资源利用和性能，验证了从验证代码到RTL的可行性。
*   **实验设置合理性**：实验覆盖了多个主流LLM和不同反馈模式（无反馈 vs. 验证器反馈），任务类型包括排序、算术运算等，设置较为全面；硬件合成测试基于标准FPGA平台，提供了详细的时序和资源报告。
*   **局限性**：部分验证通过的代码因Dafny特有结构（如递归、动态循环）无法通过HLS合成，导致约30%的代码在最终阶段被排除，表明流程在兼容性上仍有改进空间。

## Further Thoughts

论文中通过强化学习优化提示以提升代码验证成功率的思路非常具有启发性，这种方法不仅限于代码生成，或许可以推广到其他需要精确输出的任务（如数学推理或逻辑证明）。此外，将形式验证与硬件合成结合的框架让我思考：是否可以将硬件性能指标（如延迟、资源利用）直接纳入RL奖励函数，生成既正确又高效的硬件设计？这种‘硬件感知’的提示优化可能成为未来研究的一个重要方向。