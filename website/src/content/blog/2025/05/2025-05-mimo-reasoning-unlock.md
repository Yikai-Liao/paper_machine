---
title: "MiMo: Unlocking the Reasoning Potential of Language Model -- From Pretraining to Posttraining"
pubDatetime: 2025-05-12T14:30:11+00:00
slug: "2025-05-mimo-reasoning-unlock"
type: "arxiv"
id: "2505.07608"
score: 0.8272327589535349
author: "grok-3-latest"
authors: ["Xiaomi LLM-Core Team"]
tags: ["LLM", "Reasoning", "Pre-Training", "Post-Training", "RLHF"]
institution: ["Xiaomi"]
description: "本文通过优化预训练和后训练流程，成功挖掘小型语言模型MiMo-7B的推理潜力，使其在数学和代码推理任务上超越更大规模模型如OpenAI o1-mini。"
---

> **Summary:** 本文通过优化预训练和后训练流程，成功挖掘小型语言模型MiMo-7B的推理潜力，使其在数学和代码推理任务上超越更大规模模型如OpenAI o1-mini。 

> **Keywords:** LLM, Reasoning, Pre-Training, Post-Training, RLHF

**Authors:** Xiaomi LLM-Core Team

**Institution(s):** Xiaomi


## Problem Background

大型语言模型（LLM）在复杂推理任务（如数学和代码生成）中表现出强大潜力，但当前主流强化学习（RL）方法多依赖于较大规模模型（如32B参数），而小型模型（如7B参数）在同时提升数学和代码推理能力方面面临挑战。
作者认为，模型推理潜力的发挥不仅依赖于后训练（Post-Training）优化，更需要在预训练（Pre-Training）阶段就针对推理任务进行设计，因此本文致力于通过优化预训练和后训练流程，充分挖掘小型模型的推理能力，使其性能媲美甚至超越更大规模模型。

## Method

*   **预训练阶段：构建推理潜力基础**
    *   **数据优化与推理模式密度提升**：改进数据预处理流程，开发专门的HTML和PDF提取工具以保留数学公式和代码片段；采用快速全局去重（URL和MinHash去重）和多维度数据过滤（使用小型LLM作为质量评估器）；通过高级推理模型生成大量多样化的合成推理数据，增强数据中的推理模式密度。
    *   **三阶段数据混合策略**：第一阶段平衡各类数据，第二阶段将数学和代码数据比例提升至约70%以强化专业技能，第三阶段加入约10%合成推理数据并将上下文长度从8,192扩展至32,768 token，总计预训练数据量达25万亿token。
    *   **多Token预测（MTP）目标**：引入多Token预测作为辅助训练目标以提升性能并加速推理速度，预训练时使用单MTP层，推理时通过复制和微调扩展为多层以支持推测解码（Speculative Decoding），显著降低生成延迟。
*   **后训练阶段：强化推理能力**
    *   **监督微调（SFT）**：基于500K高质量样本进行初步优化，通过三阶段预处理避免数据泄露并控制样本多样性，主要用于格式对齐（如数学答案格式），为后续RL奠定基础。
    *   **强化学习（RL）数据构建**：精心构建包含100K数学问题和30K代码问题的训练数据集，通过模型难度评估过滤简单问题（去除90%通过率以上的问题），采用规则验证器（如Math-Verify和测试用例）评估正确性，避免奖励欺骗（Reward Hacking）。
    *   **RL训练策略优化**：基于改进的Group Relative Policy Optimization (GRPO)算法，移除KL损失以释放模型潜力，采用动态采样（过滤通过率为1或0的样本）和Clip-Higher策略（提高上剪切界限）优化探索能力；提出‘测试难度驱动的奖励机制’（Test Difficulty Driven Reward），通过对代码问题测试用例按难度分级并分配细粒度奖励（灵感来源于国际信息学奥林匹克IOI评分规则），解决稀疏奖励问题；引入‘简单数据重采样’（Easy Data Re-Sampling）策略，将通过率高的简单问题存入池中并以10%概率采样，提升后期训练的采样效率和稳定性。
    *   **RL基础设施优化**：开发Seamless Rollout Engine，通过连续滚动、异步奖励计算和提前终止机制减少GPU空闲时间，提升训练效率2.29倍，验证效率1.96倍；增强vLLM推理引擎，支持MTP模块并优化鲁棒性（如前缀缓存一致性和调度步数调整）。

## Experiment

*   **预训练模型效果（MiMo-7B-Base）**：在多个基准测试（如BBH得分75.2、LiveCodeBench v5得分32.9、AIME 2024得分32.9）上显著优于同规模模型（如Llama-3.1-8B、Qwen2.5-7B），甚至在部分任务上接近或超越32B模型；Pass@k指标显示其推理潜力随采样次数增加而持续扩大，表明基础模型已具备强大推理能力。
*   **后训练模型效果（MiMo-7B-RL）**：经过RL优化后，MiMo-7B-RL在数学和代码任务上表现卓越，在AIME 2025上得分55.4%，超越OpenAI o1-mini（50.7%）；在LiveCodeBench v6上得分49.3%，大幅领先QwQ-32B-Preview（39.1%）；与从基础模型直接RL训练的MiMo-7B-RL-Zero相比，从SFT模型开始RL训练的MiMo-7B-RL达到更高性能天花板。
*   **实验设置与合理性**：实验覆盖语言理解、数学推理、代码生成及长上下文任务，基准测试多样且全面（如MMLU-Pro、AIME、LiveCodeBench）；对比模型包括同规模及更大规模的开源和闭源模型（如GPT-4o、Claude-3.5-Sonnet），评估设置一致（如采样温度0.6、top-p 0.95）；数据清洗和去重措施有效避免数据泄露，增强结果可信度。
*   **性能提升显著性**：从MiMo-7B-Base到MiMo-7B-RL，性能提升显著，尤其在数学和代码任务上（如AIME 2024从32.9%提升至68.2%）；RL基础设施优化带来的效率提升（2.29倍训练加速）也为大规模实验提供了支持。

## Further Thoughts

论文强调预训练阶段通过数据优化和三阶段混合策略提升推理模式密度，启发我们可以在预训练中针对特定任务设计更精细的数据分布策略，甚至动态调整数据比例以适应模型学习阶段；测试难度驱动奖励机制不仅适用于代码任务，也可能推广到其他稀疏奖励场景，通过分层奖励引导模型逐步攻克复杂任务；MiMo-7B证明小型模型通过精心设计流程可媲美大模型，启发我们重新审视模型规模与性能的关系，或许可以通过高效训练策略减少对大规模模型的依赖；Seamless Rollout Engine的成功表明硬件和系统优化在RL训练中至关重要，未来可探索更多软硬件协同优化的方式进一步提升效率。