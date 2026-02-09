# OmniAgent: 自反思视觉推理系统

## 1. 问题定义

给定空间推理数据集 $S = \{(I_i, Q_i, O_i, A_i)\}_{i=1}^{N}$，其中：
- $I_i$：自然场景图像
- $Q_i$：关于 3D 空间关系的问题（如"哪个物体离相机更近？"）
- $O_i = \{o_A, o_B, o_C, o_D\}$：候选选项
- $A_i \in \{A, B, C, D\}$：正确答案

**目标**：最大化 $\text{Acc} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\hat{A}_i = A_i]$。

## 2. 核心理念：VLM 即技能路由器

传统方法预定义若干辅助工具（深度估计、分割等），然后设计规则决定何时调用。本系统的核心理念截然不同：

> **VLM 的思维过程本身就是技能路由器。**

VLM 在推理过程中自然产生"不确定性信号"（置信度），同时能用自然语言描述它需要什么样的视觉辅助——例如"在两个物体之间画一条连接地面的透视线以显示深度差异"。图像编辑模型直接执行该描述，无需任何中间翻译或预定义技能。

## 3. 自反思推理循环

```
Phase 1: 初始推理
  VLM(I, Q, O) → {thinking, answer, confidence, edit_instruction}

Phase 2: 增强推理（仅在 confidence < θ 时触发）
  Editor(I, edit_instruction) → I'
  VLM(I, I', Q, O) → {thinking, final_answer}
```

形式化：

$$\hat{A} = \begin{cases}
a_1 & \text{if } c_1 \geq \theta \\
\text{VLM}(I, I', Q, O) & \text{if } c_1 < \theta, \text{ where } I' = \text{Editor}(I, e_1)
\end{cases}$$

其中 $(a_1, c_1, e_1) = \text{Parse}(\text{VLM}(I, Q, O))$，$\theta = 0.8$ 为置信度阈值。

## 4. 模块说明

| 模块 | 职责 |
|------|------|
| `config.py` | 模型路径、推理参数、显存策略 |
| `prompts.py` | Phase1/Phase2 提示词模板 |
| `vlm.py` | Qwen3-VL-8B-Thinking 封装 |
| `image_editor.py` | Qwen-Image-Edit-2511 + Lightning LoRA 封装 |
| `agent.py` | 自反思推理循环 |
| `utils.py` | 图片缓存、结果保存、日志 |
| `run.py` | 评估入口 |

## 5. 提示词设计

### Phase 1 提示词策略
- 系统提示定义"空间推理专家"角色
- 要求 VLM 在 `<think>...</think>` 中进行链式推理
- 输出结构化字段：ANSWER、CONFIDENCE、EDIT_INSTRUCTION
- EDIT_INSTRUCTION 仅在 VLM 认为需要视觉辅助时生成

### Phase 2 提示词策略
- 提供原图 + 辅助图两张图像
- 告知 VLM 辅助图是根据其请求生成的
- 要求结合两张图像做最终判断

## 6. 显存管理策略

支持两种模式：

### Coexist（共存模式）
- VLM 和 Editor 同时驻留显存
- 适用于显存充足的环境（≥48GB）
- 优势：无需加载/卸载，推理速度快

### Sequential（顺序模式）
- 任一时刻仅一个模型在显存中
- Phase1 完成后卸载 VLM → 加载 Editor → 编辑 → 卸载 Editor → 加载 VLM → Phase2
- 适用于显存受限环境（24-40GB）
- 代价：模型切换开销

## 7. 容错设计

- 图像编辑失败时，直接使用 Phase1 答案作为最终答案
- 图片下载失败时，跳过该样本并记录
- 结果逐条保存（JSONL），支持断点续跑
- VLM 输出解析失败时，尝试 fallback 正则提取
