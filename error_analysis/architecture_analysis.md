# OmniAgent 架构问题分析与重设计规划

- **日期**：2026-02-09
- **当前版本准确率**：82.89%（562/678）
- **纯 Round 1 准确率（假设不用编辑器）**：83.78%（568/678）
- **编辑器净贡献**：-6（纠正 3 个，改错 9 个）

---

## 一、当前架构

```
原图 + 题目 + 选项
       │
       ▼
 ┌─ VLM Triage（GPU 0）──────────────┐
 │  VLM 自评 confidence (1-10)       │
 │  VLM 自评 NEED_DEPTH_MAP (YES/NO) │
 └──────────┬────────────────────────┘
            │
     confidence >= 8 且
     NEED_DEPTH_MAP=NO ?
            │
     ┌──YES─┴──NO──┐
     ▼              ▼
  Path A         Path B
  直接回答       Round 1: 直接回答
  (单轮)              │
     │           Round 2: 生成深度图(GPU 1)
     │              → VLM 看深度图修正(GPU 0)
     │              │
     ▼              ▼
  最终答案       最终答案
```

### 当前提示词

**SYSTEM_PROMPT**
```
You are a 3D spatial reasoning expert. Answer spatial reasoning multiple-choice questions based on an image.
```

**TRIAGE_USER_TEMPLATE**
```
Carefully observe this image and assess the difficulty of the following spatial reasoning question.

Question: {question}
Options:
{options_text}

Important: Judging 3D depth relationships from a 2D image is error-prone. Be honest about your ability and do not be overconfident.
Only consider depth assistance unnecessary when the depth relationship is very obvious (e.g., one object is clearly in the foreground and the other is clearly in the distant background).

Output strictly in the following format:
CONFIDENCE: <an integer from 1 to 10, where 10 means completely certain and 1 means completely uncertain>
NEED_DEPTH_MAP: <YES or NO>
REASON: <one sentence explanation>
```

**DIRECT_ANSWER_TEMPLATE**
```
Carefully observe this image and answer the following spatial reasoning question.

Question: {question}

Options:
{options_text}

Think step by step in 1-2 sentences, then give your answer.
You must end with exactly this format:
Answer: <a single letter A, B, C, or D>
```

**DEPTH_ASSISTED_ANSWER_TEMPLATE**
```
Here is a depth map of the same scene. Your previous answer may be wrong. Use the depth map to verify or correct it.

Reminder — the question is:
{question}
Options:
{options_text}

How to read the depth map:
- BRIGHTER (whiter) pixels = CLOSER to the camera.
- DARKER (blacker) pixels = FARTHER from the camera.

You MUST follow these steps:
1. Identify the objects mentioned in the question. For EACH object, find its EXACT region in the depth map and describe its brightness (bright/dark/medium). Do NOT describe irrelevant objects.
2. Compare the brightness of the relevant objects to determine their relative depth.
3. Give your final answer based ONLY on this depth comparison. If it contradicts your previous answer, you MUST change it.

Answer: <a single letter A, B, C, or D>
```

**DEPTH_MAP_TEMPLATE**
```
生成这张图的深度图
```

### 当前配置

| 参数 | 值 |
|------|----|
| VLM | Qwen3-VL-8B-IT, GPU 0 |
| Editor | Qwen-Image-Edit-2511 + Lightning LoRA 4-step, GPU 1 |
| confidence_threshold | 8 |
| vlm_temperature | 0.7 |
| seed | 42 |

---

## 二、量化数据

### 2.1 总体表现

| 指标 | 值 |
|------|----|
| 总样本 | 678 |
| 当前准确率 | 82.89%（562/678） |
| 纯 Round 1 准确率 | 83.78%（568/678） |
| 编辑器触发数 | 109（16.1%） |
| 编辑器路径准确率 | 75.2%（82/109） |
| 直接回答路径准确率 | 84.4%（480/569） |

### 2.2 五类结果分布

| 分类 | 数量 | 说明 |
|------|------|------|
| 自信跳过编辑器，答对 | 480 | 正常 |
| 自信跳过编辑器，答错 | 89 | **Triage 漏检** |
| 调用编辑器，两轮都对 | 79 | 编辑器无害 |
| 调用编辑器，本来对→深度图改错 | 9 | **编辑器有害** |
| 调用编辑器，两轮都错 | 18 | 编辑器无效 |
| 调用编辑器，本来错→深度图纠正 | 3 | **编辑器有效** |

### 2.3 Triage 置信度校准

| Confidence | 总数 | 错误数 | 错误率 |
|-----------|------|--------|--------|
| 8 | 232 | 47 | 20.26% |
| 9 | 329 | 42 | 12.77% |
| 10 | 8 | 0 | 0.00% |

### 2.4 编辑器触发质量

| 指标 | 值 |
|------|----|
| 正确识别需要帮助的样本 | 21/109 = 19.27% |
| 过度保守（Round 1 本来就对） | 88/109 = 80.73% |
| 真正纠错成功 | 3/21 = 14.29% |

---

## 三、根本问题分析

### 问题 1：Triage 用 VLM 自评 VLM，逻辑自相矛盾

**现象**：
- 模型给出 confidence=8，但实际错误率 20.26%
- 109 次触发编辑器中，80.73% 是不必要的（Round 1 本来就对）
- 89 个真正需要帮助的样本（自信但答错），Triage 全部没有捕获

**根因**：Triage 本质上是让一个空间推理能力有限的模型，去评估自己在空间推理上的能力。这是一个元认知任务，模型的 confidence 输出与其实际正确率之间**没有可靠的对应关系**。confidence=8 应该代表 ~80% 的把握，但实际错误率已经 20%，校准是正确的——问题是这种校准对路由决策没有帮助，因为我们无法从 confidence 值中区分"这道题我真的会"和"这道题我以为我会"。

**为什么提示词改不了**：这不是提示词表述的问题，而是 8B 模型的元认知能力上界。

### 问题 2：VLM 无法可靠地从深度图图像中提取深度信息

**现象举例**：

**例 1 — 样本 162（本来对 → 深度图改错）**

原图：机场停机坪，乘客排队登机。airport lounge 在远处背景，airplane 在中景。

| | Round 1（看原图） | Round 2（看深度图） |
|---|---|---|
| 输出 | "airport lounge 在背景，比 airplane 更远" → **A（正确）** | "airplane 是暗色轮廓...airport lounge 在深度图中不可辨识...airplane 看起来是最远的" → **B（错误）** |
| 分析 | 模型凭常识正确判断了远近 | 深度图中前景人群是亮白色，airplane 是暗色。模型承认 airport lounge "在深度图中不可辨识"，但仍然强行给出了基于 airplane 亮度的错误结论 |

**失败机制**：airport lounge 在深度图中融入了黑色背景，模型找不到它。模型被迫基于不完整信息回答，放弃了 Round 1 的正确推理。

**例 2 — 样本 154（两轮都错）**

原图：草地上一个消防栓，两侧各一根蓝色柱子（poles）。三者紧挨在一起。GT = A（fire hydrant 更远）。

| | Round 1（看原图） | Round 2（看深度图） |
|---|---|---|
| 输出 | "poles 在 fire hydrant 后面，更远" → **B（错误）** | "fire hydrant 中等亮度，poles 更暗，所以 poles 更远" → **B（错误）** |
| 分析 | 模型凭视觉线索判断错误 | 深度图中消防栓和柱子亮度非常接近，模型声称 poles "更暗"，但实际上柱子（亮白）比消防栓更亮——模型读反了 |

**失败机制**：深度图上消防栓和柱子的亮度差异极小，模型无法可靠区分，且读反了亮暗关系。同时 Round 2 确认了 Round 1 的错误。

**例 3 — 样本 046（两轮都错）**

原图：路口一根杆子上挂着 ONE WAY 标牌（中部）和 street signs（顶部）。GT = B（street signs 更远）。

| | Round 1（看原图） | Round 2（看深度图） |
|---|---|---|
| 输出 | "ONE WAY 更低，slightly behind，更远" → **A（错误）** | "ONE WAY 中等亮度，street signs 亮白色（更近）。所以 ONE WAY 更远" → **A（错误）** |
| 分析 | 模型把"更低"当"更远"——实际上地面视角仰拍，越高越远 | 深度图中整根杆子亮度接近，模型声称 street signs 更亮，但即使这是真的，也说明 street signs 更近——模型用了正确的读法（亮=近），但得出的结论反而支持了错误答案 |

**失败机制**：深度图对同一结构体上不同高度物体的深度差异几乎没有分辨力。模型在噪声级别的亮度差异上强行解读，结果不可靠。

**根因**：VLM 从未被训练过如何解读灰度深度图。"亮=近、暗=远"的文字指令无法赋予模型真正的深度图读取能力。模型在 Round 2 中做的事情本质上是：
1. 把深度图当普通灰度照片看
2. 用一般视觉能力去"猜"哪个区域更亮
3. 在亮度差异微小时，猜测的结果不可靠
4. 当目标物体在深度图中不可辨识时，转而分析无关物体

**为什么提示词改不了**：无论怎么措辞，模型都需要在一张灰度图像上做精确的亮度比较——这是一个视觉感知精度问题，不是指令理解问题。人类看深度图也很难判断两个相邻区域谁更亮 0.5 个灰度值。

### 问题 3：多轮对话引入系统性偏差

**现象**：
- 类别 3（9 个样本）：模型在 Round 2 盲目抛弃 Round 1 的正确答案
- 类别 4（18 个样本）：模型在 Round 2 确认 Round 1 的错误答案（80%）

**根因**：Round 1 的答案在上下文中，模型在 Round 2 时无法客观看待新信息：
- 如果深度图信息与 Round 1 矛盾 → 过度矫枉（例 1）
- 如果深度图信息模糊 → 确认偏误（例 2、3）

两种偏差方向相反，无法通过调整提示词语气（"更强硬"或"更温和"）同时解决。

---

## 四、核心矛盾总结

| 组件 | 设计假设 | 实际情况 | Gap |
|------|----------|----------|-----|
| Triage | VLM 能准确自评 confidence | confidence=8 时错误率 20%，89 个需要帮助的样本全部漏检 | VLM 元认知能力不足 |
| 深度图生成 | Qwen-Image-Edit 能生成高质量深度图 | 同一结构上的物体、细小物体、背景物体的深度分辨率不足 | 图像编辑模型非专业深度估计模型 |
| 深度图读取 | VLM 看深度图能提取深度信息 | VLM 找不到目标物体、读反亮暗、在噪声上强行解读 | VLM 从未被训练读灰度深度图 |
| 多轮修正 | Round 2 能基于新信息纠正 Round 1 | 要么确认偏误，要么过度矫枉 | 上下文中已有答案导致无法客观判断 |

**一句话总结**：当前架构的每一个环节都存在能力上界问题，不是提示词工程能弥补的 Gap。Triage 不可靠、深度图看不懂、多轮引入偏差——三者叠加导致编辑器路径的净贡献为负。

---

## 五、重设计方案

### 5.1 核心思路变更

**从"让 VLM 看深度图"变为"程序化提取深度值，以文本形式告诉 VLM"**

| 当前架构 | 新架构 |
|----------|--------|
| VLM 看深度图图片 → 猜亮暗 | 程序从深度图像素中读取数值 → 文本注入 VLM |
| VLM 自评 confidence 做路由 | 每题都提取深度信息，无需路由 |
| 多轮对话（Round 1 → Round 2） | 单轮回答（原图 + 深度文本一次给齐） |

### 5.2 新架构流程

```
原图 + 题目 + 选项
       │
       ├──────────────────────────────────┐
       ▼                                  ▼
   VLM 物体定位 (GPU 0)            深度估计模型 (GPU 1)
   "图中 {object} 的位置在哪"       生成深度图
   → bounding box / 区域描述        (Depth Anything V2
       │                            或当前 Qwen-Image-Edit)
       │                                  │
       └──────────────┬───────────────────┘
                      ▼
              程序化深度提取 (CPU)
         对每个物体的 bbox 区域
         计算平均深度值 (0-255)
                      │
                      ▼
              VLM 最终回答 (GPU 0)
         原图 + 题目 + 选项
         + "物体A的深度值=187（较近），
            物体B的深度值=94（较远）"
                      │
                      ▼
                 最终答案
```

### 5.3 各环节设计

#### 5.3.1 去掉 Triage，每题都提取深度信息

- **理由**：Triage 的 80% 触发是浪费，89 个漏检是损失。与其路由，不如每题都做。
- **代价**：每题多一次深度估计（~3.7s），但省掉了 Triage 推理（~1.2s），净增 ~2.5s/题。
- **收益**：消除路由错误，89 个漏检样本也能获得深度信息辅助。

#### 5.3.2 物体定位 (Grounding)

- 用 VLM 自身的能力输出题目中每个物体的 bounding box
- 或者使用 Qwen3-VL 的 grounding 能力（如果支持）
- 或者退而求其次：让 VLM 描述物体在图中的位置（"左上角"、"中下方"），程序将其映射到图像坐标

#### 5.3.3 深度图生成

- **方案 A（推荐）**：替换为专用深度估计模型（Depth Anything V2），质量更高、推理更快、GPU 显存占用更小
- **方案 B（兼容当前）**：继续用 Qwen-Image-Edit，保持现有部署不变

#### 5.3.4 程序化深度提取

```python
def extract_depth(depth_map: np.ndarray, bbox: tuple) -> float:
    """从深度图中提取某区域的平均深度值"""
    x1, y1, x2, y2 = bbox
    region = depth_map[y1:y2, x1:x2]
    return float(region.mean())
```

- 输出是精确数字（如 187.3 vs 94.1），不是"bright/dark"的模糊描述
- VLM 只需要比较两个数字的大小，这是语言模型最擅长的事

#### 5.3.5 单轮最终回答

- 不再有 Round 1 → Round 2 的多轮结构
- VLM 一次性看到：原图 + 题目 + 选项 + 深度信息文本
- 消除多轮对话的确认偏误 / 过度矫枉问题

### 5.4 预期收益

| 问题 | 当前架构 | 新架构 | 解决方式 |
|------|----------|--------|----------|
| Triage 漏检 89 个 | 这些样本没有深度信息 | 每题都有深度信息 | 去掉 Triage |
| VLM 看不懂深度图 | 67% 找错物体，24% 读反 | VLM 只需比较数字大小 | 程序化提取 + 文本注入 |
| 多轮确认偏误 | 80% 确认 Round 1 错误 | 无多轮 | 单轮回答 |
| 多轮过度矫枉 | 9 个正确答案被推翻 | 无多轮 | 单轮回答 |
| 深度图分辨率不足 | 同一结构物体无法区分 | 若数值差异 < 阈值可标记为"深度接近" | 数值化后可设置阈值 |
