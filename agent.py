"""核心 Agent：VLM Triage 分流 + 深度图辅助的视觉推理。

流程：
  Triage: VLM 判断难度
     ↓
  多层安全决策: 是否需要深度图
     ↓
  需要 → Editor 对原图生成深度图 → VLM 结合原图+深度图回答
  不需要 → VLM 直接回答
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from config import Config
from vlm import VLMEngine, TriageResult, AnswerResult
from image_editor import ImageEditorEngine
from prompts import DEPTH_MAP_TEMPLATE

logger = logging.getLogger(__name__)


# ── 决策数据类 ──────────────────────────────────────────────────────────

@dataclass
class TriageDecision:
    """多层安全决策的结果。"""
    use_editor: bool
    trigger_reason: str       # 触发 editor 的原因（或 "skipped"）


@dataclass
class SolveResult:
    """一次 Agent.solve() 调用的完整结果。"""
    final_answer: str
    triage: TriageResult
    triage_decision: TriageDecision
    direct_answer: Optional[AnswerResult] = None
    depth_answer: Optional[AnswerResult] = None
    depth_map: Optional[Image.Image] = None
    used_editor: bool = False
    error: Optional[str] = None
    timing: dict = field(default_factory=dict)


# ── 多层安全决策 ────────────────────────────────────────────────────────

def decide_use_editor(triage: TriageResult, config: Config) -> TriageDecision:
    """三层安全网决定是否使用深度图编辑器。

    Layer 1: Triage 解析失败 → 用 editor（保守兜底）
    Layer 2: VLM confidence < threshold → 用 editor
    Layer 3: VLM 自己说 NEED_DEPTH_MAP=YES → 用 editor
    只有全部通过才跳过 editor。
    """
    # Layer 1: 解析失败兜底
    if triage.confidence == 1 and not triage.reason.strip():
        return TriageDecision(use_editor=True, trigger_reason="parse_failure")

    # Layer 2: 置信度不足
    if triage.confidence < config.confidence_threshold:
        return TriageDecision(use_editor=True, trigger_reason=f"low_confidence({triage.confidence})")

    # Layer 3: VLM 自评需要深度图
    if triage.need_depth_map:
        return TriageDecision(use_editor=True, trigger_reason="vlm_requested")

    # 全部通过 → 跳过 editor
    return TriageDecision(use_editor=False, trigger_reason="skipped")


# ── Agent ────────────────────────────────────────────────────────────────

class OmniAgent:
    def __init__(self, config: Config):
        self.config = config
        self.vlm = VLMEngine(config)
        self.editor = ImageEditorEngine(config)

    def setup(self):
        self.vlm.load()
        if self.config.enable_editor:
            self.editor.load()

    def teardown(self):
        pass

    def solve(self, image: Image.Image, question: str, options: dict) -> SolveResult:
        """运行完整的推理循环：Triage → 决策 → 直接/深度图辅助回答。"""
        timing = {}

        # ── Step 1: Triage — 判断难度 ────────────────────────────────
        t0 = time.time()
        triage = self.vlm.run_triage(image, question, options)
        timing["triage"] = round(time.time() - t0, 2)
        logger.info(
            "Triage → confidence=%d, need_depth=%s",
            triage.confidence, triage.need_depth_map,
        )

        # ── Step 2: 多层安全决策 ─────────────────────────────────────
        if not self.config.enable_editor:
            decision = TriageDecision(use_editor=False, trigger_reason="editor_disabled")
        else:
            decision = decide_use_editor(triage, self.config)
        logger.info("决策: use_editor=%s, reason=%s", decision.use_editor, decision.trigger_reason)

        # ── Path A: 不需要深度图 → 直接回答 ──────────────────────────
        if not decision.use_editor:
            t0 = time.time()
            direct = self.vlm.run_direct_answer(image, question, options)
            timing["direct_answer"] = round(time.time() - t0, 2)
            logger.info("Direct answer → %s", direct.answer)

            return SolveResult(
                final_answer=direct.answer,
                triage=triage,
                triage_decision=decision,
                direct_answer=direct,
                timing=timing,
            )

        # ── Path B: 对原图生成深度图 → 深度图辅助回答 ─────────────────
        logger.info("生成原图深度图: %s", DEPTH_MAP_TEMPLATE)
        try:
            t0 = time.time()
            depth_map = self.editor.edit(image, DEPTH_MAP_TEMPLATE)
            timing["depth"] = round(time.time() - t0, 2)
        except Exception as e:
            logger.error("深度图生成失败: %s — fallback 到直接回答", e)
            t0 = time.time()
            direct = self.vlm.run_direct_answer(image, question, options)
            timing["direct_answer"] = round(time.time() - t0, 2)
            return SolveResult(
                final_answer=direct.answer,
                triage=triage,
                triage_decision=decision,
                direct_answer=direct,
                used_editor=False,
                error=f"深度图生成失败: {e}",
                timing=timing,
            )

        # VLM 结合原图+深度图回答
        try:
            t0 = time.time()
            depth_answer = self.vlm.run_depth_assisted_answer(
                image, depth_map, question, options,
            )
            timing["depth_answer"] = round(time.time() - t0, 2)
            logger.info("Depth-assisted answer → %s", depth_answer.answer)

            return SolveResult(
                final_answer=depth_answer.answer,
                triage=triage,
                triage_decision=decision,
                depth_answer=depth_answer,
                depth_map=depth_map,
                used_editor=True,
                timing=timing,
            )
        except Exception as e:
            logger.error("深度图辅助推理失败: %s — fallback 到直接回答", e)
            t0 = time.time()
            direct = self.vlm.run_direct_answer(image, question, options)
            timing["direct_answer"] = round(time.time() - t0, 2)
            return SolveResult(
                final_answer=direct.answer,
                triage=triage,
                triage_decision=decision,
                direct_answer=direct,
                depth_map=depth_map,
                used_editor=True,
                error=f"深度图辅助推理失败: {e}",
                timing=timing,
            )
