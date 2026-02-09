"""VLM 引擎：封装 Qwen3-VL-8B-IT 的加载、推理、输出解析。"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from config import Config
from prompts import (
    SYSTEM_PROMPT,
    TRIAGE_USER_TEMPLATE,
    DIRECT_ANSWER_TEMPLATE,
    DEPTH_ASSISTED_ANSWER_TEMPLATE,
)

logger = logging.getLogger(__name__)


@dataclass
class TriageResult:
    confidence: int                # 1-10
    need_depth_map: bool
    reason: str
    raw_output: str


@dataclass
class AnswerResult:
    answer: str
    raw_output: str


class VLMEngine:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.processor = None

    def load(self):
        if self.model is not None:
            return
        logger.info("加载 VLM: %s", self.config.vlm_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.vlm_path,
            torch_dtype=torch.bfloat16,
            device_map=self.config.vlm_device,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.config.vlm_path)
        logger.info(
            "VLM 加载完成 — 显存: %.1f GB",
            torch.cuda.memory_allocated() / 1e9,
        )

    def unload(self):
        if self.model is None:
            return
        logger.info("卸载 VLM")
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _generate(self, messages: list[dict], max_new_tokens: int = 0) -> str:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        images, videos = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or self.config.answer_max_new_tokens,
            temperature=self.config.vlm_temperature,
            top_p=self.config.vlm_top_p,
            top_k=self.config.vlm_top_k,
        )

        output_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]

    @staticmethod
    def _format_options(options: dict) -> str:
        lines = []
        for key in ("A", "B", "C", "D"):
            val = options.get(key)
            if val is not None:
                lines.append(f"{key}. {val}")
        return "\n".join(lines)

    # ── Triage ──────────────────────────────────────────────────────────

    def run_triage(self, image: Image.Image, question: str, options: dict) -> TriageResult:
        options_text = self._format_options(options)
        user_text = TRIAGE_USER_TEMPLATE.format(
            question=question, options_text=options_text,
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        raw = self._generate(messages, max_new_tokens=self.config.triage_max_new_tokens)
        return self._parse_triage(raw)

    # ── Direct answer (no depth map) ────────────────────────────────────

    def run_direct_answer(self, image: Image.Image, question: str, options: dict) -> AnswerResult:
        options_text = self._format_options(options)
        user_text = DIRECT_ANSWER_TEMPLATE.format(
            question=question, options_text=options_text,
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        raw = self._generate(messages, max_new_tokens=self.config.answer_max_new_tokens)
        answer = self._parse_answer(raw)
        return AnswerResult(answer=answer, raw_output=raw)

    # ── Depth-assisted answer ───────────────────────────────────────────

    def run_depth_assisted_answer(
        self,
        original_image: Image.Image,
        depth_map: Image.Image,
        question: str,
        options: dict,
    ) -> AnswerResult:
        options_text = self._format_options(options)
        user_text = DEPTH_ASSISTED_ANSWER_TEMPLATE.format(
            question=question, options_text=options_text,
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": original_image},
                    {"type": "image", "image": depth_map},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        raw = self._generate(messages, max_new_tokens=self.config.answer_max_new_tokens)
        answer = self._parse_answer(raw)
        return AnswerResult(answer=answer, raw_output=raw)

    # ── Output parsing ──────────────────────────────────────────────────

    @staticmethod
    def _parse_answer(text: str) -> str:
        m = re.search(r"答案\s*[:：]\s*([A-Da-d])", text)
        if m:
            return m.group(1).upper()
        m = re.search(r"ANSWER\s*[:：]\s*([A-Da-d])", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        letters = re.findall(r"\b([A-D])\b", text)
        return letters[-1] if letters else "A"

    @staticmethod
    def _parse_triage(raw: str) -> TriageResult:
        """Parse structured triage output. Defaults to conservative (need depth map) on failure."""
        confidence = 1
        need_depth_map = True
        reason = ""

        for line in raw.splitlines():
            line = line.strip()
            if line.upper().startswith("CONFIDENCE:"):
                val = line.split(":", 1)[1].strip()
                m = re.search(r"(\d+)", val)
                if m:
                    confidence = max(1, min(10, int(m.group(1))))
            elif line.upper().startswith("NEED_DEPTH_MAP:"):
                val = line.split(":", 1)[1].strip().upper()
                need_depth_map = val != "NO"
            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        return TriageResult(
            confidence=confidence,
            need_depth_map=need_depth_map,
            reason=reason,
            raw_output=raw,
        )
