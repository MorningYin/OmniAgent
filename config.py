"""OmniAgent configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # ── Model paths ──────────────────────────────────────────────────────
    vlm_path: str = "/root/autodl-tmp/.cache/huggingface/hub/Qwen3-VL-8B-IT"
    editor_path: str = (
        "/root/autodl-tmp/.cache/huggingface/hub/"
        "models--Qwen--Qwen-Image-Edit-2511/snapshots/main"
    )
    lightning_lora_path: str = (
        "/root/autodl-tmp/.cache/huggingface/hub/"
        "models--lightx2v--Qwen-Image-Edit-2511-Lightning/snapshots/main/"
        "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors"
    )

    # ── Device assignment ───────────────────────────────────────────────
    vlm_device: str = "cuda:0"
    editor_device: str = "cuda:1"

    # ── VLM inference parameters ─────────────────────────────────────────
    vlm_temperature: float = 0.7
    vlm_top_p: float = 0.8
    vlm_top_k: int = 20
    triage_max_new_tokens: int = 512
    answer_max_new_tokens: int = 1024

    # ── Editor inference parameters ──────────────────────────────────────
    editor_num_steps: int = 4          # Lightning LoRA 4-step inference
    editor_true_cfg_scale: float = 1.0  # Lightning LoRA 模式下用 1.0
    editor_guidance_scale: float = 1.0
    editor_negative_prompt: str = " "

    # ── Agent parameters ─────────────────────────────────────────────────
    enable_editor: bool = True         # False = VLM-only baseline
    confidence_threshold: int = 8      # 1-10, >= this to skip depth map

    # ── Paths ────────────────────────────────────────────────────────────
    dataset_path: str = "/root/OmniAgent/datasets.jsonl"
    results_dir: str = "/root/OmniAgent/results"
    image_cache_dir: str = "/root/OmniAgent/image_cache"

    # ── Misc ─────────────────────────────────────────────────────────────
    download_timeout: int = 30
    log_level: str = "INFO"
