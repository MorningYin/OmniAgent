"""Utility functions: image download/cache, result saving, logging."""

import hashlib
import json
import logging
import os
from collections import Counter
from pathlib import Path

import requests
from PIL import Image

from config import Config

logger = logging.getLogger(__name__)


def setup_logging(config: Config):
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def download_image(url: str, config: Config) -> Image.Image:
    """Download an image from URL with disk caching.  Returns a PIL Image."""
    cache_dir = Path(config.image_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_path = cache_dir / f"{url_hash}.jpg"

    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")

    logger.info("Downloading %s", url)
    resp = requests.get(url, timeout=config.download_timeout)
    resp.raise_for_status()

    cache_path.write_bytes(resp.content)
    return Image.open(cache_path).convert("RGB")


def get_image_cache_path(url: str, config: Config) -> str:
    """Return the local cache path for a URL (whether or not it exists yet)."""
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return os.path.join(config.image_cache_dir, f"{url_hash}.jpg")


def load_dataset(path: str) -> list[dict]:
    """Load the JSONL dataset."""
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_completed_indices(results_path: str) -> set[int]:
    """Load already-completed sample indices for resume support."""
    completed = set()
    if not os.path.exists(results_path):
        return completed
    with open(results_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                completed.add(record["idx"])
    return completed


def save_result(results_path: str, record: dict):
    """Append one result record to the JSONL results file."""
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def compute_summary(results_path: str) -> dict:
    """Compute accuracy, confidence distribution, and editor usage statistics."""
    records = []
    with open(results_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    if total == 0:
        return {"total": 0}

    correct = sum(1 for r in records if r.get("correct"))

    # 置信度分布 (1-10)
    confidences = [r.get("triage_confidence", 0) for r in records]
    confidence_dist = dict(Counter(confidences).most_common())

    # 触发原因分布
    reasons = [r.get("trigger_reason", "unknown") for r in records]
    reason_dist = dict(Counter(reasons).most_common())

    # 编辑器使用统计
    used_editor = [r for r in records if r.get("use_editor")]
    editor_count = len(used_editor)
    editor_correct = sum(1 for r in used_editor if r.get("correct"))

    no_editor = [r for r in records if not r.get("use_editor")]
    no_editor_correct = sum(1 for r in no_editor if r.get("correct"))

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "confidence_distribution": confidence_dist,
        "trigger_reason_distribution": reason_dist,
        "editor_used_count": editor_count,
        "editor_used_rate": editor_count / total,
        "editor_correct": editor_correct,
        "editor_accuracy": editor_correct / editor_count if editor_count else None,
        "no_editor_correct": no_editor_correct,
        "no_editor_accuracy": (
            no_editor_correct / len(no_editor) if no_editor else None
        ),
    }
