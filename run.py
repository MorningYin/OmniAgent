"""评估入口：加载数据集、运行 Agent、统计结果。"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import time

from config import Config
from agent import OmniAgent
from utils import (
    setup_logging,
    download_image,
    load_dataset,
    load_completed_indices,
    save_result,
    compute_summary,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="OmniAgent 评估")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-editor", action="store_true",
                        help="禁用编辑器（纯 VLM 基线）")
    parser.add_argument("--results-dir", default=None)
    return parser.parse_args()


def save_sample_detail(results_dir: str, idx: int, sample: dict,
                       result, image_path: str):
    """保存每个样本的详细输出到 results/samples/NNNNN/ 目录。"""
    sample_dir = os.path.join(results_dir, "samples", f"{idx:05d}")
    os.makedirs(sample_dir, exist_ok=True)

    # ── 复制原图 ─────────────────────────────────────────────────────
    orig_dst = os.path.join(sample_dir, "original.jpg")
    if not os.path.exists(orig_dst):
        shutil.copy2(image_path, orig_dst)

    files_index = {"original": "original.jpg"}

    # ── Triage 输出 ───────────────────────────────────────────────────
    with open(os.path.join(sample_dir, "triage_response.txt"), "w") as f:
        f.write(result.triage.raw_output)
    files_index["triage_response"] = "triage_response.txt"

    # ── 深度图 ────────────────────────────────────────────────────────
    if result.depth_map is not None:
        result.depth_map.save(os.path.join(sample_dir, "depth_map.png"))
        files_index["depth_map"] = "depth_map.png"

    # ── 回答输出 ──────────────────────────────────────────────────────
    answer_result = result.depth_answer or result.direct_answer
    if answer_result is not None:
        with open(os.path.join(sample_dir, "answer_response.txt"), "w") as f:
            f.write(answer_result.raw_output)
        files_index["answer_response"] = "answer_response.txt"

    # ── meta.json ─────────────────────────────────────────────────────
    correct = result.final_answer == sample["answer"]
    meta = {
        "idx": idx,
        "url": sample["url"],
        "question": sample["question"],
        "question_with_options": sample.get("question_with_options", ""),
        "options": sample["options"],
        "gt": sample["answer"],
        # Triage
        "triage_confidence": result.triage.confidence,
        "triage_need_depth": result.triage.need_depth_map,
        "triage_reason": result.triage.reason,
        "trigger_reason": result.triage_decision.trigger_reason,
        "use_editor": result.used_editor,
        # Final
        "final": result.final_answer,
        "correct": correct,
        "error": result.error,
        # Timing
        "time_triage": result.timing.get("triage"),
        "time_direct_answer": result.timing.get("direct_answer"),
        "time_depth": result.timing.get("depth"),
        "time_depth_answer": result.timing.get("depth_answer"),
        # Files
        "files": files_index,
    }
    with open(os.path.join(sample_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


def main():
    args = parse_args()
    config = Config()

    if args.dataset:
        config.dataset_path = args.dataset
    if args.results_dir:
        config.results_dir = args.results_dir
    if args.no_editor:
        config.enable_editor = False

    setup_logging(config)

    tag = "no_editor" if args.no_editor else "full"
    results_path = os.path.join(config.results_dir, f"results_{tag}.jsonl")
    summary_path = os.path.join(config.results_dir, f"summary_{tag}.json")

    dataset = load_dataset(config.dataset_path)
    logger.info("已加载 %d 条样本: %s", len(dataset), config.dataset_path)

    if args.max_samples:
        dataset = dataset[:args.max_samples]
        logger.info("限制为前 %d 条", len(dataset))

    completed = load_completed_indices(results_path)
    if completed:
        logger.info("断点续跑: 已完成 %d 条", len(completed))

    agent = OmniAgent(config)
    agent.setup()

    total_time_start = time.time()
    correct_count = 0
    editor_count = 0
    processed = 0

    for idx, sample in enumerate(dataset):
        if idx in completed:
            continue

        question = sample["question"]
        options = sample["options"]
        ground_truth = sample["answer"]
        url = sample["url"]

        logger.info("━━━ 样本 %d/%d ━━━", idx + 1, len(dataset))

        # 下载图片
        try:
            image = download_image(url, config)
        except Exception as e:
            logger.error("图片下载失败 %s: %s", url, e)
            save_result(results_path, {
                "idx": idx, "question": question, "url": url,
                "gt": ground_truth, "final": None, "correct": False,
                "error": f"下载失败: {e}", "use_editor": False,
            })
            continue

        # 推理
        result = agent.solve(image, question, options)

        correct = result.final_answer == ground_truth
        processed += 1
        if correct:
            correct_count += 1
        if result.used_editor:
            editor_count += 1

        # 保存
        url_hash = hashlib.md5(url.encode()).hexdigest()
        image_path = os.path.join(config.image_cache_dir, f"{url_hash}.jpg")
        meta = save_sample_detail(config.results_dir, idx, sample, result, image_path)
        save_result(results_path, meta)

        # 控制台
        acc = correct_count / processed * 100
        status = "正确" if correct else "错误"
        if result.used_editor:
            detail = (
                f"conf={result.triage.confidence} "
                f"reason={result.triage_decision.trigger_reason} "
                f"→ 深度图 → {result.final_answer}"
            )
        else:
            detail = (
                f"conf={result.triage.confidence} "
                f"reason={result.triage_decision.trigger_reason} "
                f"→ {result.final_answer}"
            )
        logger.info(
            "%s | GT=%s | %s | 累计 %d/%d = %.1f%%",
            detail, ground_truth, status, correct_count, processed, acc,
        )

    total_time = time.time() - total_time_start
    agent.teardown()

    # 汇总
    summary = compute_summary(results_path)
    summary["total_time_seconds"] = round(total_time, 1)
    summary["mode"] = tag

    os.makedirs(config.results_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("结果汇总 (%s)", tag)
    logger.info("=" * 60)
    logger.info("总样本数:        %d", summary["total"])
    logger.info("准确率:          %.2f%%", summary.get("accuracy", 0) * 100)
    logger.info("置信度分布:      %s", summary.get("confidence_distribution", {}))
    logger.info("触发原因分布:    %s", summary.get("trigger_reason_distribution", {}))
    logger.info("编辑器使用:      %d 次 (%.1f%%)",
                summary.get("editor_used_count", 0),
                summary.get("editor_used_rate", 0) * 100)
    logger.info("总耗时:          %.0f 秒 (%.1f 秒/样本)",
                total_time, total_time / max(processed, 1))
    logger.info("样本详情:        %s", os.path.join(config.results_dir, "samples/"))


if __name__ == "__main__":
    main()
