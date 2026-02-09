"""评估入口：加载数据集、运行 Agent、统计结果。"""

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import time

import numpy as np
import torch

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


def _build_conversation_log(result) -> str:
    """构建完整对话记录文本。"""
    lines = []
    sep = "=" * 60

    if result.used_editor and result.direct_answer and result.depth_answer:
        # ── 两轮对话：直接回答 + 深度图修正 ──
        lines.append(sep)
        lines.append("[Round 1] Direct answer")
        lines.append(sep)
        lines.append("")
        lines.append("[User] <image> + answer prompt")
        lines.append("")
        lines.append("[Assistant]")
        lines.append(result.direct_answer.raw_output)
        lines.append("")
        lines.append(sep)
        lines.append("[Round 2] Depth-assisted answer")
        lines.append(sep)
        lines.append("")
        lines.append("[User] <depth_map> + depth prompt")
        lines.append("")
        lines.append("[Assistant]")
        lines.append(result.depth_answer.raw_output)
        lines.append("")
    elif result.direct_answer:
        # ── 单轮：直接回答 ──
        lines.append(sep)
        lines.append("[Round 1] Direct answer")
        lines.append(sep)
        lines.append("")
        lines.append("[User] <image> + answer prompt")
        lines.append("")
        lines.append("[Assistant]")
        lines.append(result.direct_answer.raw_output)
        lines.append("")

    # 最终答案
    lines.append(sep)
    lines.append(f"Final answer: {result.final_answer}")
    if result.direct_answer and result.depth_answer:
        lines.append(f"  Round 1: {result.direct_answer.answer}")
        lines.append(f"  Round 2: {result.depth_answer.answer} (final)")
    lines.append(sep)

    return "\n".join(lines)


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

    # ── 深度图 ────────────────────────────────────────────────────────
    if result.depth_map is not None:
        result.depth_map.save(os.path.join(sample_dir, "depth_map.png"))
        files_index["depth_map"] = "depth_map.png"

    # ── 完整对话记录 ──────────────────────────────────────────────────
    conversation = _build_conversation_log(result)
    with open(os.path.join(sample_dir, "conversation.txt"), "w") as f:
        f.write(conversation)
    files_index["conversation"] = "conversation.txt"

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


def print_summary_table(summary, total_time, processed, tag, config):
    """打印最终统计表格。"""
    total = summary.get("total", 0)
    correct = summary.get("correct", 0)
    accuracy = summary.get("accuracy", 0)
    editor_count = summary.get("editor_used_count", 0)
    editor_rate = summary.get("editor_used_rate", 0)
    editor_correct = summary.get("editor_correct", 0)
    editor_acc = summary.get("editor_accuracy")
    no_editor_correct = summary.get("no_editor_correct", 0)
    no_editor_acc = summary.get("no_editor_accuracy")
    no_editor_count = total - editor_count

    conf_dist = summary.get("confidence_distribution", {})
    reason_dist = summary.get("trigger_reason_distribution", {})

    w = 62
    sep = "─" * w
    print()
    print(f"┌{sep}┐")
    print(f"│{'结果汇总 (' + tag + ')':^{w}}│")
    print(f"├{sep}┤")
    print(f"│ {'指标':<28}{'数值':>{w-31}} │")
    print(f"├{sep}┤")
    print(f"│ {'总样本':<28}{total:>{w-31}} │")
    print(f"│ {'有效样本':<26}{processed:>{w-29}} │")
    print(f"│ {'总准确率':<26}{f'{correct}/{total} = {accuracy*100:.1f}%':>{w-29}} │")
    print(f"├{sep}┤")
    print(f"│ {'使用编辑器':<24}{f'{editor_count} ({editor_rate*100:.1f}%)':>{w-27}} │")
    if editor_acc is not None:
        print(f"│ {'  编辑器准确率':<22}{f'{editor_correct}/{editor_count} = {editor_acc*100:.1f}%':>{w-25}} │")
    print(f"│ {'未使用编辑器':<22}{f'{no_editor_count} ({(1-editor_rate)*100:.1f}%)':>{w-25}} │")
    if no_editor_acc is not None:
        print(f"│ {'  直接回答准确率':<20}{f'{no_editor_correct}/{no_editor_count} = {no_editor_acc*100:.1f}%':>{w-23}} │")
    print(f"├{sep}┤")
    print(f"│ {'置信度分布':<24}{' ':>{w-27}} │")
    for conf in sorted(conf_dist.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
        cnt = conf_dist[conf]
        print(f"│   confidence={str(conf):<5} {cnt:>5} 次{' ':>{w-32}} │")
    print(f"├{sep}┤")
    print(f"│ {'触发原因分布':<22}{' ':>{w-25}} │")
    for reason, cnt in sorted(reason_dist.items(), key=lambda x: -x[1]):
        label = f"  {reason}"
        print(f"│ {label:<30}{f'{cnt} 次':>{w-33}} │")
    print(f"├{sep}┤")
    avg_time = total_time / max(processed, 1)
    print(f"│ {'总耗时':<28}{f'{total_time:.0f}s ({avg_time:.1f}s/样本)':>{w-31}} │")
    print(f"│ {'Seed':<30}{f'{config.seed}':>{w-33}} │")
    print(f"│ {'样本详情':<26}{os.path.join(config.results_dir, 'samples/'):>{w-29}} │")
    print(f"└{sep}┘")
    print()


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

    # 固定全局 seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

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

    # ── 打印统计表格 ────────────────────────────────────────────────
    print_summary_table(summary, total_time, processed, tag, config)


if __name__ == "__main__":
    main()
