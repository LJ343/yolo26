#!/usr/bin/env python3
"""
已清洗数据集统计脚本 (多进程极速版)
直接扫描 data_v1_filtered/labels 下的 train 和 val 文件夹
采用文件分块与 ProcessPoolExecutor，榨干多核 CPU 性能.
"""

import multiprocessing
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

# ================= 配置区域 =================
# DATASET_DIR = "/home/jzyh/xbzl/traffic_light/data_v1_filtered_augment_smartcrop"
DATASET_DIR = "/home/jzyh/xbzl/traffic_light/data_v1_rebalanced"
# 当前实际保留的类别
CURRENT_CLASSES = [
    "round_red",
    "round_yellow",
    "round_green",
    "up_red",
    "up_yellow",
    "up_green",
    "left_red",
    "left_yellow",
    "left_green",
    "right_red",
    "right_yellow",
    "right_green",
    "turn_around_red",
    "turn_around_yellow",
    "turn_around_green",
]


def process_file_chunk(file_paths):
    """独立进程的 Worker 函数：一次性处理一大批文件，极大降低通信开销 返回: (该批次的分类计数统计, 处理的文件数量).
    """
    local_counts = defaultdict(int)
    for path in file_paths:
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:  # 标准 YOLO 格式至少 5 列 (id x y w h)
                    try:
                        local_counts[int(parts[0])] += 1
                    except ValueError:
                        continue
    return local_counts, len(file_paths)


def scan_and_count():
    # counts 结构: { phase: { class_id: count } }
    counts = {"train": defaultdict(int), "val": defaultdict(int), "total": defaultdict(int)}

    total_labels_processed = 0
    phases = ["train", "val"]

    # 获取合适的进程数（预留2个核心给系统）
    max_workers = max(1, multiprocessing.cpu_count() - 2)

    for phase in phases:
        labels_dir = os.path.join(DATASET_DIR, "labels", phase)
        if not os.path.exists(labels_dir):
            print(f"⚠️ 警告: 未找到 {phase} 标签目录 {labels_dir}")
            continue

        # 收集所有文件的完整路径
        label_files = [
            os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith(".txt") and f != "classes.txt"
        ]

        if not label_files:
            continue

        # 【核心加速】将十几万个文件切分为多个 Chunk，每个 Chunk 包含 1000 个文件
        chunk_size = 1000
        chunks = [label_files[i : i + chunk_size] for i in range(0, len(label_files), chunk_size)]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有 Chunk 任务
            futures = [executor.submit(process_file_chunk, chunk) for chunk in chunks]

            # 使用 tqdm 更新总文件数的进度
            with tqdm(total=len(label_files), desc=f"⚡ 极速扫描 {phase} 集", unit="文件") as pbar:
                for future in as_completed(futures):
                    chunk_counts, processed_num = future.result()

                    # 将这个 Chunk 的结果汇入总账
                    for cls_id, cnt in chunk_counts.items():
                        counts[phase][cls_id] += cnt
                        counts["total"][cls_id] += cnt

                    total_labels_processed += processed_num
                    pbar.update(processed_num)

    return counts, total_labels_processed


def main():
    print(f"🚀 开始多进程扫描数据集: {DATASET_DIR}\n")
    counts, total_files = scan_and_count()

    # 动态获取类别数量
    num_classes = len(CURRENT_CLASSES)

    # 检查是否有越界 ID（用于防错）
    out_of_bounds = {k: v for k, v in counts["total"].items() if k >= num_classes}
    if out_of_bounds:
        print(f"\n[⚠️ 严重警告] 发现超出 0-{num_classes - 1} 范围的异常类别 ID: {out_of_bounds}")
        print("请检查清洗或增强脚本是否正确执行！\n")

    # 输出统计结果到 txt 文件
    output_file = os.path.join(os.path.dirname(__file__), "current_dataset_stats.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Traffic Light Dataset Statistics (Filtered {num_classes} Classes)\n")
        f.write("=" * 65 + "\n")
        f.write(f"{'ID':<4} | {'Class Name':<20} | {'Train':<10} | {'Val':<10} | {'Total':<10}\n")
        f.write("-" * 65 + "\n")

        for cls_id, class_name in enumerate(CURRENT_CLASSES):
            train_cnt = counts["train"].get(cls_id, 0)
            val_cnt = counts["val"].get(cls_id, 0)
            total_cnt = counts["total"].get(cls_id, 0)
            f.write(f"{cls_id:<4} | {class_name:<20} | {train_cnt:<10} | {val_cnt:<10} | {total_cnt:<10}\n")

        f.write("-" * 65 + "\n")
        f.write(
            f"Total Annotations: Train={sum(counts['train'].values())}, Val={sum(counts['val'].values())}, All={sum(counts['total'].values())}\n"
        )
        f.write(f"Total Label Files Parsed: {total_files}\n")

    # 控制台打印预览
    print("\n" + "=" * 65)
    print(f"{'ID':<4} | {'Class Name':<20} | {'Train':<10} | {'Val':<10} | {'Total':<10}")
    print("-" * 65)
    for cls_id, class_name in enumerate(CURRENT_CLASSES):
        train_cnt = counts["train"].get(cls_id, 0)
        val_cnt = counts["val"].get(cls_id, 0)
        total_cnt = counts["total"].get(cls_id, 0)
        print(f"{cls_id:<4} | {class_name:<20} | {train_cnt:<10} | {val_cnt:<10} | {total_cnt:<10}")
    print("=" * 65)

    print(f"\n✅ 扫描完成！共极速解析 {total_files} 个文件。")
    print(f"总计保留有效红绿灯框: {sum(counts['total'].values())} 个")
    print(f"统计报告已保存至: {output_file}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
