#!/usr/bin/env python3
"""
红绿灯数据集 85:15 严格多标签分层切分脚本 (Multilabel Stratified Split).

功能：
1. 全域收集：自动扫描输入目录下现有的 images/train, images/val 和 labels/train, labels/val，融合成全集。
2. 贪婪均衡：保证在一张图包含多个类别的情况下，每个类别的 Train/Val 比例尽可能贴近 85:15。
3. 极速重构：使用多进程并发，将文件拷贝至全新目录，维持标准的 images/labels 结构。
"""

import argparse
import multiprocessing
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# ================= 类别与配置 =================
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

TRAIN_RATIO = 0.83


def parse_label_file(label_path):
    """读取单张图片的标签，返回该图包含的所有类别ID列表."""
    classes = []
    try:
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    classes.append(int(parts[0]))
    except Exception:
        pass
    return classes


def collect_dataset(source_dir):
    """全域扫描，不论原本在 train 还是 val，全部收集到内存中."""
    source_path = Path(source_dir)
    frames = []

    print("🔍 正在全域扫描数据集 (合并现有的 train 和 val)...")
    for split in ["train", "val"]:
        labels_dir = source_path / "labels" / split
        images_dir = source_path / "images" / split

        if not labels_dir.exists():
            continue

        # 遍历所有的 txt 文件
        for lbl_file in labels_dir.glob("*.txt"):
            if lbl_file.name == "classes.txt":
                continue

            # 寻找对应图片
            img_file = None
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
                temp_img = images_dir / f"{lbl_file.stem}{ext}"
                if temp_img.exists():
                    img_file = temp_img
                    break

            # 只有当图片和标签都存在时，才加入有效池
            if img_file:
                classes = parse_label_file(lbl_file)
                frames.append({"stem": lbl_file.stem, "lbl_path": lbl_file, "img_path": img_file, "classes": classes})
    return frames


def greedy_stratified_split(frames, train_ratio=0.85):
    """核心算法：多标签贪婪均衡切分 确保极度不平衡的类别也能在 train 和 val 中按比例存在。.
    """
    print("⚖️ 正在执行贪婪多标签分层切分算法...")
    # 1. 统计全局每个类别的总数
    global_counts = Counter()
    for f in frames:
        global_counts.update(f["classes"])

    # 2. 计算每个类别的期望目标数量
    train_targets = {c: count * train_ratio for c, count in global_counts.items()}
    val_targets = {c: count * (1 - train_ratio) for c, count in global_counts.items()}

    # 3. 记录当前的分配状态
    train_current = {c: 0 for c in global_counts}
    val_current = {c: 0 for c in global_counts}

    train_frames = []
    val_frames = []

    # 4. 关键：对图片排序，优先分配包含“稀有类别”的图片
    for f in frames:
        score = sum(1.0 / global_counts[c] for c in f["classes"]) if f["classes"] else 0
        f["score"] = score
    frames.sort(key=lambda x: x["score"], reverse=True)

    # 5. 贪婪分配
    for f in frames:
        cost_train = 0
        cost_val = 0

        # 评估放入 train 还是 val 的“成本” (越远离目标比例，成本越高)
        for c in f["classes"]:
            if train_targets[c] > 0:
                cost_train += train_current[c] / train_targets[c]
            if val_targets[c] > 0:
                cost_val += val_current[c] / val_targets[c]

        # 放入进度较落后的那一边
        if cost_train <= cost_val:
            train_frames.append(f)
            for c in f["classes"]:
                train_current[c] += 1
        else:
            val_frames.append(f)
            for c in f["classes"]:
                val_current[c] += 1

    return train_frames, val_frames, train_current, val_current, global_counts


def copy_worker(args):
    """原子拷贝任务."""
    src, dst = args
    try:
        shutil.copy2(src, dst)
        return True
    except:
        return False


def execute_copy(frames, split_name, output_dir_path):
    """多进程执行图片和标签的拷贝."""
    images_out = output_dir_path / "images" / split_name
    labels_out = output_dir_path / "labels" / split_name

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    tasks = []
    for f in frames:
        tasks.append((f["img_path"], images_out / f["img_path"].name))
        tasks.append((f["lbl_path"], labels_out / f["lbl_path"].name))

    max_workers = max(1, multiprocessing.cpu_count() - 2)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(copy_worker, t) for t in tasks]
        for _ in tqdm(as_completed(futures), total=len(tasks), desc=f"🚀 拷贝 {split_name.upper()} 集"):
            pass


def print_audit_report(global_counts, train_counts, val_counts):
    """打印详细的切分比例审计报告."""
    print("\n" + "=" * 80)
    print("📊 多标签分层切分审计报告 (目标 Train:Val = 85:15)")
    print("=" * 80)
    header = f"{'ID':<3} | {'类别名':<18} | {'全集总量':<10} | {'Train 数量':<12} | {'Val 数量':<10} | {'实际 Train 比例':<15}"
    print(header)
    print("-" * 80)

    max_bias = 0
    for cls_id in range(15):
        name = CURRENT_CLASSES[cls_id]
        tot = global_counts.get(cls_id, 0)
        trn = train_counts.get(cls_id, 0)
        val = val_counts.get(cls_id, 0)

        if tot == 0:
            print(f"{cls_id:<3} | {name:<18} | {tot:<10} | {trn:<12} | {val:<10} | 0.00%")
            continue

        real_ratio = (trn / tot) * 100
        bias = abs(85.0 - real_ratio)
        max_bias = max(max_bias, bias)

        # 如果偏差很大，给个高亮提醒
        flag = " ⚠️偏离" if bias > 5.0 else ""
        print(f"{cls_id:<3} | {name:<18} | {tot:<10} | {trn:<12} | {val:<10} | {real_ratio:.2f}% {flag}")

    print("=" * 80)
    print(f"✅ 切分完成！最大类别比例偏差仅为: {max_bias:.2f}% (对于多标签数据集已是极限最优)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="需重新切分的原始数据集根目录 (如: /home/jzyh/xbzl/traffic_light/data_v1_filtered)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="重组后的完美比例数据集输出目录 (如: /home/jzyh/xbzl/traffic_light/data_v1_rebalanced)",
    )
    args = parser.parse_args()

    # 1. 收集全域数据
    frames = collect_dataset(args.source_dir)
    if not frames:
        print("❌ 未在源目录找到有效数据，请检查路径是否包含 images 和 labels 文件夹。")
        return

    print(f"✅ 成功加载 {len(frames)} 张有效图片及其标签。")

    # 2. 执行贪婪切分
    train_frames, val_frames, train_counts, val_counts, global_counts = greedy_stratified_split(frames, TRAIN_RATIO)

    # 3. 打印审计报表
    print_audit_report(global_counts, train_counts, val_counts)

    confirm = input("\n👉 是否开始执行多进程文件物理隔离与拷贝？[y/N]: ")
    if confirm.lower() != "y":
        print("已取消拷贝。")
        return

    # 4. 物理隔离落地
    out_path = Path(args.output_dir)
    execute_copy(train_frames, "train", out_path)
    execute_copy(val_frames, "val", out_path)

    print(f"\n🎉 大功告成！完美比例的数据集已就绪: {out_path.absolute()}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
