import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def delete_file(file_path):
    """原子操作：删除单个文件."""
    try:
        file_path.unlink()
        return True
    except Exception:
        return False


def copy_file(args):
    """原子操作：复制单个文件 (src, dst)."""
    src, dst = args
    try:
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False


def run_fast_fix():
    # ================= 配置区域 =================
    src_base = Path("/home/jzyh/xbzl/traffic_light/data_v1_rebalanced_augment")
    dst_base = Path("/home/jzyh/xbzl/traffic_light/data_v1_rebalanced")

    splits = ["train"]
    # 自动获取 CPU 核心数，留 2 个核心给系统
    max_workers = max(1, os.cpu_count() - 2)

    # ================= 1. 多进程清理阶段 =================
    print(f"🧹 正在并行清理错误图片 (使用 {max_workers} 核心)...")
    files_to_delete = []
    for split in splits:
        labels_dir = dst_base / "labels" / split
        if labels_dir.exists():
            for ext in ["*.jpg", "*.JPG", "*.png", "*.PNG"]:
                files_to_delete.extend(list(labels_dir.glob(ext)))

    if files_to_delete:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(delete_file, files_to_delete), total=len(files_to_delete), desc="清理进度"))
    else:
        print("✅ labels 目录下很干净，无需清理。")

    print("\n" + "=" * 60 + "\n")

    # ================= 2. 多进程复制阶段 =================
    print(f"🚀 正在并行复制数据 (使用 {max_workers} 核心)...")

    copy_tasks = []
    for split in splits:
        # 定义路径映射：源目录 train/pic->images/train, train/yolo->labels/train
        mapping = [
            (src_base / split / "pic", dst_base / "images" / split, "*.*"),  # 图像到图像
            (src_base / split / "yolo", dst_base / "labels" / split, "*.txt"),  # 标签到标签
        ]

        for s_dir, d_dir, pattern in mapping:
            if not s_dir.exists():
                print(f"⚠️ 跳过: {s_dir} 不存在")
                continue

            d_dir.mkdir(parents=True, exist_ok=True)
            # 排除可能的 classes.txt 文件
            for f in s_dir.glob(pattern):
                if f.name == "classes.txt":
                    continue
                copy_tasks.append((f, d_dir / f.name))

    if copy_tasks:
        # 使用 as_completed 来驱动进度条，实时感更强
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(copy_file, task) for task in copy_tasks]
            for _ in tqdm(as_completed(futures), total=len(copy_tasks), desc="复制总进度"):
                pass
    else:
        print("⚠️ 未发现可复制的文件，请检查路径。")

    print("\n✨ 处理完成！数据已各就各位。")


if __name__ == "__main__":
    run_fast_fix()
