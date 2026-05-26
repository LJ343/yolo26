import os
import glob
from collections import defaultdict

def clean_validation_data():
    # 定义目录路径
    img_dir = "/home/jzyh/xbzl/traffic_light/data_v1_filtered/images/val"
    lbl_dir = "/home/jzyh/xbzl/traffic_light/data_v1_filtered/labels/val"
    
    # 检查目录是否存在
    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        print("错误：找不到指定的目录，请检查路径。")
        return

    # 定义按类型匹配的模式
    # patterns = {
    #     '.jpg': os.path.join(img_dir, "*aug*.jpg"),
    #     '.npy': os.path.join(img_dir, "*aug*.npy"),
    #     '.txt': os.path.join(lbl_dir, "*aug*.txt")
    # }
    patterns = {

        '.npy': os.path.join(img_dir, "*.npy"),

    }
    
    # 收集要删除的文件列表，按类型存储
    files_to_delete = []   # 存储 (文件路径, 类型)
    type_counts = defaultdict(int)
    
    for ext, pattern in patterns.items():
        matched = glob.glob(pattern)
        for f in matched:
            files_to_delete.append((f, ext))
            type_counts[ext] += 1
    
    if not files_to_delete:
        print("没有找到需要删除的文件，目录很干净！")
        return
    
    # 打印按类型统计的数量（控制台输出）
    print("找到以下待删除文件：")
    for ext, count in type_counts.items():
        print(f"  {ext} 文件: {count} 个")
    print(f"总计: {len(files_to_delete)} 个文件")
    
    # 将详细文件列表保存到 txt
    list_file = "to_delete_list.txt"
    with open(list_file, 'w', encoding='utf-8') as f:
        f.write("待删除文件详细列表\n")
        f.write("=" * 60 + "\n")
        # 按类型分组写入，便于查看
        for ext in ['.jpg', '.npy', '.txt']:
            if type_counts[ext] == 0:
                continue
            f.write(f"\n【{ext} 文件】共 {type_counts[ext]} 个\n")
            f.write("-" * 40 + "\n")
            for file_path, file_ext in files_to_delete:
                if file_ext == ext:
                    f.write(file_path + "\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"总计: {len(files_to_delete)} 个文件\n")
    
    print(f"详细文件列表已保存到: {list_file}")
    
    # 确认是否删除
    confirm = input("请检查上述文件列表，是否确认删除？(y/n): ")
    if confirm.lower() != 'y':
        print("操作已取消。")
        # 可选择是否保留列表文件，这里提示一下
        print(f"列表文件 {list_file} 已保留，如需删除请手动清理。")
        return
    
    # 执行删除操作，并统计实际成功删除的数量（按类型）
    deleted_counts = defaultdict(int)
    deleted_total = 0
    for file_path, ext in files_to_delete:
        try:
            os.remove(file_path)
            deleted_counts[ext] += 1
            deleted_total += 1
        except Exception as e:
            print(f"删除 {file_path} 时出错: {e}")
    
    print("\n清理完成！")
    for ext, count in deleted_counts.items():
        print(f"成功删除 {ext} 文件: {count} 个")
    print(f"总计成功删除: {deleted_total} 个文件")
    
    # 可选：删除完成后自动删除列表文件
    if deleted_total > 0:
        try:
            os.remove(list_file)
            print(f"已删除列表文件 {list_file}")
        except:
            pass

if __name__ == "__main__":
    clean_validation_data()