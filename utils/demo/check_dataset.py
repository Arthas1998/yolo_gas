import os

def extract_annotation_folder_name(video_filename):
    """
    从视频文件名中提取注释文件夹名：
    例：Dec07_20241227_n_C_04___640___560___250.mp4 -> Dec07_20241227_n_C_04
    """
    return video_filename.split("___")[0]

def check_dataset(txt_file, dataset_root, output_dir):
    print(f"正在检查：{txt_file}")

    with open(txt_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    missing_videos = []
    missing_annotations = []
    valid_lines = []

    for rel_path in lines:
        video_path = os.path.normpath(os.path.join(dataset_root, rel_path))
        video_exists = os.path.isfile(video_path)

        video_name = os.path.basename(video_path)
        base_dir = os.path.dirname(os.path.dirname(video_path))  # 如 test_set_xxxxx
        annotation_folder = extract_annotation_folder_name(video_name)
        annotation_dir = os.path.join(base_dir, "Annotations", annotation_folder)
        annotation_exists = os.path.isdir(annotation_dir)

        # 相对路径保存
        if not video_exists:
            missing_videos.append(rel_path)
        if not annotation_exists:
            rel_annotation = os.path.relpath(annotation_dir, dataset_root)
            missing_annotations.append(rel_annotation)

        if video_exists and annotation_exists:
            valid_lines.append(rel_path)

    print(f"总共 {len(lines)} 个条目")
    print(f"缺失视频文件数: {len(missing_videos)}")
    print(f"缺失注释文件夹数: {len(missing_annotations)}")
    print(f"有效条目数: {len(valid_lines)}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(txt_file))[0]

    # 保存缺失视频路径
    with open(os.path.join(output_dir, f"{base_name}_missing_videos.txt"), "w") as f:
        for path in missing_videos:
            f.write(path + "\n")

    # 保存缺失注释文件夹路径
    with open(os.path.join(output_dir, f"{base_name}_missing_annotations.txt"), "w") as f:
        for path in missing_annotations:
            f.write(path + "\n")

    # 保存新的有效条目 txt
    with open(os.path.join(output_dir, f"{base_name}_filtered.txt"), "w") as f:
        for path in valid_lines:
            f.write(path + "\n")

    print(f"缺失信息和新版 txt 已保存至目录：{output_dir}\n")

if __name__ == "__main__":
    # 修改为你的数据集根目录的绝对路径或相对路径
    dataset_root = "D:\data\dataset\DATA_SET_HIT"  # 例如 "/data/project/dataset"
    output_dir = "D:\data\dataset\DATA_SET_HIT\missing_report"

    check_dataset("D:\data\dataset\DATA_SET_HIT\Training_set_HIT.txt", dataset_root, output_dir)
    check_dataset("D:\data\dataset\DATA_SET_HIT\Test_set_HIT.txt", dataset_root, output_dir)

