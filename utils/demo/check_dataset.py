import os


def check_dataset(txt_file, dataset_root, output_dir):
    print(f"正在检查：{txt_file}")

    with open(txt_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    missing_videos = []
    missing_annotations = []

    for rel_path in lines:
        # 构造视频的完整路径（基于 dataset_root）
        video_path = os.path.normpath(os.path.join(dataset_root, rel_path))
        if not os.path.isfile(video_path):
            missing_videos.append(rel_path)

        # 构造注释路径
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        base_dir = os.path.dirname(os.path.dirname(video_path))  # 到达 test_set_xxx
        annotation_dir = os.path.join(base_dir, "Annotations", video_name)

        if not os.path.isdir(annotation_dir):
            # 为了与原始 txt 文件风格一致，输出相对路径
            rel_annotation_dir = os.path.relpath(annotation_dir, dataset_root)
            missing_annotations.append(rel_annotation_dir)

    # 输出基本信息
    print(f"总共 {len(lines)} 个条目")
    print(f"缺失视频文件数: {len(missing_videos)}")
    print(f"缺失注释文件夹数: {len(missing_annotations)}")
    print("=" * 60)

    # 保存到 output_dir
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(txt_file))[0]
    video_out = os.path.join(output_dir, f"{base_name}_missing_videos.txt")
    anno_out = os.path.join(output_dir, f"{base_name}_missing_annotations.txt")

    with open(video_out, "w") as f:
        for path in missing_videos:
            f.write(path + "\n")

    with open(anno_out, "w") as f:
        for path in missing_annotations:
            f.write(path + "\n")

    print(f"结果已保存至: {video_out} 和 {anno_out}\n")


if __name__ == "__main__":
    # 修改为你的数据集根目录的绝对路径或相对路径
    dataset_root = "D:\data\dataset\DATA_SET_HIT"  # 例如 "/data/project/dataset"
    output_dir = "D:\data\dataset\DATA_SET_HIT\missing_report"

    check_dataset("D:\data\dataset\DATA_SET_HIT\Training_set_HIT.txt", dataset_root, output_dir)
    check_dataset("D:\data\dataset\DATA_SET_HIT\Test_set_HIT.txt", dataset_root, output_dir)

