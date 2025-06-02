import torch
from models.yolo import DetectionModel
from utils.general import check_yaml
from pathlib import Path

def export_param_shapes(cfg_path, output_txt):
    # 检查并读取模型配置
    cfg_path = check_yaml(cfg_path)
    print(f"Loading model config: {cfg_path}")

    # 初始化模型（使用占位通道数=3，类别数=80）
    model = DetectionModel(cfg=cfg_path, ch=3, nc=1)
    model.eval()
    print("Model initialized.")

    model_weight = {name: param.shape for name, param in model.named_parameters()}

    # 输出参数名称和尺寸
    output_txt = Path(output_txt)
    with output_txt.open('w') as f:
        for name, shape in model_weight.items():
            f.write(f"{name}: {shape}\n")

    print(f"Parameter shapes written to: {output_txt}")

if __name__ == "__main__":
    # 示例用法（直接写入参数）
    cfg = r"D:\data\PythonProject\HITProject\Code_HIT\models\yolov5s.yaml"
    output = r"D:\data\PythonProject\HITProject\Code_HIT\yolov5s_weight.txt"
    export_param_shapes(cfg, output)

