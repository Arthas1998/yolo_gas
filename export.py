#!/usr/bin/env python
# coding: utf-8

# In[1]:


# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Export a YOLOv5 PyTorch model to TorchScript, ONNX, CoreML, TensorFlow (saved_model, pb, TFLite, TF.js,) formats
TensorFlow exports authored by https://github.com/zldrobit

Usage:
    $ python path/to/export.py --weights yolov5s.pt --include torchscript onnx coreml saved_model pb tflite tfjs

Inference:
    $ python path/to/detect.py --weights yolov5s.pt
                                         yolov5s.onnx  (must export with --dynamic)
                                         yolov5s_saved_model
                                         yolov5s.pb
                                         yolov5s.tflite

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolov5/yolov5s_web_model public/yolov5s_web_model
    $ npm start
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

from models.common import Conv
from models.experimental import attempt_load
from models.yolo import Detect, Decoupled_Detect
from utils.activations import SiLU
from utils.dataloaders import LoadImages
from utils.general import colorstr, check_dataset, check_img_size, check_requirements, file_size, print_args, \
    set_logging, url2file
from utils.torch_utils import select_device

def export_formats():
    # YOLOv5 export formats
    x = [
        ['PyTorch', '-', '.pt', True, True],
        ['TorchScript', 'torchscript', '.torchscript', True, True],
        ['ONNX', 'onnx', '.onnx', True, True],
        ['OpenVINO', 'openvino', '_openvino_model', True, False],
        ['TensorRT', 'engine', '.engine', False, True],
        ['CoreML', 'coreml', '.mlmodel', True, False],
        ['TensorFlow SavedModel', 'saved_model', '_saved_model', True, True],
        ['TensorFlow GraphDef', 'pb', '.pb', True, True],
        ['TensorFlow Lite', 'tflite', '.tflite', True, False],
        ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', False, False],
        ['TensorFlow.js', 'tfjs', '_web_model', False, False],
        ['PaddlePaddle', 'paddle', '_paddle_model', True, True],]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])


def try_export(inner_func):
    # YOLOv5 export decorator, i..e @try_export
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        prefix = inner_args['prefix']
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(f'{prefix} export success ✅ {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)')
            return f, model
        except Exception as e:
            LOGGER.info(f'{prefix} export failure ❌ {dt.t:.1f}s: {e}')
            return None, None

    return outer_func

def export_onnx(model, im, file, opset, train, dynamic, simplify, prefix=colorstr('ONNX:')):
    # YOLOv5 ONNX export
    try:
        # 检查是否安装了所需的包 'onnx'
        check_requirements(('onnx',))
        import onnx  # 导入 ONNX 库

        print(f'\n{prefix} starting export with onnx {onnx.__version__}...')  # 打印 ONNX 版本信息
        f = file.with_suffix('.onnx')  # 将文件名修改为 .onnx 后缀
        # # 确保模型处于部署模式（如果模型包含 QARepVGGBlock）
        # model.switch_to_deploy()
        # if hasattr(model, 'switch_to_deploy'):
        #     model.switch_to_deploy()
        #     print(f'{prefix} model switched to deploy mode.')

        # 设置模型为评估模式
        model.eval()

        # 使用 torch.onnx.export 导出模型为 ONNX 格式
        torch.onnx.export(
            model,  # 要导出的模型
            im,  # 输入示例图像
            f,  # 输出的文件路径
            verbose=False,  # 是否打印详细的导出过程信息
            opset_version=opset,  # ONNX 操作集版本
            training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,  # 设置训练模式还是评估模式
            do_constant_folding=not train,  # 是否进行常量折叠优化
            input_names=['input1'],  # 输入节点名称
            output_names=['box_8', 'obj_8', 'box_16', 'obj_16', 'box_32', 'obj_32', "heatmap"],  # 输出节点名称
            dynamic_axes={  # 如果使用动态轴，设置动态变化的维度
                'images': {0: 'batch', 2: 'height', 3: 'width'},  # 输入图像的动态维度
                'output': {0: 'batch', 1: 'anchors'}  # 输出的动态维度
            } if dynamic else None  # 如果不需要动态轴，则为 None
        )
        print("onnx?")
        # 加载并检查导出的 ONNX 模型
        model_onnx = onnx.load(f)  # 加载导出的 ONNX 模型
        onnx.checker.check_model(model_onnx)  # 检查模型结构和一致性

        for node in model_onnx.graph.node:
            # 遍历节点的属性
            for attr in node.attribute:
                # 检查属性名称是否为 'coordinate_transformation_mode'
                if attr.name == 'coordinate_transformation_mode':
                    # 检查属性值是否为 'pytorch_half_pixel'
                    if attr.s == b'pytorch_half_pixel':
                        # 将属性值修改为 'half_pixel'
                        attr.s = b'half_pixel'




        # print(onnx.helper.printable_graph(model_onnx.graph))  # 打印模型图的可读字符串表示（调试用）

        # 如果需要简化模型
        if simplify:
            try:
                # 检查是否安装了所需的包 'onnx-simplifier'
                check_requirements(('onnx-simplifier',))
                import onnxsim  # 导入 onnx-simplifier 库

                print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')  # 打印 onnx-simplifier 版本信息
                # 使用 onnx-simplifier 简化 ONNX 模型
                model_onnx, check = onnxsim.simplify(
                    model_onnx,  # 要简化的模型
                    dynamic_input_shape=dynamic,  # 是否使用动态输入形状
                    input_shapes={'images': list(im.shape)} if dynamic else None  # 动态输入形状的字典，如果需要
                )
                assert check, 'assert check failed'  # 确保简化过程成功
                onnx.save(model_onnx, f)  # 保存简化后的模型
            except Exception as e:
                print(f'{prefix} simplifier failure: {e}')  # 如果简化失败，打印错误信息

        # 打印导出成功的信息
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')  # 打印导出成功的信息和文件大小
        print(
            f"{prefix} run --dynamic ONNX model inference with: 'python detect.py --weights {f}'")  # 提示如何使用动态 ONNX 模型进行推理
    except Exception as e:
        print(f'{prefix} export failure: {e}')  # 如果导出失败，打印错误信息


@torch.no_grad()  # 禁用梯度计算，适用于推理阶段，节省内存和计算资源
def run(data='data/coco128.yaml',  # 数据集的 YAML 配置文件路径
        weights='yolov5s.pt',  # 模型权重文件路径
        imgsz=(640, 640),  # 输入图像的大小（高度，宽度）
        batch_size=1,  # 批处理大小
        device='cpu',  # 使用的设备，例如 'cpu' 或 'cuda:0'
        include=('torchscript', 'onnx', 'coreml'),  # 包含的导出格式
        half=False,  # 使用半精度浮点数（FP16）进行导出
        inplace=False,  # 设置 YOLOv5 的检测层 inplace=True
        train=False,  # 模型训练模式
        optimize=False,  # TorchScript: 优化移动设备
        int8=False,  # CoreML/TF INT8 量化
        dynamic=False,  # ONNX/TF: 动态轴
        simplify=False,  # ONNX: 简化模型
        opset=12,  # ONNX: opset 版本
        topk_per_class=100,  # TF.js NMS: 每类保留的 topk
        topk_all=100,  # TF.js NMS: 所有类别保留的 topk
        iou_thres=0.45,  # TF.js NMS: IoU 阈值
        conf_thres=0.25  # TF.js NMS: 置信度阈值
        ):
    t = time.time()  # 记录当前时间，作为计时起点

    # 将 include 元素转换为小写
    include = [x.lower() for x in include]

    # 检查 include 中是否包含 TensorFlow 导出格式
    tf_exports = list(x in include for x in ('saved_model', 'pb', 'tflite', 'tfjs'))

    # 如果 img_size 只有一个值，则将其乘以 2 以扩展为 (height, width)
    imgsz *= 2 if len(imgsz) == 1 else 1

    # 处理 weights 路径，如果是 URL 则下载
    file = Path(url2file(weights) if str(weights).startswith(('http:/', 'https:/')) else weights)

    # 加载 PyTorch 模型
    device = select_device(device)  # 选择设备
    assert not (device.type == 'cpu' and half), '--half only compatible with GPU export, i.e. use --device 0'
    model = attempt_load(weights, device=device, inplace=True, fuse=True)  # 加载 FP32 模型
    # model = torch.load(weights, map_location=device)
    nc, names = model.nc, model.names  # 获取类别数量和类别名称

    # #####REPVGG########
    # model = torch.load(weights)
    # state_dict = model.state_dict()
    # save_path = weight_path.replace("save", "convert")
    # # 这里保存的是model 经过convert后的权重
    # repvgg_model_convert(model, save_path=save_path)





    # 输入
    gs = int(max(model.stride))  # 获取最大步长（grid size）
    imgsz = [check_img_size(x, gs) for x in imgsz]  # 确认 img_size 是 gs 的倍数
    im = torch.zeros(batch_size, 8, *imgsz).to(device)  # 创建一个全零的输入张量，用于模型输入

    # 更新模型
    if half:
        im, model = im.half(), model.half()  # 转换为半精度 FP16
    model.train() if train else model.eval()  # 设置模型为训练或评估模式
    count = 0
    for k, m in model.named_modules():
        count += 1
        if isinstance(m, Conv):  # 如果是卷积层
            if isinstance(m.act, nn.SiLU):  # 如果激活函数是 SiLU
                m.act = SiLU()  # 将其替换为导出友好的 SiLU 版本
        elif isinstance(m, Decoupled_Detect):  # 如果是检测层
            m.inplace = inplace  # 设置 inplace 属性
            m.onnx_dynamic = dynamic  # 设置动态 ONNX 属性
            m.heat_map = True  # 启用热力图
            m.export = True  # 启用热力图

    # 定义一个新的 nn.Module 类，用于处理 8 张图像的输入
    class input8img(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x: list):
            new_x = torch.cat(x, dim=2)  # 将输入列表拼接在一起
            new_x = new_x.unsqueeze(0)  # 添加一个新的维度
            return self.model(new_x)  # 将处理后的输入传递给模型

    # 进行两次干运行，以消除初始延迟
    print("check1")
    for _ in range(2):
        y = model(im)  # 干运行模型，不计算梯度
    print("check2")
    print(f"\n{colorstr('PyTorch:')} starting from {file} ({file_size(file):.1f} MB)")

    # 导出模型
    if 'onnx' in include:
        export_onnx(model, im, file, opset, train, dynamic, simplify)  # 导出为 ONNX 格式

    # 结束
    print(f'\nExport complete ({time.time() - t:.2f}s)'  # 打印导出完成信息和时间
          f"\nResults saved to {colorstr('bold', file.parent.resolve())}"  # 结果保存路径
          f'\nVisualize with https://netron.app')  # 提示使用 Netron 可视化模型


# def repvgg_model_convert(model: torch.nn.Module, save_path=None, do_copy=True):
#     if do_copy:
#         # 默认采用深拷贝
#         model = copy.deepcopy(model)
#     for module in model.modules():
#         if hasattr(module, 'switch_to_deploy'):
#             module.switch_to_deploy()
#     # 如果有保存路径，则直接保存，权重
#     if save_path is not None:
#         torch.save(model.state_dict(), save_path)
#     return model


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default='Gas_pipdata2k_rep_V66455033_20250331.pt', help='weights path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[512, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--train', action='store_true', help='model.train() mode')
    parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
    parser.add_argument('--int8', action='store_true', help='CoreML/TF INT8 quantization')
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TF: dynamic axes')
    parser.add_argument('--simplify', action='store_false', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=11, help='ONNX: opset version')
    parser.add_argument('--topk-per-class', type=int, default=100, help='TF.js NMS: topk per class to keep')
    parser.add_argument('--topk-all', type=int, default=100, help='TF.js NMS: topk for all classes to keep')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='TF.js NMS: IoU threshold')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='TF.js NMS: confidence threshold')
    parser.add_argument('--include', nargs='+',
                        default=['torchscript', 'onnx'],
                        help='available formats are (torchscript, onnx, coreml, saved_model, pb, tflite, tfjs)')
    opt = parser.parse_args([])
    return opt


def main(opt):
    set_logging()
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

# In[ ]:




