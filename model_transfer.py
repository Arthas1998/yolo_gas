import os
from models.common import *
import torch

weights = f"runs//train//exp13//weights//best.pt"
ckpt = torch.load(weights, map_location='cuda')
print(ckpt)
model = ckpt['model'].float()
count = 0
for m in model.modules():
    count+=1
    if type(m) is QARepVGGBlock:
        m.switch_to_deploy()
    if type(m) is C3_QARepVGGBlock:
        for mm in m.children():
            if isinstance(mm, torch.nn.Sequential):
                for mmm in mm.children():
                    if isinstance(mmm, Bottleneck_QARepVGGBlock):
                        for mmmm in mmm.children():
                            if type(mmmm) is QARepVGGBlock:
                                mmmm.switch_to_deploy()


ckpt = {
                    'epoch': ckpt["epoch"],
                    'best_fitness': ckpt["best_fitness"],
                    'model': ckpt["model"],
                    'ema': model,
                    'updates': ckpt["updates"],
                    'optimizer': ckpt["optimizer"],
                    'opt': ckpt["opt"],
                    'date': ckpt["date"]}




torch.save(ckpt, "Gas_pipdata2k_rep_V66455033_20250331.pt")
print("Entire model saved successfully.")


# import torch
# import yaml
# from models.yolo import Model  # 确保导入你的模型定义
#
# # 定义权重路径
# weights = "runs/train/exp257/weights/best.pt"
#
# # 加载权重文件
# ckpt = torch.load(weights, map_location='cuda')
#
#
# # ckpt = {
# #                     'epoch': epoch,
# #                     'best_fitness': best_fitness,
# #                     'model': deepcopy(de_parallel(model)).half(),
# #                     'ema': deepcopy(ema.ema).half(),
# #                     'updates': ema.updates,
# #                     'optimizer': optimizer.state_dict(),
# #                     'opt': vars(opt),
# #                     'date': datetime.now().isoformat()}
# print(ckpt["ema"])




# # 检查 ckpt 是否是一个字典
# if isinstance(ckpt, dict):
#     # 如果是字典，加载模型结构并加载权重
#     yaml_file = "models/yolov5nqa.yaml"  # YAML 文件路径
#     with open(yaml_file, 'r') as f:
#         cfg = yaml.safe_load(f)
#     model = Model(cfg).to('cuda')  # 根据 YAML 文件初始化模型结构
#
#     # 加载权重
#     if 'ema' in ckpt:
#         model.load_state_dict(ckpt['ema'])
#     else:
#         model.load_state_dict(ckpt['model'])
# else:
#     # 如果不是字典，直接使用模型对象
#     model = ckpt
#
# # 转换为 FP32
# model.float()
#
# # 遍历模型模块并调用 switch_to_deploy
# for m in model.modules():
#     if isinstance(m, QARepVGGBlock):
#         m.switch_to_deploy()
#     elif isinstance(m, C3_QARepVGGBlock):
#         for mm in m.children():
#             if isinstance(mm, torch.nn.Sequential):
#                 for mmm in mm.children():
#                     if isinstance(mmm, Bottleneck_QARepVGGBlock):
#                         for mmmm in mmm.children():
#                             if isinstance(mmmm, QARepVGGBlock):
#                                 mmmm.switch_to_deploy()
#
# # 保存整个模型对象
# torch.save(model, "yolov5n_qa_infer_20250219.pt")
# print("Entire model saved successfully.")

