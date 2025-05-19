import math
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Optional, List
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from torch import Tensor
from matplotlib import cm
from torchvision.transforms.functional import to_pil_image
import cv2
import numpy as np

# 定义一个函数，用于读取视频文件中的每一帧
def Video_read(path, gray=None):
    # 获取视频文件的路径
    videoname = path

    # 使用 OpenCV 打开视频文件
    capture = cv2.VideoCapture(videoname)

    # 创建一个列表，用于存储视频的每一帧图像
    frame_list = []

    # 检查视频是否成功打开
    if capture.isOpened():
        try:
            # 循环读取视频的每一帧
            while True:
                # 读取视频的当前帧
                ret, img = capture.read()

                # 如果参数 gray 为 True，则将图像转换为灰度图像
                if gray:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('uint8')

                # 将读取的帧添加到列表中
                frame_list.append(img)

                # 如果读取帧失败（即到达视频末尾），则退出循环
                if not ret:
                    break
            # 如果循环正常结束（非 break 退出），则打印错误信息
            else:
                print('视频打开失败！')
        # 捕获并忽略任何异常，返回已读取的帧列表
        except:
            return frame_list

    # 返回除最后一帧外的所有帧，因为最后一帧可能是不完整的
    return frame_list[:-1]





def pre_processing(frame, frame_processed=None, img_size=(512, 640, 3)):
    """
    对输入的视频帧进行预处理，包括转换为灰度图、归一化、调整大小以及维护帧序列。

    参数：
    - frame: 输入的视频帧 (numpy array)，形状为 (height, width, channels)。
    - frame_processed: 上一次处理后的帧序列 (numpy array)，形状为 (batch, sequence, height, width)。
    - img_size: 目标图像大小，默认为 (512, 640)。

    返回：
    - frame_processed: 处理后的帧序列 (numpy array)，形状为 (batch, sequence, height, width)。
    - frame: 防止输入形状不是(512, 640)的，将所有的视频帧都规范到(512, 640,3)。

    """

    # 将输入帧转换为灰度图，并进行归一化处理。
    # 从BGR色彩空间转换到灰度可以减少数据的维度，
    # 这有助于提高性能并减少计算负载。

    if (frame.shape != img_size) & (frame.shape[0] <= img_size[0]) & (frame.shape[1] <= img_size[1]):
        frame_ = np.ones(img_size, dtype=np.float32) * 114
        # 将原始帧复制到新的数组中，保持其原有的尺寸。
        frame_[:frame.shape[0], :frame.shape[1], :] = frame
        # 更新帧变量为填充后的版本。
        frame = frame_
    else:
        r_h = frame.shape[0] / img_size[0]
        r_w = frame.shape[1] / img_size[1]
        max_ratio = max(r_h, r_w)
        if (frame.shape == img_size):
            pass
        else:
            frame_ = np.ones(img_size, dtype=np.float32) * 114
            frame_new = cv2.resize(frame, (int(frame.shape[1] // max_ratio), int(frame.shape[0] // max_ratio)))
            frame_[:frame_new.shape[0], :frame_new.shape[1], :] = frame_new
            frame = frame_

    frame_norm = frame.copy()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 将像素值从0-255归一化到0-1以获得更好的数值稳定性。
    frame = frame.astype(np.float32) / 255.0

    # 确保帧的尺寸与期望的尺寸(512, 640)相匹配。
    # 如果帧较小，则使用零填充来匹配所需的尺寸。

    # 对第一帧进行特殊处理。
    if frame_processed is None:
        # 对于第一帧，创建包含8个相同帧的数组。
        # 这通常用于初始化缓冲区中的第一帧。
        frame_processed = np.stack([frame] * 8, axis=0)
        # 在已处理的帧上增加一个新的维度（批次维度）。
        frame_processed = frame_processed[np.newaxis, ...]
    else:
        # 对于后续帧，在缓冲区中移位现有的帧，并添加新的帧。
        # 通过切片移除缓冲区中最旧的帧。
        frame_processed = frame_processed[:, 1:, :, :]
        # 在当前帧上增加两个新的维度以匹配缓冲区的维度。
        new_frame = frame[np.newaxis, np.newaxis, :, :]
        # 沿时间轴将新帧连接到缓冲区的末尾。
        frame_processed = np.concatenate((frame_processed, new_frame), axis=1)

    # 返回处理后的帧或帧序列。
    return frame_processed, frame_norm

def forward_hook(module, inp, outp):  # 定义hook
    feature_map.append(outp)  # 把输出装入字典feature_map



video_name = '00005_CH4_20241030_M3S23_in_A_03___640___560___250.mp4'
frames = Video_read(video_name, gray=None)
model_path = 'Gas_pipdata2k_rep_V66455033_20250331.pt'
model = torch.load(model_path)
net = model['model']
frame_processed = None
net.backbone.register_forward_hook(forward_hook)

for frame in frames:
    frame_processed, frame = pre_processing(frame, frame_processed)
    feature_map = []  # 建立列表容器，用于盛放输出特征图

    net.features.register_forward_hook(forward_hook)



