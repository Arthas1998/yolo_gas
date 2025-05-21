# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid

class DecoupledHead(nn.Module):
    def __init__(self, ch=256, nc=80, anchors=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors

        self.cls_preds = nn.Conv2d(ch, self.nc * self.na, 1)
        self.reg_preds = nn.Conv2d(ch, 4 * self.na, 1)
        self.obj_preds = nn.Conv2d(ch, 1 * self.na, 1)
    def forward(self, x):
        x1 = self.cls_preds(x)
        x21 = self.reg_preds(x)
        x22 = self.obj_preds(x)
        return (x21, x22, x1)
#
class Decoupled_Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(DecoupledHead(x, nc, anchors) for x in ch)
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        # print("x[i]_detect_input:", len(x), x[1].shape, x[0].shape, x[2].shape)
        if torch.onnx.is_in_onnx_export():
            return self.forward_onnx(x)

        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i][0].shape
            # breakpoint()
            reg = x[i][0].view(bs, self.na, 4, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            obj = x[i][1].view(bs, self.na, 1, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            cls = x[i][2].view(bs, self.na, self.nc, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            x[i] = torch.cat([reg, obj, cls], -1)  # 1, 18, h, w   ---> 1, 3, h, w, 6
            # x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # print("x[i]_detect_output:", len(x), x[1].shape, x[0].shape, x[2].shape)
            if not self.training:  # inference
                # print(print("x[i]_______:", len(x), x[1].shape, x[0].shape, x[2].shape))
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))
        # return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        # if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
        yv, xv = torch.meshgrid(y, x, indexing='ij')
        # else:
        #     yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid

    def forward_onnx(self, x):
        lala = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            # bs, _, ny, nx = x[i][0].shape

            # reg = x[i][0].view(bs, self.na, 4, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # obj = x[i][1].view(bs, self.na, 1, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            reg = x[i][0].sigmoid()
            obj = x[i][1].sigmoid()
            # cls 通道没有用   舍弃
            #  cls = x[i][2].view(bs, self.na, self.nc, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # x[i] = torch.cat([reg, obj, cls], -1)  # 1, 18, h, w   ---> 1, 3, h, w, 6
            lala.append(reg)
            lala.append(obj)

        return lala


# class Decoupled_Detect(nn.Module):
#     stride = None  # strides computed during build
#     onnx_dynamic = False  # ONNX export parameter
#     export = False  # export mode
#     heat_map = False
#
#     def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
#         super().__init__()
#         self.nc = nc  # number of classes
#         print(self.nc)
#         self.no = nc + 5  # number of outputs per anchor
#         self.nl = len(anchors)  # number of detection layers
#         self.na = len(anchors[0]) // 2  # number of anchors
#         self.grid = [torch.zeros(1)] * self.nl  # init grid
#         self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
#         self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
#         self.m = nn.ModuleList(DecoupledHead(x, nc, anchors) for x in ch)
#         self.inplace = inplace  # use in-place ops (e.g. slice assignment)
#         self.heatmap = heat_map
#
#     def forward(self, x):
#         if torch.onnx.is_in_onnx_export():
#             return self.forward_onnx(x)
#         z = []  # inference output\
#         heat_map_out = []
#         for i in range(self.nl):
#             if self.heat_map:
#                 heat_map_out.append(x[i].sum(dim=1, keepdim=True))
#
#             x[i] = self.m[i](x[i])  # conv
#             bs, _, ny, nx = x[i][0].shape
#
#             reg = x[i][0].view(bs, self.na, 4, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#             obj = x[i][1].view(bs, self.na, 1, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#             cls = x[i][2].view(bs, self.na, self.nc, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#
#             x[i] = torch.cat([reg, obj, cls], -1)  # 1, 18, h, w   ---> 1, 3, h, w, 6
#             # x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#
#             if not self.training:  # inference
#                 if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
#                     self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
#                 y = x[i].sigmoid()
#                 if self.inplace:
#                     y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
#                     y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
#                 else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
#                     xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
#                     xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
#                     wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
#                     y = torch.cat((xy, wh, conf), 4)
#                 z.append(y.view(bs, -1, self.no))
#         if self.heat_map:
#             for i in range(self.nl):
#                 heat_map_out[i] = torch.nn.functional.interpolate(heat_map_out[i], (512, 640), mode='bilinear')
#
#             heat_map_out = 255 - (0.39 * heat_map_out[0] + 0.32 * heat_map_out[1] + 0.29 * heat_map_out[2])
#             print(heat_map_out.shape)
#             return (x, heat_map_out)
#
#         # return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
#         return x if self.training else (torch.cat(z, 1), x)
#
#     def _make_grid(self, nx=20, ny=20, i=0):
#         d = self.anchors[i].device
#         t = self.anchors[i].dtype
#         shape = 1, self.na, ny, nx, 2  # grid shape
#         y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
#         # if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
#         yv, xv = torch.meshgrid(y, x, indexing='ij')
#         # else:
#         #     yv, xv = torch.meshgrid(y, x)
#         grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
#         anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
#         return grid, anchor_grid
#
#     def forward_onnx(self, x):
#         lala = []
#         print("x:", len(x))
#         heatmap_out = []
#         for i in range(self.nl):
#             if self.heat_map:
#                 heatmap_out.append(x[i].sum(dim=1, keepdim=True))
#             print(x[i].shape)
#             x[i] = self.m[i](x[i])  # conv
#             bs, _, ny, nx = x[i][0].shape
#
#             # reg = x[i][0].view(bs, self.na, 4, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#             # obj = x[i][1].view(bs, self.na, 1, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#
#             reg = x[i][0].sigmoid()
#             obj = x[i][1].sigmoid()
#             # cls 通道没有用   舍弃
#             #  cls = x[i][2].view(bs, self.na, self.nc, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#             # x[i] = torch.cat([reg, obj, cls], -1)  # 1, 18, h, w   ---> 1, 3, h, w, 6
#             lala.append(reg)
#             lala.append(obj)
#         if self.heat_map:
#             for i in range(self.nl):  # 对热图进行插值，调整尺寸
#                 heatmap_out[i] = torch.nn.functional.interpolate(heatmap_out[i], (512, 640), mode='bilinear')
#             heatmap_out = 255 - (0.39 * heatmap_out[0] + 0.32 * heatmap_out[1] + 0.29 * heatmap_out[2])  # 生成最终的热图输出
#             return lala, heatmap_out  # 返回 ONNX 模式下的结果，包括热图输出
#
#         return lala

#联用

# class Decoupled_Detect(nn.Module):
#     # YOLOv8 检测头，用于目标检测模型
#     dynamic = False  # 是否强制重建网格
#     export = False  # 是否处于导出模式
#     shape = None  # 用于动态调整网格的输入形状
#     anchors = torch.empty(0)  # 初始化锚点
#     strides = torch.empty(0)  # 初始化步长
#
#     def __init__(self, nc=80, ch=()):  # 检测层
#         super().__init__()
#         self.nc = nc  # 类别数量
#         self.nl = len(ch)  # 检测层的数量
#         self.reg_max = 16  # DFL 通道数（用于调整不同模型大小的边界框回归）
#         self.no = nc + self.reg_max * 4  # 每个锚点的输出数量（类别 + 边界框）
#         self.stride = torch.zeros(self.nl)  # 在构建过程中计算的步长
#
#         # 计算卷积层的通道数
#         c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)
#         # 创建边界框预测的卷积层
#         self.cv2 = nn.ModuleList(
#             nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
#         # 创建类别预测的卷积层
#         self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
#         # 初始化 DFL 模块，如果 reg_max > 1，则使用 DFL，否则使用 Identity
#         self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
#
#     def forward(self, x):
#         shape = x[0].shape  # 获取输入张量的形状（BCHW）
#         # 遍历每个检测层
#         for i in range(self.nl):
#             # 将边界框和类别预测拼接在一起
#             x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
#         # 如果处于训练模式，直接返回拼接后的预测
#         if self.training:
#             return x
#         # 如果处于动态模式或输入形状发生变化，更新锚点和步长
#         elif self.dynamic or self.shape != shape:
#             self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
#             self.shape = shape
#
#         # 处理导出模式（Edge TPU）
#         if self.export and self.format == 'edgetpu':  # FlexSplitV ops issue
#             x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
#             box = x_cat[:, :self.reg_max * 4]
#             cls = x_cat[:, self.reg_max * 4:]
#         else:
#             # 将拼接后的预测分为边界框和类别预测
#             box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
#         # 使用 DFL 解码边界框预测
#         dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
#         # 将解码后的边界框和类别预测拼接在一起
#         y = torch.cat((dbox, cls.sigmoid()), 1)
#         # 如果处于导出模式，返回最终预测；否则返回预测和输入特征
#         return y if self.export else (y, x)
#
#     def bias_init(self):
#         # 初始化检测头的偏置，注意：需要步长信息
#         m = self  # self.model[-1]  # 检测头模块
#         # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
#         # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # 名义类别频率
#         # 初始化边界框和类别预测的偏置
#         for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
#             a[-1].bias.data[:] = 1.0  # 边界框
#             b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # 类别（0.01 对象，80 类别，640 图像）
#
#
# class DFL(nn.Module):
#     # 分布焦点损失（DFL）的积分模块，提出于《Generalized Focal Loss: Towards Efficient Representation Learning for Dense Object Detection》
#     def __init__(self, c1=16):
#         super().__init__()
#         # 初始化一个 1x1 卷积层，不使用偏置，并禁用梯度
#         self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
#         # 创建一个从 0 到 c1-1 的张量，表示分布的索引
#         x = torch.arange(c1, dtype=torch.float)
#         # 将索引张量作为卷积层的权重
#         self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
#         # 保存通道数
#         self.c1 = c1
#
#     def forward(self, x):
#         # 获取输入张量的形状（batch, channels, anchors）
#         b, c, a = x.shape
#         # 将输入张量重塑为 (batch, 4, c1, anchors)，并对每个通道进行 softmax 操作
#         # 然后通过卷积层将其转换为边界框坐标
#         return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
#         # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

# class Decoupled_Detect(nn.Module):
#     stride = None  # strides computed during build
#     onnx_dynamic = False  # ONNX export parameter
#     export = False  # export mode
#     heat_map = False
#
#     def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
#         super().__init__()
#         self.nc = nc  # number of classes
#         print(self.nc)
#         self.no = nc + 5  # number of outputs per anchor
#         self.nl = len(anchors)  # number of detection layers
#         self.na = len(anchors[0]) // 2  # number of anchors
#         self.grid = [torch.zeros(1)] * self.nl  # init grid
#         self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
#         self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
#         self.m = nn.ModuleList(DecoupledHead(x, nc, anchors) for x in ch)
#         self.inplace = inplace  # use in-place ops (e.g. slice assignment)
#         self.heatmap = heat_map
#
#     def forward(self, x):
#
#
#
#         if torch.onnx.is_in_onnx_export():
#             return self.forward_onnx(x)
#         z = []  # inference output\
#         heat_map_out = []
#         for i in range(self.nl):
#             if self.heat_map:
#                 heat_map_out.append(x[i].sum(dim=1, keepdim=True))
#
#             x[i] = self.m[i](x[i])  # conv
#             bs, _, ny, nx = x[i][0].shape
#
#             reg = x[i][0].view(bs, self.na, 4, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#             obj = x[i][1].view(bs, self.na, 1, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#             cls = x[i][2].view(bs, self.na, self.nc, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#
#             x[i] = torch.cat([reg, obj, cls], -1)  # 1, 18, h, w   ---> 1, 3, h, w, 6
#             # x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#
#             if not self.training:  # inference
#                 if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
#                     self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
#                 y = x[i].sigmoid()
#                 if self.inplace:
#                     y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
#                     y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
#                 else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
#                     xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
#                     xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
#                     wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
#                     y = torch.cat((xy, wh, conf), 4)
#                 z.append(y.view(bs, -1, self.no))
#         if self.heat_map:
#             for i in range(self.nl):
#                 heat_map_out[i] = torch.nn.functional.interpolate(heat_map_out[i], (512, 640), mode='bilinear')
#
#             heat_map_out = 255 - (0.39 * heat_map_out[0] + 0.32 * heat_map_out[1] + 0.29 * heat_map_out[2])
#             print(heat_map_out.shape)
#             return (x, heat_map_out)
#
#         # return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
#         return x if self.training else (torch.cat(z, 1), x)
#
#     def _make_grid(self, nx=20, ny=20, i=0):
#         d = self.anchors[i].device
#         t = self.anchors[i].dtype
#         shape = 1, self.na, ny, nx, 2  # grid shape
#         y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
#         # if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
#         yv, xv = torch.meshgrid(y, x, indexing='ij')
#         # else:
#         #     yv, xv = torch.meshgrid(y, x)
#         grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
#         anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
#         return grid, anchor_grid
#
#     def forward_onnx(self, x):
#         lala = []
#         print("x:", len(x))
#         heatmap_out = []
#         for i in range(self.nl):
#
#
#             if self.heat_map:
#                 heatmap_out.append(x[i].sum(dim=1, keepdim=True))
#             print(x[i].shape)
#             x[i] = self.m[i](x[i])  # conv
#             bs, _, ny, nx = x[i][0].shape
#
#             # reg = x[i][0].view(bs, self.na, 4, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#             # obj = x[i][1].view(bs, self.na, 1, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#
#             reg = x[i][0].sigmoid()
#             obj = x[i][1].sigmoid()
#             # cls 通道没有用   舍弃
#             #  cls = x[i][2].view(bs, self.na, self.nc, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#             # x[i] = torch.cat([reg, obj, cls], -1)  # 1, 18, h, w   ---> 1, 3, h, w, 6
#             lala.append(reg)
#             lala.append(obj)
#         if self.heat_map:
#             for i in range(self.nl):  # 对热图进行插值，调整尺寸
#                 heatmap_out[i] = torch.nn.functional.interpolate(heatmap_out[i], (512, 640), mode='bilinear')
#             heatmap_out = 255 - (0.39 * heatmap_out[0] + 0.32 * heatmap_out[1] + 0.29 * heatmap_out[2])  # 生成最终的热图输出
#             return lala, heatmap_out  # 返回 ONNX 模式下的结果，包括热图输出
#
#         return lala



class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    # YOLOv5 base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            # try:
            #     print("check2:",[m,x.shape])
            # except:
            #     print("check2:", [m, x[0].shape,x[1].shape])

            x = m(x)  # run
            # print("check3:",x.shape )
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    # def fuse(self):
    #     print('Fusing layers...')
    #
    #     for m in self.model.modules():
    #         if isinstance(m, RepConv):
    #             m.fuse_repvgg_block()
    #     elif isinstance(m, RepConv_OREPA):
    #         m.switch_to_deploy()
    #     elif type(m) is Conv and hasattr(m, 'bn'):
    #         m.conv = fuse_conv_and_bn(m.conv, m.bn)
    #         delattr(m, 'bn')
    #         m.forward = m.fuseforward
    #     self.info()
    #     return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    # YOLOv5 detection model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Decoupled_Detect, Segment)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            # self._initialize_biases()  # only run once
            self._initialize_biases_decouple()

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_biases_decouple(self, cf=None):  # initialize biases into DecoupleDetect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # DecoupleDetect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.obj_preds.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 0] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            mi.obj_preds.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

            b = mi.cls_preds.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, :] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.cls_preds.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    def __init__(self, cfg='yolov5s-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # Create a YOLOv5 classification model from a YOLOv5 detection model
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        # Create a YOLOv5 classification model from a *.yaml file
        self.model = None


def parse_model(d, ch):  # model_dict, input_channels(3)
    # Parse a YOLOv5 model.yaml dictionary
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
                Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, PSA, RepVGGBlock, csla, C3_csla,QARepVGGBlock,C3_QARepVGGBlock}:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            elif m is RepVGGBlock:
                c1, c2 = ch[f], args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x,C3_QARepVGGBlock}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect, Decoupled_Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()
