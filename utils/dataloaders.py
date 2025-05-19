# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""

import contextlib
import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torchvision
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm
import copy
from matplotlib import pyplot as plt

# from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
#                                  letterbox, mixup, random_perspective, paste_background_to_image_hard, paste_background_to_image_hard2, paste_negative_to_image)
from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, TQDM_BAR_FORMAT, check_dataset, check_requirements,
                           check_yaml, clean_str, cv2, is_colab, is_kaggle, segments2boxes, unzip_file, xyn2xy,
                           xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first
from utils.metrics import bbox_iou

# Parameters
HELP_URL = 'See https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

# def check(imgs):
#     random_num = random.randint(1,1234567892222)
#     print("check2:",len(imgs),imgs[0].shape)
#     dir_path = f"check_image_/{random_num}"
#
#     # 创建新的目录
#     if not os.path.exists(dir_path):
#         os.makedirs(dir_path, exist_ok=True)
#     for frame_num in range(len(imgs)):
#         frame = imgs[frame_num].astype('uint8')
#         cv2.imwrite(f"{dir_path}//{frame_num:05}.jpg", frame)

def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    # Returns exif-corrected PIL size
    if len(img) == 2:
        s = img
    else:
        s = img.size  # (width, height)

    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90}.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def seed_worker(worker_id):
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def creat_dataset(path,
                  imgsz,
                  batch_size,
                  stride,
                  augment=False,  # augmentation
                  hyp=None,  # hyperparameters
                  rect=False,  # rectangular batches
                  rank=-1,
                  cache=False,
                  single_cls=False,
                  pad=0.0,
                  image_weights=False,
                  prefix='',
                  k=8):
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabels(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            k=k)
    return dataset


def create_dataloader(dataset,
                      imgsz,
                      batch_size,
                      stride,
                      single_cls=False,
                      hyp=None,
                      augment=False,
                      cache=False,
                      pad=0.0,
                      rect=False,
                      rank=-1,
                      workers=8,
                      image_weights=False,
                      quad=False,
                      prefix='',
                      shuffle=False):
    if rect and shuffle:
        LOGGER.warning('WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=PIN_MEMORY,
                  collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn,
                  worker_init_fn=seed_worker,
                  generator=generator)


def create_dataloader_2(path,
                        imgsz,
                        batch_size,
                        stride,
                        single_cls=False,
                        hyp=None,
                        augment=False,
                        cache=False,
                        pad=0.0,
                        rect=False,
                        rank=-1,
                        workers=8,
                        image_weights=False,
                        quad=False,
                        prefix='',
                        shuffle=False,
                        k=8):
    if rect and shuffle:
        LOGGER.warning('WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabels(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            k=k)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    # generator.manual_seed(int(time.time()*1000) + RANK)
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=PIN_MEMORY,
                  collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn,
                  worker_init_fn=seed_worker,
                  generator=generator), dataset


class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadScreenshots:
    # YOLOv5 screenshot dataloader, i.e. `python detect.py --source "screen 0 100 100 512 256"`
    def __init__(self, source, img_size=640, stride=32, auto=True, transforms=None):
        # source = [screen_number left top width height] (pixels)
        check_requirements('mss')
        import mss

        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None  # default to full screen 0
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)
        self.img_size = img_size
        self.stride = stride
        self.transforms = transforms
        self.auto = auto
        self.mode = 'stream'
        self.frame = 0
        self.sct = mss.mss()

        # Parse monitor shape
        monitor = self.sct.monitors[self.screen]
        self.top = monitor["top"] if top is None else (monitor["top"] + top)
        self.left = monitor["left"] if left is None else (monitor["left"] + left)
        self.width = width or monitor["width"]
        self.height = height or monitor["height"]
        self.monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def __iter__(self):
        return self

    def __next__(self):
        # mss screen capture: get raw pixels from the screen as np array
        im0 = np.array(self.sct.grab(self.monitor))[:, :, :3]  # [:, :, :3] BGRA to BGR
        s = f"screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: "

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
        self.frame += 1
        return str(self.screen), im, im0, None, s  # screen, img, original img, im0s, s


# class LoadImages:
#     # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
#     def __init__(self, path, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
#         files = []
#         for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
#             p = str(Path(p).resolve())
#             if '*' in p:
#                 files.extend(sorted(glob.glob(p, recursive=True)))  # glob
#             elif os.path.isdir(p):
#                 files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
#             elif os.path.isfile(p):
#                 files.append(p)  # files
#             else:
#                 raise FileNotFoundError(f'{p} does not exist')
#
#         images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
#         videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
#         ni, nv = len(images), len(videos)
#
#         self.img_size = img_size
#         self.stride = stride
#         self.files = images + videos
#         self.nf = ni + nv  # number of files
#         self.video_flag = [False] * ni + [True] * nv
#         self.mode = 'image'
#         self.auto = auto
#         self.transforms = transforms  # optional
#         self.vid_stride = vid_stride  # video frame-rate stride
#         if any(videos):
#             self._new_video(videos[0])  # new video
#         else:
#             self.cap = None
#         assert self.nf > 0, f'No images or videos found in {p}. ' \
#                             f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'
#
#     def __iter__(self):
#         self.count = 0
#         return self
#
#     def __next__(self):
#         if self.count == self.nf:
#             raise StopIteration
#         path = self.files[self.count]
#
#         if self.video_flag[self.count]:
#             # Read video
#             self.mode = 'video'
#             for _ in range(self.vid_stride):
#                 self.cap.grab()
#             ret_val, im0 = self.cap.retrieve()
#             while not ret_val:
#                 self.count += 1
#                 self.cap.release()
#                 if self.count == self.nf:  # last video
#                     raise StopIteration
#                 path = self.files[self.count]
#                 self._new_video(path)
#                 ret_val, im0 = self.cap.read()
#
#             self.frame += 1
#             # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
#             s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '
#
#         else:
#             # Read image
#             self.count += 1
#             im0 = cv2.imread(path)  # BGR
#             assert im0 is not None, f'Image Not Found {path}'
#             s = f'image {self.count}/{self.nf} {path}: '
#
#         if self.transforms:
#             im = self.transforms(im0)  # transforms
#         else:
#             im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
#             im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
#             im = np.ascontiguousarray(im)  # contiguous
#
#         return path, im, im0, self.cap, s
#
#     def _new_video(self, path):
#         # Create a new video capture object
#         self.frame = 0
#         self.cap = cv2.VideoCapture(path)
#         self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
#         self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees
#         # self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # disable https://github.com/ultralytics/yolov5/issues/8493
#
#     def _cv2_rotate(self, im):
#         # Rotate a cv2 video manually
#         if self.orientation == 0:
#             return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
#         elif self.orientation == 180:
#             return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
#         elif self.orientation == 90:
#             return cv2.rotate(im, cv2.ROTATE_180)
#         return im
#
#     def __len__(self):
#         return self.nf  # number of files


class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1, k=8):
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        self.K = k
        self.img0list = []
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1
            # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            assert im0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '
        if self.frame == 1:  # 首帧
            # self.img0list = [im0] * self.K
            self.img0list = [im0] * 8
        else:
            del self.img0list[0]
            self.img0list.append(im0)
        imglist = copy.deepcopy(self.img0list)
        if self.transforms:
            # im = self.transforms(im0)  # transforms
            for i in range(len(imglist)):
                imglist[i] = self.transforms(imglist[i])
                imglist[i] = cv2.cvtColor(imglist[i], cv2.COLOR_BGR2GRAY)
        else:
            # im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
            # im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            # im = np.ascontiguousarray(im)  # contiguous
            for i in range(len(imglist)):
                imglist[i] = cv2.cvtColor(imglist[i], cv2.COLOR_BGR2GRAY)
            imglist = letterbox(imglist, self.img_size, color=114, stride=self.stride, auto=self.auto)[0]

        # return path, im, im0, self.cap, s
        return path, np.stack(imglist, 0), im0, self.cap, s

    def _new_video(self, path):
        # Create a new video capture object
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees
        # self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # disable https://github.com/ultralytics/yolov5/issues/8493

    def _cv2_rotate(self, im):
        # Rotate a cv2 video manually
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        return self.nf  # number of files


class LoadStreams:
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.vid_stride = vid_stride  # video frame-rate stride
        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            if urlparse(s).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):  # if source is YouTube video
                # YouTube format i.e. 'https://www.youtube.com/watch?v=Zgi9g1ksQHc' or 'https://youtu.be/Zgi9g1ksQHc'
                check_requirements(('pafy', 'youtube_dl==2020.12.2'))
                import pafy
                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            if s == 0:
                assert not is_colab(), '--source 0 webcam unsupported on Colab. Rerun command in a local environment.'
                assert not is_kaggle(), '--source 0 webcam unsupported on Kaggle. Rerun command in a local environment.'
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'{st}Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            LOGGER.info(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, img_size, stride=stride, auto=auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        self.auto = auto and self.rect
        self.transforms = transforms  # optional
        if not self.rect:
            LOGGER.warning('WARNING ⚠️ Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        n, f = 0, self.frames[i]  # frame number, frame array
        while cap.isOpened() and n < f:
            n += 1
            cap.grab()  # .read() = .grab() followed by .retrieve()
            if n % self.vid_stride == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning('WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(0.0)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        im0 = self.imgs.copy()
        if self.transforms:
            im = np.stack([self.transforms(x) for x in im0])  # transforms
        else:
            im = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0] for x in im0])  # resize
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(im)  # contiguous

        return self.sources, im, im0, None, ''

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}Frames{os.sep}', f'{os.sep}Labels_yolo_zhaoxiaoqian{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


class LoadImagesAndLabels(Dataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix='',
                 k=8):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        # self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.mosaic_border = [-img_size[0] // 2, -img_size[1] // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations(size=img_size) if augment else None
        self.K = k
        self.negatives = []

        # print(self.path,self.img_size, self.augment,self.hyp,self.image_weights,self.rect,self.mosaic,self.mosaic_border,self.stride,self.albumentations,self.K, self.negatives  )

        # try:
        f = []  # 初始化一个空列表，用于存储找到的图像文件路径
        # 检查path是否是列表，如果不是，则将其转换为单元素列表
        for p in path if isinstance(path, list) else [path]:
            p = Path(p)  # 将路径转换为Path对象，以实现操作系统无关性
            # print(p,p.is_dir())
            if p.is_dir():  # 如果路径是一个目录
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)  # 使用glob模式匹配所有文件，递归搜索
                # f = list(p.rglob('*.*'))  # 使用pathlib的rglob方法递归搜索所有文件（pathlib方式）
            elif p.is_file():  # 如果路径是一个文件
                with open(p) as t:  # 打开文件
                    t = t.read().strip().splitlines()  # 读取文件内容，去除空白字符，按行分割
                    parent = str(p.parent) + os.sep  # 获取文件的父目录路径，并添加操作系统特定的路径分隔符
                    # 将文件中的相对路径转换为绝对路径
                    f += [x.replace('./', parent, 1) if x.startswith('./') else x for x in t]
                    # 新列表，用于存储所有帧的路径
                    new_f = []

                    # 遍历原始列表中的每个视频文件路径
                    for video_path in f:
                        video_name = video_path.split('___')[0]
                        width = video_path.split('___')[1]
                        Height = video_path.split('___')[2]
                        total_num = video_path.split('___')[3]
                        frame_count = int(video_path.split('___')[-1].split('.mp4')[0])
                        #     # 为每个视频创建所有帧的路径
                        for i in range(frame_count):
                            frame_filename = f"{video_name}___{width}___{Height}___{total_num}____{(i + 1):05d}.mp4"
                            new_f.append(frame_filename)
                            # print(frame_filename,frame_count)
                    f = new_f
                    # f += [p.parent / x.lstrip(os.sep) for x in t]  # 使用pathlib方式转换路径
            else:
                raise FileNotFoundError(f'{prefix}{p} does not exist')  # 如果路径既不是文件也不是目录，抛出文件不存在错误

        self.vid_files = sorted(
            x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in VID_FORMATS)  # 筛选出视频文件，并排序
        # self.im_files = sorted(
        #     x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)  # 筛选出图像文件，并排序
        # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # 使用pathlib方式筛选图像文件
        assert self.vid_files, f'{prefix}No images found'  # 断言确保至少找到了一个图像文件
        # except Exception as e:  # 如果在尝试过程中发生任何异常
        #     print(self.path, self.img_size, self.augment, self.hyp, self.image_weights, self.rect, self.mosaic,
        #           self.mosaic_border, self.stride, self.albumentations, self.K, self.negatives)  # 打印一些调试信息
        #     raise Exception(f'{prefix}Error loading data from {path}: {e}\n{HELP_URL}') from e  # 重新抛出异常，并提供错误信息和帮助链接

        # 滤掉每个视频最前面的k-1帧
        topk_names = []
        for i in range(self.K):
            topk_names.append('{:0>5}.mp4'.format(i))
        self.vid_files = [x for x in self.vid_files if x.rsplit('____', 1)[-1] not in topk_names]
        # Check cache


        self.label_files = img2label_paths(self.vid_files)  # labels
        # print("check:", self.label_files, self.vid_files)



        # # 处理后的文件路径列表
        # new_file_paths = [
        #     os.path.join(
        #         os.sep.join(
        #             # 替换前一个文件夹名称
        #             ['Labels_yolo_zhaoxiaoqian' if i == 'videos_test_set' else i for i in
        #              os.path.dirname(file_path).split('/')]
        #         ),
        #         # 构造新的文件名
        #         f"{os.path.basename(file_path).split('___')[0]}/{os.path.basename(file_path).split('___')[-1]}"
        #     )
        #     for file_path in self.label_files
        #     if 'TrueLeakedGas' in self.label_files
        # ]
        # print(new_file_paths)

        # 处理后的文件路径列表
        new_file_paths = []

        # 遍历原始列表中的每个文件路径
        for file_path in self.label_files:
            # 将路径分割为目录和文件名
            directory, filename = os.path.split(file_path)
            # 获取路径的各个部分
            path_parts = directory.split("/")
            # 找到 "TrueLeakedGas" 的索引

            true_leaked_gas_index = path_parts.index('Videos')
            path_parts[true_leaked_gas_index] = 'Annotations'

            # # 替换前一个文件夹名称
            # if true_leaked_gas_index > 0:
            #     path_parts[true_leaked_gas_index - 1] = 'Labels_yolo_zhaoxiaoqian'

            # 重新组合路径
            new_path = os.sep.join(path_parts)

            # 将文件名分割为文件名前缀和数字部分
            filename_parts = filename.split('___')
            num_parts = filename.split('____')[-1]
            # 构造新的文件名，去掉额外的下划线和数字
            new_filename = f"{filename_parts[0]}/{num_parts}"

            # 构造新的文件路径
            new_file_path = os.path.join(new_path, new_filename)
            # 将新的文件路径添加到列表中
            new_file_paths.append(new_file_path)

        self.label_files = new_file_paths
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')

        # 尝试从指定的缓存路径 cache_path 加载缓存数据
        try:
            # 使用 numpy 的 load 函数加载缓存文件，该文件包含一个字典对象
            # allow_pickle=True 允许加载 pickle 序列化的对象
            # item() 方法将 numpy 的 ndarray 对象转换为 Python 的字典对象
            # 同时将 exists 设置为 True，表示缓存文件存在且已成功加载
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True

            # 断言操作，检查缓存字典中的版本号是否与当前版本号匹配
            # 如果不匹配，将引发 AssertionError 异常
            assert cache['version'] == self.cache_version  # matches current version

            # 再次使用断言检查缓存字典的哈希值是否与当前标签文件和图像文件的哈希值相同
            # 如果不相同，也将引发 AssertionError 异常
            # get_hash 函数用于计算文件列表的哈希值，确保缓存数据与当前数据一致
            assert cache['hash'] == get_hash(self.label_files + self.vid_files)  # identical hash

        # 如果在尝试加载缓存文件的过程中发生任何异常（例如文件不存在、版本不匹配或哈希值不同）
        # 则执行以下备份操作
        except Exception:
            # 调用 self.cache_labels 方法来重新生成缓存文件
            # 并将 exists 设置为 False，表示缓存文件不存在或需要重新生成
            cache, exists = self.cache_labels(cache_path, prefix), False

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        assert nf > 0 or not augment, f'{prefix}No labels found in {cache_path}, can not start training. {HELP_URL}'

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        # 从缓存中移除不需要的键值对，如 'hash', 'version', 'msgs' 等，以便后续处理。

        labels, shapes, self.segments = zip(*cache.values())
        # 将缓存中的值解压成三个列表：标签、图像形状和图像分割信息。

        nl = len(np.concatenate(labels, 0))  # number of labels
        # 计算所有图像标签的总数，通过将所有标签数组连接起来并计算其长度。

        assert nl > 0 or not augment, f'{prefix}All labels empty in {cache_path}, can not start training. {HELP_URL}'
        # 断言：如果数据增强开启，则必须有标签数据。否则，抛出异常并提示无法开始训练。

        self.labels = list(labels)
        # 将标签列表赋值给类的标签属性。

        self.shapes = np.array(shapes)
        # 将图像形状列表转换为 NumPy 数组，并赋值给类的形状属性。

        self.im_files = list(cache.keys())  # update
        # 更新图像文件路径列表，使用缓存中的键值。

        self.label_files = img2label_paths(cache.keys())  # update
        # 更新标签文件路径列表，通过将图像路径转换为对应的标签路径。

        # Filter images
        if min_items:
            include = np.array([len(x) >= min_items for x in self.labels]).nonzero()[0].astype(int)
            # 如果设置了最小标签数量，则筛选出标签数量大于或等于最小值的图像索引。
            LOGGER.info(f'{prefix}{n - len(include)}/{n} images filtered from dataset')
            # 记录筛选信息，显示被过滤掉的图像数量。
            self.im_files = [self.im_files[i] for i in include]
            # 更新图像文件路径列表，仅保留符合条件的图像路径。
            self.label_files = [self.label_files[i] for i in include]
            # 更新标签文件路径列表，仅保留符合条件的标签路径。
            self.labels = [self.labels[i] for i in include]
            # 更新标签列表，仅保留符合条件的标签。
            self.segments = [self.segments[i] for i in include]
            # 更新图像分割信息列表，仅保留符合条件的分割信息。
            self.shapes = self.shapes[include]  # wh
            # 更新图像形状数组，仅保留符合条件的图像形状。

        # Create indices
        n = len(self.shapes)  # number of images
        # 计算图像数量，即形状数组的长度。

        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        # 计算每个图像的批次索引，通过将图像索引除以批次大小并向下取整得到。

        nb = bi[-1] + 1  # number of batches
        # 计算批次总数，即最后一个图像的批次索引加一。

        self.batch = bi  # batch index of image
        # 将批次索引数组赋值给类的批次属性。

        self.n = n
        # 将图像数量赋值给类的 n 属性。

        self.indices = range(n)
        # 创建图像索引范围，从 0 到 n-1。
        # print("check:-------------------------------------------------------------------------------------------------------------")
        # print("check:", [n, len(self.shapes), len(self.indices),nl,len(self.labels),len(self.im_files),len(self.label_files)])
        # 打印检查信息，显示图像数量、形状数组长度和图像索引范围。




        # Update labels
        include_class = [0]  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = segment[j]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0
                if segment:
                    self.segments[i][:, 0] = 0

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.vid_files = [self.vid_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.segments = [self.segments[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

        # Cache images into RAM/disk for faster training
        if cache_images == 'ram' and not self.check_cache_ram(prefix=prefix):
            cache_images = False
        self.ims = [None] * n
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        if cache_images:
            b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
            self.im_hw0, self.im_hw = [None] * n, [None] * n
            fcn = self.cache_images_to_disk if cache_images == 'disk' else self.load_image
            results = ThreadPool(NUM_THREADS).imap(fcn, range(n))
            pbar = tqdm(enumerate(results), total=n, bar_format=TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache_images == 'disk':
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f'{prefix}Caching images ({b / gb:.1f}GB {cache_images})'
            pbar.close()

        # # 筛选出纯气瓶的数据文件和对应的标签文件
        # self.im_files_gasbottle = [x for x in self.im_files if 'gasbottle' in x.rsplit('/', 2)[1]]
        # self.label_files_gasbottle = img2label_paths(self.im_files_gasbottle)  # labels_gasbottle
        #
        # # 筛选出纯黑体的数据文件和对应的标签文件
        # self.im_files_heiti = [x for x in self.im_files if 'heiti' in x.rsplit('/', 2)[1]]
        # self.label_files_heiti = img2label_paths(self.im_files_heiti)  # labels_heiti
        #
        # # 筛选出纯加热板的数据文件和对应的标签文件
        # self.im_files_hotplate = [x for x in self.im_files if 'hotplate' in x.rsplit('/', 2)[1]]
        # self.label_files_hotplate = img2label_paths(self.im_files_hotplate)  # labels_hotplate

    def check_cache_ram(self, safety_margin=0.1, prefix=''):
        # Check image caching requirements vs available memory
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.n, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            # ratio = self.img_size / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            ratio = min(self.img_size[0] / im.shape[0], self.img_size[1] / im.shape[1])
            b += im.nbytes * ratio ** 2
        mem_required = b * self.n / n  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required * (1 + safety_margin) < mem.available  # to cache or not to cache, that is the question
        if not cache:
            LOGGER.info(f"{prefix}{mem_required / gb:.1f}GB RAM required, "
                        f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, "
                        f"{'caching images ✅' if cache else 'not caching images ⚠️'}")
        return cache

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # 缓存数据集标签，检查图像并读取形状
        x = {}  # 用于存储缓存数据的字典
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # 初始化统计变量和消息列表
        # nm: missing, nf: found, ne: empty, nc: corrupt, msgs: messages
        desc = f"{prefix}Scanning {path.parent / path.stem}..."  # 进度条描述
        with Pool(NUM_THREADS) as pool:  # 使用进程池来并行处理数据
            # 使用tqdm显示进度条，imap函数从pool中映射verify_image_label函数到数据集
            # print("check1:", [self.vid_files,"...", self.label_files,"...", repeat(prefix)])

            pbar = tqdm(pool.imap(verify_image_label, zip(self.vid_files, self.label_files, repeat(prefix))),
                        desc=desc,  # 进度条描述
                        total=len(self.vid_files),  # 总进度
                        bar_format=TQDM_BAR_FORMAT)  # 进度条格式

            for vid_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:  # 遍历进度条
                # print("check1:", [vid_file,"...", lb,"...", shape,"...", segments,"...", nm_f,"...", nf_f,"...", ne_f,"...", nc_f,"...", msg])
                nm += nm_f  # 更新统计数据
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if vid_file:  # 如果有伪图像文件
                    x[vid_file] = [lb, shape, segments]  # 将标签和形状信息存储到字典
                if msg:  # 如果有消息
                    msgs.append(msg)  # 将消息添加到列表
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"  # 更新进度条描述

            pbar.close()  # 关闭进度条
            if msgs:  # 如果有消息
                LOGGER.info('\n'.join(msgs))  # 记录消息
            if nf == 0:  # 如果没有找到标签
                LOGGER.warning(f'{prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}')  # 记录警告

        # 将文件列表的哈希值、结果统计、消息、版本号添加到缓存字典
        x['hash'] = get_hash(self.label_files + self.vid_files)
        x['results'] = nf, nm, ne, nc, len(self.vid_files)
        x['msgs'] = msgs  # 警告消息
        x['version'] = self.cache_version  # 缓存版本
        # print("check2:",x)
        try:
            np.save(path, x)  # 保存缓存字典到文件
            path.with_suffix('.cache.npy').rename(path)  # 重命名文件，移除.npy后缀
            LOGGER.info(f'{prefix}New cache created: {path}')  # 记录缓存创建信息
        except Exception as e:  # 如果保存缓存时发生异常
            LOGGER.warning(f'{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable: {e}')  # 记录警告信息
        return x  # 返回缓存字典

    def __len__(self):
        return len(self.vid_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        if index > len(self.indices):
            print("必须要进行corrupt的检查！！！！！！")



        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            imglist, labels = self.load_mosaic(index)
            # print(len(imglist),imglist[0].shape,len(labels),labels,index)
            #

            shapes = None

            # MixUp augmentation
            if random.random() < hyp['mixup']:
                # img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))
                imglist, labels = mixup(imglist, labels, *self.load_mosaic(random.randint(0, self.n - 1)))
        else:
            # Load image
            # img, (h0, w0), (h, w) = self.load_image(index)
            imglist_orig, imglist, (h0, w0), (h, w) = self.load_imglist(index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            # img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            # print(len(imglist),imglist[0].shape)
            # img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]
            imglist, ratio, pad = letterbox(imglist, shape, color=114, auto=False, scaleup=self.augment)
            # for img in imglist:
            #     img, ratio, pad = letterbox(img, shape, color=114, auto=False, scaleup=self.augment)

            # print("change:",len(imglist), imglist[0].shape)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                imglist, labels = random_perspective(imglist,
                                                     labels,
                                                     degrees=hyp['degrees'],
                                                     translate=hyp['translate'],
                                                     scale=hyp['scale'],
                                                     shear=hyp['shear'],
                                                     perspective=hyp['perspective'])
        # if self.augment:
        # 粘贴其他连续帧上的气体到本连续帧上，并更新labels

        # 随机粘贴背景负样本
        # if random.random() < hyp['paste_negative']:
        # # 两种负样本粘贴策略随机选择其中一种
        # if random.random() < 0.5:
        #     # 根据平均每个像素点在时间维度上的方差选择具有混淆能力的负样本
        #     imglist, labels = paste_background_to_image_hard(imglist, labels, self.im_files, self.label_files, self.K,
        #                                                      max_trials=hyp['max_trials'],
        #                                                      overlapThresh=float(hyp['overlapThresh']),
        #                                                      stdThresh=hyp['stdThresh'])
        # else:
        #     # 上一个训练epoch中训练数据集上预测错误的框作为负样本集，从中选择粘贴
        #     imglist, labels = paste_background_to_image_hard2(imglist, labels, self.negatives,
        #                                                       max_trials=hyp['max_trials'],
        #                                                       overlapThresh=float(hyp['overlapThresh']))

        # # 只选择根据方差选择具有混淆能力的负样本
        # imglist, labels = paste_background_to_image_hard(imglist, labels, self.im_files, self.label_files, self.K,
        #                                                  max_trials=hyp['max_trials'],
        #                                                  overlapThresh=float(hyp['overlapThresh']),
        #                                                  stdThresh=hyp['stdThresh'])

        # 只选择将上一个训练epoch中训练数据集上预测错误的框作为负样本集，从中选择粘贴
        # imglist, labels = paste_background_to_image_hard2(imglist, labels, self.negatives,
        #                                                   max_trials=hyp['max_trials'],
        #                                                   overlapThresh=float(hyp['overlapThresh']))

        ###########
        # # 随机粘贴气瓶
        # if random.random() < hyp['paste_gasbottle']:
        #     imglist, labels = paste_negative_to_image(imglist, labels, self.im_files_gasbottle,
        #                                                self.label_files_gasbottle, self.K,
        #                                                max_trials=hyp['max_trials'],
        #                                                max_expand_ratio=hyp['max_expand_ratio'],
        #                                                overlapThresh=float(hyp['overlapThresh']))
        # # 随机粘贴黑体
        # if random.random() < hyp['paste_heiti']:
        #     imglist, labels = paste_negative_to_image(imglist, labels, self.im_files_heiti,
        #                                                self.label_files_heiti, self.K,
        #                                                max_trials=hyp['max_trials'],
        #                                                max_expand_ratio=hyp['max_expand_ratio'],
        #                                                overlapThresh=float(hyp['overlapThresh']))
        #
        # # 随机粘贴黑体
        # if random.random() < hyp['paste_hotplate']:
        #     imglist, labels = paste_negative_to_image(imglist, labels, self.im_files_hotplate,
        #                                               self.label_files_hotplate, self.K,
        #                                               max_trials=hyp['max_trials'],
        #                                               max_expand_ratio=hyp['max_expand_ratio'],
        #                                               overlapThresh=float(hyp['overlapThresh']))
        ###########
        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=imglist[0].shape[1], h=imglist[0].shape[0], clip=True,
                                        eps=1E-3)

        if self.augment:
            # Albumentations
            # img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations
            r = random.randint(0,256)
            # colour_change
            # if random.random() < hyp['hsv_v']:
            #     for i in range(len(imglist)):
            #         imglist[i] = augment_hsv(imglist[i],r)
            #     #check(imglist)



            # augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                for i in range(len(imglist)):
                    imglist[i] = np.flipud(imglist[i])
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                for i in range(len(imglist)):
                    imglist[i] = np.fliplr(imglist[i])
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            if random.random() < hyp['hsv_v']:
                random_int = int(np.random.rand() * 255)

                for i in range(len(imglist)):
                    imglist[i] = cv2.absdiff(random_int,imglist[i])

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

            # 保存带有labels的imglist以验证上面的粘贴增强是否正确
            # end_index = int(self.im_files[index].split('/')[-1].split('.')[0])
            # name = self.im_files[index].split('/')[-2]
            # if labels.size:
            #     labels_show = xywhn2xyxy(labels[:, 1:], imglist[0].shape[1], imglist[0].shape[0])
            # dpi = 64
            # # if self.negatives:
            # for i in range(len(imglist)):
            #     image=imglist[i]
            #     image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            #     height, width, nbands = image_rgb.shape
            #     figsize = width / float(dpi), height / float(dpi)
            #     fig = plt.figure(figsize=figsize, dpi=float(dpi))
            #     ax = fig.add_axes([0, 0, 1, 1])
            #     ax.axis('off')
            #     ax.imshow(image_rgb, interpolation='nearest')
            #     if not labels.size:
            #         plt.savefig('tmp/' + name + '_' + '{:0>5}.png'.format(end_index-self.K + 1 + i))
            #         plt.close()
            #         continue
            #     for lb in labels_show:
            #         x1, y1, x2, y2 = lb[0], lb[1], lb[2], lb[3]
            #         ax.add_patch(plt.Rectangle(
            #             (x1, y1),
            #             x2 - x1,
            #             y2 - y1,
            #             fill=False,
            #             edgecolor='yellow',
            #             linewidth=2))
            #         plt.savefig('tmp/' + name + '_' + '{:0>5}.png'.format(end_index-self.K + 1 + i))
            #     plt.close()

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # for img in imglist:
        #     img = img.transpose((2, 0, 1))
        #     img = np.ascontiguousarray(img)

        # return torch.from_numpy(img), labels_out, self.im_files[index], shapes
        if mosaic:
            return None, torch.from_numpy(np.stack(imglist, 0)), labels_out, self.im_files[index], shapes
        else:
            return torch.from_numpy(np.stack(imglist_orig, 0)), torch.from_numpy(np.stack(imglist, 0)), labels_out, \
                   self.im_files[index], shapes  # torch.from_numpy(np.stack(imglist, 0)):tensor(K,H,W)

    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i],
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                assert im is not None, f'Image Not Found {f}'
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def load_imglist(self, i):
        # 获取当前视频文件的路径
        f = self.vid_files[i]

        # 从文件名中提取结束索引
        end_index = int(f.split('____')[-1].split('.')[0])
        imglist = []  # 初始化一个空列表，用于存储加载的图像

        # 从一个大的列表中选择8帧
        if self.K == 8:
            # 如果K等于8，直接选择前8帧
            selected_numbers = range(self.K)
        else:
            # 否则，从1到K-1中随机选择7个数字，并将0添加到列表中
            numbers = range(1, self.K)
            selected_numbers = random.sample(numbers, 7)
            selected_numbers.append(0)
            selected_numbers.sort(reverse=False)  # 对选择的数字进行排序


        # print("check1:", [i, f, end_index,selected_numbers])  # 打印当前索引和视频文件路径，用于调试
        video_path = f.split('____')[0]
        # # 创建VideoCapture对象
        cap = cv2.VideoCapture(video_path)

        # # 获取视频的宽度和高度
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #
        # # 初始化一个列表来存储提取的帧
        # tensor_2 = np.zeros((frame_height, frame_width, 8), dtype=np.uint8)

        reversed_imglist = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, end_index - selected_numbers[-1]-1)


        # count = 0
        for frame_number in range(selected_numbers[-1] +1):
            ret, frame = cap.read()
            # print("ret:", cv2.CAP_PROP_POS_FRAMES,frame_number, end_index, selected_numbers, ret, frame.shape)

            if (selected_numbers[-1] - frame_number) in selected_numbers:
                # tensor_2[:, :, count] = frame[:, :, 0]
                # count += 1
                try:
                    # print("checks: ", [video_path], os.path.exists(video_path), ret, cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    reversed_imglist.append(frame[:, :, 0])
                except:

                    # print("checks: ", [video_path],os.path.exists(video_path),ret,cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    # print(frame.shape)
                    print("checks: ", [video_path])
        # print(len(reversed_imglist))
        # # 固定选择8帧
        # for i in selected_numbers:
        #     # 构造图像文件的路径，并加载图像
        #     img_path = os.path.join(f.rsplit('/', 1)[0], '{:0>5}.jpg'.format(end_index - i))
        #     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图像
        #     imglist.append(img)  # 将加载的图像添加到列表中

        # 将图像列表反转
        h0, w0 = frame_height, frame_width

        #######################
        # print(len(reversed_imglist))
        # print(reversed_imglist[0].shape)
        # try: reversed_imglist = stabilize_video(reversed_imglist)
        # except: print(len(reversed_imglist),reversed_imglist[0].shape)
        ########################





        # reversed_imglist = imglist[::-1]
        # h0, w0 = reversed_imglist[0].shape[:2]  # 获取原始图像的高度和宽度
        # imglist_orig = copy.deepcopy(reversed_imglist)  # 创建原始图像列表的深拷贝
        imglist_orig = copy.deepcopy(reversed_imglist)
        # 计算图像的缩放比例

        r = min(self.img_size[0] / h0, self.img_size[1] / w0)
        if r != 1:  # 如果需要缩放
            interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
            for j in range(len(reversed_imglist)):
                # 对图像进行缩放
                reversed_imglist[j] = cv2.resize(reversed_imglist[j], (int(w0 * r), int(h0 * r)), interpolation=interp)

        # 返回原始图像列表、缩放后的图像列表、原始尺寸和缩放后的尺寸
        return imglist_orig, reversed_imglist, (h0, w0), reversed_imglist[0].shape[:2]

    def cache_images_to_disk(self, i):
        # Saves an image as an *.npy file for faster loading
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.vid_files[i]))


    # def load_mosaic(self, index):
    #     # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
    #     labels4, segments4 = [], []
    #     s = self.img_size
    #     yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
    #     indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    #     random.shuffle(indices)
    #     for i, index in enumerate(indices):
    #         # Load image
    #         img, _, (h, w) = self.load_image(index)
    #
    #         # place img in img4
    #         if i == 0:  # top left
    #             img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
    #             x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
    #             x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
    #         elif i == 1:  # top right
    #             x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
    #             x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
    #         elif i == 2:  # bottom left
    #             x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
    #             x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
    #         elif i == 3:  # bottom right
    #             x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
    #             x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
    #
    #         img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
    #         padw = x1a - x1b
    #         padh = y1a - y1b
    #
    #         # Labels
    #         labels, segments = self.labels[index].copy(), self.segments[index].copy()
    #         if labels.size:
    #             labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
    #             segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
    #         labels4.append(labels)
    #         segments4.extend(segments)
    #
    #     # Concat/clip labels
    #     labels4 = np.concatenate(labels4, 0)
    #     for x in (labels4[:, 1:], *segments4):
    #         np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    #     # img4, labels4 = replicate(img4, labels4)  # replicate
    #
    #     # Augment
    #     img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
    #     img4, labels4 = random_perspective(img4,
    #                                        labels4,
    #                                        segments4,
    #                                        degrees=self.hyp['degrees'],
    #                                        translate=self.hyp['translate'],
    #                                        scale=self.hyp['scale'],
    #                                        shear=self.hyp['shear'],
    #                                        perspective=self.hyp['perspective'],
    #                                        border=self.mosaic_border)  # border to remove
    #
    #     return img4, labels4

    def load_mosaic(self, index):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4, segments4 = [], []
        s = self.img_size
        # yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        yc, xc = (int(random.uniform(-x, 2 * y + x)) for x, y in zip(self.mosaic_border, s))  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            imglist_orig, imglist, _, (h, w) = self.load_imglist(index)

            # place img in img4
            if i == 0:  # top left
                img4list = [np.full((s[0] * 2, s[1] * 2), 114, dtype=np.uint8) for _ in
                            range(len(imglist))]  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s[1] * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s[0] * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s[1] * 2), min(s[0] * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            for j in range(len(img4list)):
                img4list[j][y1a:y2a, x1a:x2a] = imglist[j][y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:5:2], *segments4):
            np.clip(x, 0, 2 * s[1], out=x)  # clip when using random_perspective()
        for x in (labels4[:, 2:5:2], *segments4):
            np.clip(x, 0, 2 * s[0], out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        # img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
        img4list, labels4 = random_perspective(img4list,
                                               labels4,
                                               segments4,
                                               degrees=self.hyp['degrees'],
                                               translate=self.hyp['translate'],
                                               scale=self.hyp['scale'],
                                               shear=self.hyp['shear'],
                                               perspective=self.hyp['perspective'],
                                               border=self.mosaic_border)  # border to remove

        return img4list, labels4

    def paste_to_image_hard(self, imglist, labels):
        # 获取第一幅图像的高度和宽度
        img_h = imglist[0].shape[0]
        img_w = imglist[0].shape[1]

        itrial = 0  # 初始化尝试次数
        while itrial < self.hyp['max_trials']:  # 当尝试次数小于最大尝试次数时循环
            index = random.choice(self.indices)  # 从索引中随机选择一个
            labels_other = self.labels[index].copy()  # 获取对应标签并复制

            # 如果 labels_other 的大小不等于 1，则跳过本次循环
            if len(labels_other) != 1:
                continue

            # 加载另一幅图像及其标签
            _, imglist_other, _, (img_other_h, img_other_w) = self.load_imglist(index)

            # 将其他图像的标签从相对坐标转换为绝对坐标
            labels_other[:, 1:] = xywhn2xyxy(labels_other[:, 1:], img_other_w, img_other_h)

            for lb in labels_other:  # 遍历其他图像的标签
                if not itrial < self.hyp['max_trials']:  # 如果尝试次数超过最大限制，返回当前图像及标签
                    return imglist, labels

                itrial += 1  # 增加尝试次数

                # 获取每个标签的边界框坐标
                x1, y1, x2, y2 = lb[1], lb[2], lb[3], lb[4]
                cx = (x1 + x2) / 2  # 计算边界框中心的 x 坐标
                cy = (y1 + y2) / 2  # 计算边界框中心的 y 坐标
                w = x2 - x1  # 计算边界框的宽度
                h = y2 - y1  # 计算边界框的高度

                # 如果边界框的宽度或高度超过当前图像的尺寸，跳过
                if img_w < w or img_h < h:
                    continue

                # 随机生成新的中心坐标
                cx_new = random.uniform(0, 1) * img_w
                cy_new = random.uniform(0, 1) * img_h
                x1_new = cx_new - w / 2  # 计算新的边界框左上角 x 坐标
                y1_new = cy_new - h / 2  # 计算新的边界框左上角 y 坐标
                x2_new = cx_new + w / 2  # 计算新的边界框右下角 x 坐标
                y2_new = cy_new + h / 2  # 计算新的边界框右下角 y 坐标

                # 限制新的边界框坐标在图像范围内
                x1_new_clip = max(0, x1_new)
                y1_new_clip = max(0, y1_new)
                x2_new_clip = min(x2_new, img_w - 1)
                y2_new_clip = min(y2_new, img_h - 1)

                # 生成扩展比例
                expand_ratio = random.uniform(2.0, self.hyp['max_expand_ratio'])
                w_expand = expand_ratio * w  # 计算扩展后的宽度
                h_expand = expand_ratio * h  # 计算扩展后的高度

                # 计算扩展后的边界框坐标
                x1_expand = max(0, cx - w_expand / 2)
                y1_expand = max(0, cy - h_expand / 2)
                x2_expand = min(cx + w_expand / 2, img_other_w - 1)
                y2_expand = min(cy + h_expand / 2, img_other_h - 1)

                # 计算扩展框的新中心坐标
                cx_expand = (x1_expand + x2_expand) / 2
                cy_expand = (y1_expand + y2_expand) / 2
                w_expand = x2_expand - x1_expand
                h_expand = y2_expand - y1_expand

                # 计算中心坐标偏移量
                cx_expand_offset = cx_expand - cx
                cy_expand_offset = cy_expand - cy

                # 计算新的扩展边界框坐标
                cx_new_expand = cx_new + cx_expand_offset
                cy_new_expand = cy_new + cy_expand_offset
                x1_new_expand = cx_new_expand - w_expand / 2
                y1_new_expand = cy_new_expand - h_expand / 2
                x2_new_expand = cx_new_expand + w_expand / 2
                y2_new_expand = cy_new_expand + h_expand / 2

                # 限制扩展后的边界框坐标在图像范围内
                x1_new_expand_clip = max(0, x1_new_expand)
                y1_new_expand_clip = max(0, y1_new_expand)
                x2_new_expand_clip = min(img_w - 1, x2_new_expand)
                y2_new_expand_clip = min(img_h - 1, y2_new_expand)

                # 计算新边界框与现有标签的重叠度
                ious = bbox_iou(
                    torch.tensor([x1_new_expand_clip, y1_new_expand_clip, x2_new_expand_clip, y2_new_expand_clip]),
                    torch.from_numpy(labels[:, 1:]),
                    xywh=False
                ) if labels.size else torch.tensor([0.0])

                # 如果重叠度小于阈值，则添加新边界框
                if torch.max(ious).item() < float(self.hyp['overlapThresh']):
                    bbox_new = np.array([lb[0], x1_new_clip, y1_new_clip, x2_new_clip, y2_new_clip])
                    bbox_new = bbox_new[None, :]  # 将新边界框变为二维数组
                    labels = np.concatenate((labels, bbox_new), axis=0)  # 添加新边界框到标签中

                    # 将扩展后的边界框坐标转换为整数
                    w_expand_int = int(w_expand)
                    h_expand_int = int(h_expand)
                    x1_expand_int = int(x1_expand)
                    y1_expand_int = int(y1_expand)
                    x2_expand_int = x1_expand_int + w_expand_int
                    y2_expand_int = y1_expand_int + h_expand_int

                    x1_new_expand_int = int(x1_new_expand)
                    y1_new_expand_int = int(y1_new_expand)
                    x2_new_expand_int = int(x1_new_expand_int + w_expand_int)
                    y2_new_expand_int = int(y1_new_expand_int + h_expand_int)

                    # 限制新扩展边界框坐标在图像范围内
                    x1_new_expand_int_clip = max(0, x1_new_expand_int)
                    y1_new_expand_int_clip = max(0, y1_new_expand_int)
                    x2_new_expand_int_clip = min(img_w - 1, x2_new_expand_int)
                    y2_new_expand_int_clip = min(img_h - 1, y2_new_expand_int)

                    # 计算新边界框的宽度和高度
                    w_new = x2_new_expand_int_clip - x1_new_expand_int_clip
                    h_new = y2_new_expand_int_clip - y1_new_expand_int_clip

                    # 计算边界框的起始位置
                    x1_start = abs(x1_new_expand_int - x1_new_expand_int_clip) if x1_new_expand_int < 0 else 0
                    y1_start = abs(y1_new_expand_int - y1_new_expand_int_clip) if y1_new_expand_int < 0 else 0

                    # 将其他图像的内容粘贴到当前图像中
                    for img, img_other in zip(imglist, imglist_other):
                        img[y1_new_expand_int_clip:y2_new_expand_int_clip,
                        x1_new_expand_int_clip:x2_new_expand_int_clip] \
                            = img_other[(y1_expand_int + y1_start):(y1_expand_int + y1_start + h_new),
                              (x1_expand_int + x1_start):(x1_expand_int + x1_start + w_new)]

        return imglist, labels  # 返回处理后的图像列表和标签

    # def paste_to_image_hard(self, imglist, labels):
    #     img_h = imglist[0].shape[0]
    #     img_w = imglist[0].shape[1]
    #     itrial = 0
    #     while itrial < self.hyp['max_trials']:
    #         index = random.choice(self.indices)
    #         labels_other = self.labels[index].copy()
    #         # if not labels_other.size:
    #         #     continue
    #         if len(labels_other) != 1:
    #             continue
    #         _, imglist_other, _, (img_other_h, img_other_w) = self.load_imglist(index)
    #         labels_other[:, 1:] = xywhn2xyxy(labels_other[:, 1:], img_other_w, img_other_h)
    #         for lb in labels_other:
    #             if not itrial < self.hyp['max_trials']:
    #                 return imglist, labels
    #             itrial += 1
    #             x1, y1, x2, y2 = lb[1], lb[2], lb[3], lb[4]
    #             cx = (x1 + x2) / 2
    #             cy = (y1 + y2) / 2
    #             w = x2 - x1
    #             h = y2 - y1
    #             if img_w < w or img_h < h:
    #                 continue
    #             cx_new = random.uniform(0, 1) * img_w
    #             cy_new = random.uniform(0, 1) * img_h
    #             x1_new = cx_new - w / 2
    #             y1_new = cy_new - h / 2
    #             x2_new = cx_new + w / 2
    #             y2_new = cy_new + h / 2
    #             x1_new_clip = max(0, x1_new)
    #             y1_new_clip = max(0, y1_new)
    #             x2_new_clip = min(x2_new, img_w - 1)
    #             y2_new_clip = min(y2_new, img_h - 1)
    #
    #             expand_ratio = random.uniform(2.0, self.hyp['max_expand_ratio'])
    #             w_expand = expand_ratio * w
    #             h_expand = expand_ratio * h
    #             x1_expand = max(0, cx - w_expand / 2)
    #             y1_expand = max(0, cy - h_expand / 2)
    #             x2_expand = min(cx + w_expand / 2, img_other_w - 1)
    #             y2_expand = min(cy + h_expand / 2, img_other_h - 1)
    #             cx_expand = (x1_expand + x2_expand) / 2
    #             cy_expand = (y1_expand + y2_expand) / 2
    #             w_expand = x2_expand - x1_expand
    #             h_expand = y2_expand - y1_expand
    #             cx_expand_offset = cx_expand - cx
    #             cy_expand_offset = cy_expand - cy
    #
    #             cx_new_expand = cx_new + cx_expand_offset
    #             cy_new_expand = cy_new + cy_expand_offset
    #             x1_new_expand = cx_new_expand - w_expand / 2
    #             y1_new_expand = cy_new_expand - h_expand / 2
    #             x2_new_expand = cx_new_expand + w_expand / 2
    #             y2_new_expand = cy_new_expand + h_expand / 2
    #             x1_new_expand_clip = max(0, x1_new_expand)
    #             y1_new_expand_clip = max(0, y1_new_expand)
    #             x2_new_expand_clip = min(img_w - 1, x2_new_expand)
    #             y2_new_expand_clip = min(img_h - 1, y2_new_expand)
    #
    #             ious = bbox_iou(torch.tensor([x1_new_expand_clip, y1_new_expand_clip, x2_new_expand_clip, y2_new_expand_clip]), torch.from_numpy(labels[:, 1:]), xywh=False) if labels.size else torch.tensor([0.0])
    #             if torch.max(ious).item() < float(self.hyp['overlapThresh']):
    #                 bbox_new = np.array([lb[0], x1_new_clip, y1_new_clip, x2_new_clip, y2_new_clip])
    #                 bbox_new = bbox_new[None, :]
    #                 labels = np.concatenate((labels, bbox_new), axis=0)
    #
    #                 w_expand_int = int(w_expand)
    #                 h_expand_int = int(h_expand)
    #                 x1_expand_int = int(x1_expand)
    #                 y1_expand_int = int(y1_expand)
    #                 x2_expand_int = x1_expand_int + w_expand_int
    #                 y2_expand_int = y1_expand_int + h_expand_int
    #
    #                 x1_new_expand_int = int(x1_new_expand)
    #                 y1_new_expand_int = int(y1_new_expand)
    #                 x2_new_expand_int = x1_new_expand_int + w_expand_int
    #                 y2_new_expand_int = y1_new_expand_int + h_expand_int
    #
    #                 x1_new_expand_int_clip = max(0, x1_new_expand_int)
    #                 y1_new_expand_int_clip = max(0, y1_new_expand_int)
    #                 x2_new_expand_int_clip = min(img_w - 1, x2_new_expand_int)
    #                 y2_new_expand_int_clip = min(img_h - 1, y2_new_expand_int)
    #
    #                 w_new = x2_new_expand_int_clip - x1_new_expand_int_clip
    #                 h_new = y2_new_expand_int_clip - y1_new_expand_int_clip
    #                 x1_start = abs(x1_new_expand_int - x1_new_expand_int_clip) if x1_new_expand_int < 0 else 0
    #                 y1_start = abs(y1_new_expand_int - y1_new_expand_int_clip) if y1_new_expand_int < 0 else 0
    #
    #                 for img, img_other in zip(imglist, imglist_other):
    #                     img[y1_new_expand_int_clip:y2_new_expand_int_clip, x1_new_expand_int_clip:x2_new_expand_int_clip] \
    #                         = img_other[(y1_expand_int + y1_start):(y1_expand_int + y1_start + h_new), (x1_expand_int + x1_start):(x1_expand_int + x1_start + w_new)]
    #     return imglist, labels

    def load_mosaic9(self, index):
        # YOLOv5 9-mosaic loader. Loads 1 image + 8 random images into a 9-image mosaic
        labels9, segments9 = [], []
        s = self.img_size
        indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
        random.shuffle(indices)
        hp, wp = -1, -1  # height, width previous
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
            labels9.append(labels)
            segments9.extend(segments)

            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # mosaic center x, y
        img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

        # Concat/clip labels
        labels9 = np.concatenate(labels9, 0)
        labels9[:, [1, 3]] -= xc
        labels9[:, [2, 4]] -= yc
        c = np.array([xc, yc])  # centers
        segments9 = [x - c for x in segments9]

        for x in (labels9[:, 1:], *segments9):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img9, labels9 = replicate(img9, labels9)  # replicate

        # Augment
        img9, labels9, segments9 = copy_paste(img9, labels9, segments9, p=self.hyp['copy_paste'])
        img9, labels9 = random_perspective(img9,
                                           labels9,
                                           segments9,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove

        return img9, labels9

    @staticmethod
    def collate_fn(batch):
        im_orig, im, label, path, shapes = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return im_orig, torch.stack(im, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        im, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        im4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im1 = F.interpolate(im[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear',
                                    align_corners=False)[0].type(im[i].type())
                lb = label[i]
            else:
                im1 = torch.cat((torch.cat((im[i], im[i + 1]), 1), torch.cat((im[i + 2], im[i + 3]), 1)), 2)
                lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            im4.append(im1)
            label4.append(lb)

        for i, lb in enumerate(label4):
            lb[:, 0] = i  # add target image index for build_targets()

        return torch.stack(im4, 0), torch.cat(label4, 0), path4, shapes4


# Ancillary functions --------------------------------------------------------------------------------------------------
def flatten_recursive(path=DATASETS_DIR / 'coco128'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(f'{str(path)}_flat')
    if os.path.exists(new_path):
        shutil.rmtree(new_path)  # delete output folder
    os.makedirs(new_path)  # make new output folder
    for file in tqdm(glob.glob(f'{str(Path(path))}/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path=DATASETS_DIR / 'coco128'):  # from utils.dataloaders import *; extract_boxes()
    # Convert detection dataset into classification dataset, with one directory per class
    path = Path(path)  # images dir
    shutil.rmtree(path / 'classification') if (path / 'classification').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file) as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path=DATASETS_DIR / 'coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.dataloaders import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], 'a') as f:
                f.write(f'./{img.relative_to(path.parent).as_posix()}' + '\n')  # add image to txt file


def verify_image_label(args):
    # 验证单个图像-标签对
    im_file, lb_file, prefix = args  # 解包参数，包括图像文件路径、标签文件路径和前缀字符串
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # 初始化统计变量和消息列表
    try:
        # # 验证图像
        # im = Image.open(im_file)  # 使用PIL打开图像文件
        # im.verify()  # 使用PIL的verify方法检查图像文件是否损坏
        # shape = exif_size(im)  # 获取图像的尺寸
        # assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'  # 确保图像尺寸大于9像素
        # assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'  # 确保图像格式有效
        #
        # # 检查JPEG图像是否损坏，并尝试修复
        # if im.format.lower() in ('jpg', 'jpeg'):
        #     with open(im_file, 'rb') as f:
        #         f.seek(-2, 2)
        #         if f.read() != b'\xff\xd9':  # 检查JPEG结束标志
        #             ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
        #             msg = f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'


        width = int(im_file.split('___')[1])
        height = int(im_file.split('___')[2])
        shape = exif_size((width,height))

        # 验证标签
        if os.path.isfile(lb_file):
            nf = 1  # 标记找到标签
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                # 处理包含分割信息的标签
                if any(len(x) > 6 for x in lb):
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            # 验证标签格式和值
            if nl:
                assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # 检查重复标签
                    lb = lb[i]
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f'{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # 标记标签为空
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # 标记标签缺失
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


class HUBDatasetStats():
    """ Class for generating HUB dataset JSON and `-hub` dataset directory

    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        autodownload:   Attempt to download dataset if not found locally

    Usage
        from utils.dataloaders import HUBDatasetStats
        stats = HUBDatasetStats('coco128.yaml', autodownload=True)  # usage 1
        stats = HUBDatasetStats('path/to/coco128.zip')  # usage 2
        stats.get_json(save=False)
        stats.process_images()
    """

    def __init__(self, path='coco128.yaml', autodownload=False):
        # Initialize class
        zipped, data_dir, yaml_path = self._unzip(Path(path))
        try:
            with open(check_yaml(yaml_path), errors='ignore') as f:
                data = yaml.safe_load(f)  # data dict
                if zipped:
                    data['path'] = data_dir
        except Exception as e:
            raise Exception("error/HUB/dataset_stats/yaml_load") from e

        check_dataset(data, autodownload)  # download dataset if missing
        self.hub_dir = Path(data['path'] + '-hub')
        self.im_dir = self.hub_dir / 'images'
        self.im_dir.mkdir(parents=True, exist_ok=True)  # makes /images
        self.stats = {'nc': data['nc'], 'names': list(data['names'].values())}  # statistics dictionary
        self.data = data

    @staticmethod
    def _find_yaml(dir):
        # Return data.yaml file
        files = list(dir.glob('*.yaml')) or list(dir.rglob('*.yaml'))  # try root level first and then recursive
        assert files, f'No *.yaml file found in {dir}'
        if len(files) > 1:
            files = [f for f in files if f.stem == dir.stem]  # prefer *.yaml files that match dir name
            assert files, f'Multiple *.yaml files found in {dir}, only 1 *.yaml file allowed'
        assert len(files) == 1, f'Multiple *.yaml files found: {files}, only 1 *.yaml file allowed in {dir}'
        return files[0]

    def _unzip(self, path):
        # Unzip data.zip
        if not str(path).endswith('.zip'):  # path is data.yaml
            return False, None, path
        assert Path(path).is_file(), f'Error unzipping {path}, file not found'
        unzip_file(path, path=path.parent)
        dir = path.with_suffix('')  # dataset directory == zip name
        assert dir.is_dir(), f'Error unzipping {path}, {dir} not found. path/to/abc.zip MUST unzip to path/to/abc/'
        return True, str(dir), self._find_yaml(dir)  # zipped, data_dir, yaml_path

    def _hub_ops(self, f, max_dim=1920):
        # HUB ops for 1 image 'f': resize and save at reduced quality in /dataset-hub for web/app viewing
        f_new = self.im_dir / Path(f).name  # dataset-hub image filename
        try:  # use PIL
            im = Image.open(f)
            r = max_dim / max(im.height, im.width)  # ratio
            if r < 1.0:  # image too large
                im = im.resize((int(im.width * r), int(im.height * r)))
            im.save(f_new, 'JPEG', quality=50, optimize=True)  # save
        except Exception as e:  # use OpenCV
            LOGGER.info(f'WARNING ⚠️ HUB ops PIL failure {f}: {e}')
            im = cv2.imread(f)
            im_height, im_width = im.shape[:2]
            r = max_dim / max(im_height, im_width)  # ratio
            if r < 1.0:  # image too large
                im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(f_new), im)

    def get_json(self, save=False, verbose=False):
        # Return dataset JSON for Ultralytics HUB
        def _round(labels):
            # Update labels to integer class and 6 decimal place floats
            return [[int(c), *(round(x, 4) for x in points)] for c, *points in labels]

        for split in 'train', 'val', 'test':
            if self.data.get(split) is None:
                self.stats[split] = None  # i.e. no test set
                continue
            dataset = LoadImagesAndLabels(self.data[split])  # load dataset
            x = np.array([
                np.bincount(label[:, 0].astype(int), minlength=self.data['nc'])
                for label in tqdm(dataset.labels, total=dataset.n, desc='Statistics')])  # shape(128x80)
            self.stats[split] = {
                'instance_stats': {
                    'total': int(x.sum()),
                    'per_class': x.sum(0).tolist()},
                'image_stats': {
                    'total': dataset.n,
                    'unlabelled': int(np.all(x == 0, 1).sum()),
                    'per_class': (x > 0).sum(0).tolist()},
                'labels': [{
                    str(Path(k).name): _round(v.tolist())} for k, v in zip(dataset.im_files, dataset.labels)]}

        # Save, print and return
        if save:
            stats_path = self.hub_dir / 'stats.json'
            print(f'Saving {stats_path.resolve()}...')
            with open(stats_path, 'w') as f:
                json.dump(self.stats, f)  # save stats.json
        if verbose:
            print(json.dumps(self.stats, indent=2, sort_keys=False))
        return self.stats

    def process_images(self):
        # Compress images for Ultralytics HUB
        for split in 'train', 'val', 'test':
            if self.data.get(split) is None:
                continue
            dataset = LoadImagesAndLabels(self.data[split])  # load dataset
            desc = f'{split} images'
            for _ in tqdm(ThreadPool(NUM_THREADS).imap(self._hub_ops, dataset.im_files), total=dataset.n, desc=desc):
                pass
        print(f'Done. All images saved to {self.im_dir}')
        return self.im_dir


# Classification dataloaders -------------------------------------------------------------------------------------------
class ClassificationDataset(torchvision.datasets.ImageFolder):
    """
    YOLOv5 Classification Dataset.
    Arguments
        root:  Dataset path
        transform:  torchvision transforms, used by default
        album_transform: Albumentations transforms, used if installed
    """

    def __init__(self, root, augment, imgsz, cache=False):
        super().__init__(root=root)
        self.torch_transforms = classify_transforms(imgsz)
        self.album_transforms = classify_albumentations(augment, imgsz) if augment else None
        self.cache_ram = cache is True or cache == 'ram'
        self.cache_disk = cache == 'disk'
        self.samples = [list(x) + [Path(x[0]).with_suffix('.npy'), None] for x in self.samples]  # file, index, npy, im

    def __getitem__(self, i):
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram and im is None:
            im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f))
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        if self.album_transforms:
            sample = self.album_transforms(image=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))["image"]
        else:
            sample = self.torch_transforms(im)
        return sample, j


def create_classification_dataloader(path,
                                     imgsz=224,
                                     batch_size=16,
                                     augment=True,
                                     cache=False,
                                     rank=-1,
                                     workers=8,
                                     shuffle=True):
    # Returns Dataloader object to be used with YOLOv5 Classifier
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = ClassificationDataset(root=path, imgsz=imgsz, augment=augment, cache=cache)
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=shuffle and sampler is None,
                              num_workers=nw,
                              sampler=sampler,
                              pin_memory=PIN_MEMORY,
                              worker_init_fn=seed_worker,
                              generator=generator)  # or DataLoader(persistent_workers=True)


def moving_average(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed


def smooth_trajectory(trajectory, radius=50):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:, i] = moving_average(trajectory[:, i], radius)
    return smoothed_trajectory


def stabilize_video(frame_list):
    frame_list = frame_list[::-1]

    transforms = np.zeros((8 - 1, 3), np.float32)
    for i in range(1, 8):
        prev_gray = frame_list[i - 1]
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        curr_gray = frame_list[i]
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        if prev_pts.shape[0] < 4:
            m = np.eye(2, 3, dtype=np.float32)
        else:
            m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)

        if m is None:
            m = np.eye(2, 3, dtype=np.float32)

        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])

        transforms[i - 1] = [dx, dy, da]
    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth_trajectory(trajectory)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference
    frame_stabilized_list = [frame_list[0]]
    for i in range(1, 8):
        curr_gray = frame_list[i]
        dx = transforms_smooth[i - 1, 0]
        dy = transforms_smooth[i - 1, 1]
        da = transforms_smooth[i - 1, 2]

        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        frame_stabilized = cv2.warpAffine(curr_gray, m, (curr_gray.shape[1], curr_gray.shape[0]))
        frame_stabilized_list.append(frame_stabilized)
    frame_stabilized_list = frame_stabilized_list[::-1]
    return frame_stabilized_list
