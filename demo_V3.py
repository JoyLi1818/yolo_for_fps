import ctypes
import argparse
import os
import platform
import sys
import threading
import time
from pathlib import Path
import cv2
import numpy as np
import pynput
import win32gui, win32ui, win32con, win32api
import torch

from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import (LOGGER, Profile, check_img_size, check_requirements, cv2,
                           non_max_suppression, print_args, scale_coords, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode
from aim_lock import lock
from screen_inf import grab_screen_win32, get_parameters
import ghub_mouse as ghub

"""
设置系统环境
这一部分的主要作用有两个:

    将当前项目添加到系统路径上,以使得项目中的模块可以调用.
    将当前项目的相对路径保存在ROOT中,便于寻找项目中的文件.
"""
FILE = Path(__file__).resolve()  # __file__指的是当前文件(即detect.py),FILE最终保存着当前文件的绝对路径,比如D://yolov5/detect.py
ROOT = FILE.parents[0]  # ROOT保存着当前项目的父目录,比如 D://yolov5
if str(ROOT) not in sys.path:  # sys.path即当前python环境可以运行的路径,假如当前项目不在该路径中,就无法运行其中的模块,所以就需要加载路径
    sys.path.append(str(ROOT))  # 把ROOT添加到运行路径上
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # ROOT设置为相对路径


@smart_inference_mode()  # 轻量预测模式
def run(
        window_size=(0, 0, 1920, 1080),
        region=(1, 1),
        weights=ROOT / 'yolov5s.pt',  # 事先训练完成的权重文件，比如yolov5s.pt,假如使用官方训练好的文件（比如yolov5s）,则会自动下载
        data=ROOT / 'data/coco128.yaml',  # 数据集文件
        imgsz=(640, 640),  # 预测时的放缩后图片大小(因为YOLO算法需要预先放缩图片), 两个值分别是height, width
        conf_thres=0.25,  # 置信度阈值, 高于此值的bounding_box才会被保留
        iou_thres=0.45,  # IOU阈值,高于此值的bounding_box才会被保留
        max_det=1000,  # 一张图片上检测的最大目标数量
        device='',  # 所使用的GPU编号，如果使用CPU就写cpu
        view_img=False,  # 是否在推理时预览图片
        classes=None,  # 过滤指定类的预测结果
        agnostic_nms=False,  # 如为True,则为class-agnostic. 否则为class-specific
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        line_thickness=3,  # 绘制Bounding_box的线宽度
        hide_labels=False,  # True: 隐藏标签
        hide_conf=False,  # True: 隐藏置信度
        half=False,  # use FP16 half-precision inference 是否使用半精度推理（节约显存）
        dnn=False,  # use OpenCV DNN for ONNX inference
        print_results=False,
):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    image_count = 0  # 当前检测图片序号
    mouse = pynput.mouse.Controller()  # 创建鼠标控制对象

    top_x, top_y, x, y = get_parameters()
    len_x, len_y = int(x * region[0]), int(y * region[1])
    top_x, top_y = int(top_x + x // 2 * (1. - region[0])), int(top_y + y // 2 * (1. - region[1]))

    # mouse_thread = threading.Thread(target=recoil)
    # mouse_thread.start()
    while True:
        im0s = grab_screen_win32(region=(top_x, top_y, top_x + len_x, top_y + len_y))
        im0s = cv2.resize(im0s, (len_x, len_y))

        # 填充图片
        im = letterbox(im0s, 640, stride=32, auto=pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        bs = 1  # batch_size

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # 使用空白图片（零矩阵）预先用GPU跑一遍预测流程，可以加速预测
        seen, dt = 0, (Profile(), Profile(), Profile())

        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            # 如果图片是3维(RGB) 就在前面添加一个维度,即batch_size
            # 因为输入网络的图片需要是4维的 [batch_size, channel, w, h]
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=augment, visualize=visualize)  # 模型预测出来的所有检测框，torch.size=[1,18900,85]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions   把所有的检测框画到原图中
        aims = []  # 存放多个目标，单个目标是个数组，里面存放[cls, x_c, y_c, w, h]
        for i, det in enumerate(pred):  # 每次迭代处理一张图片    i：每个batch的信息，det:表示5个检测框的信息
            seen += 1
            im0 = im0s.copy()
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 得到原图的宽和高

            if view_img:
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))  # 得到一个绘图的类，类中预先存储了原图、线条宽度、类名

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # 将标注的bounding_box大小调整为和原图一致（因为训练时原图经过了放缩）

                # Print results
                if print_results:
                    image_count += 1
                    s = f'image {image_count}: '
                    s += '%gx%g ' % im.shape[2:]  # 显示推理前裁剪后的图像尺寸
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # 打印出所有的预测结果  比如1 person（检测出一个人）

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # 将坐标转变成x y w h 的形式，并归一化
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # line的形式是： ”类别 x y w h“，假如save_conf为true，则line的形式是：”类别 x y w h 置信度“
                    line = (cls, *xywh, conf)  # label format
                    line = ('%g ' * len(line)).rstrip() % line + '\n'
                    aim = line.split(' ')  # 对空格进行分割，组成一个数组
                    # print(aim)
                    aims.append(aim)  # aim -- 存放目标信息[cls, x_c, y_c, w, h]

                    if view_img:  # 给图片添加推理后的bounding_box边框
                        c = int(cls)  # 类别标号
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')  # 类别名
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # 绘制边框

            lock_mode = True if win32api.GetKeyState(0x04) else False
            if len(aims):
                if lock_mode:
                    lock(aims, mouse, top_x, top_y, len_x, len_y)

            # Stream results
            if view_img:  # 如果view_img为true,则显示该图片
                im0 = annotator.result()  # im0是绘制好的图片，返回画好的图片
                cv2.namedWindow('cf-detect', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('cf-detect', window_size[2] // 3, window_size[3] // 3)
                cv2.imshow('cf-detect', im0)
                # 设置检测窗口置顶
                hwnd = win32gui.FindWindow(None, 'cf-detect')
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                      win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
                # -----------------------模型预测-------------------------

                # waitKey()函数的功能是不断刷新图像，单位是ms
                # 如果waitKey(0)则会一直显示同一帧数
                if cv2.waitKey(1) & 0xff == ord('q'):
                    cv2.destroyAllWindows()  # 如果之前没有释放掉内存的操作的话destroyallWIndows会释放掉被那个变量占用的内存
                    break

        if print_results:
            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")  # 打印耗时

            # Print results
            t = tuple(x.t / seen * 1E3 for x in dt)  # 平均每张图片所耗费时间
            LOGGER.info(
                f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}\n\n\n' % t)


# 参数设置
def parse_opt():
    parser = argparse.ArgumentParser()
    # action='store_true'  ---- 如果运行代码时加了 --image ，那么 image为True,没加为False
    parser.add_argument('--window_size', default=(0, 0, 1920, 1080), action='store_true', help='window_size')
    parser.add_argument('--region', type=tuple, default=(0.3, 0.3),
                        help='检测范围；分别为横向和竖向，(1.0, 1.0)表示全屏检测，越低检测范围越小(始终保持屏幕中心为中心)')
    # 权重文件路径
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / './weights/yolov5s.pt', help='model path(s)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    # 预测图片的尺寸
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.30, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.50, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    # 是否推理时预览图片，默认False
    parser.add_argument('--view-img', default=False, help='show results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', default=True, help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--print_results', action='store_true', help='print_results')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def recoil():
    while True:
        if win32api.GetKeyState(0x01) == 0 or win32api.GetKeyState(0x01) == 1:  # 鼠标抬起
            pass
        else:
            time.sleep(0.1)
            ghub.mouse_xy(0, 10)


# 入口函数
def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


# 入口
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
