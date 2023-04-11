# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
代码来源：  https://blog.csdn.net/weixin_69398563/article/details/126378699
参考：     https://blog.csdn.net/weixin_46183779/article/details/125792291

Run inference on images, videos, directories, streams, etc.
Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()  # 绝对路径
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:  # 模块查询路径的列表
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative 绝对路径转换为相对路径

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # 权重文件地址 默认 weights/best.pt
        source=ROOT / 'data/images',  # 测试数据文件(图片或视频)的保存路径 默认data/images
        data=ROOT / 'data/coco128.yaml',  # 数据存放在yaml文件中，包含了训练、验证，预测的路径
        imgsz=(640, 640),  # inference size (height, width) 输入图片的大小 默认640(pixels)
        conf_thres=0.25,  # object置信度阈值 默认0.25  用在nms中
        iou_thres=0.45,  # 做nms的iou阈值 默认0.45   用在nms中
        max_det=1000,  # 每张图片最多的目标数量  用在nms中
        device='',  # 设置代码执行的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # 是否展示预测之后的图片或视频 默认False
        save_txt=False,  # 是否将预测的框坐标以txt文件格式保存 默认False
        save_conf=False,  # 是否保存预测每个目标的置信度到预测tx文件中 默认False
        save_crop=False,  # 是否需要将预测到的目标从原图中扣出来 剪切好 并保存 会在runs/detect/expn下生成crops文件，将剪切的图片保存在里面  默认False
        nosave=False,  # 是否不要保存预测后的图片  默认False 就是默认要保存预测后的图片
        classes=None,  # 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留
        agnostic_nms=False,  # 进行nms是否也除去不同类别之间的框 默认False
        augment=False,  # 预测是否也要采用数据增强 TTA 默认False
        visualize=False,  # visualize features
        update=False,  # 预测是否也要采用数据增强 TTA 默认False
        project=ROOT / 'runs/detect',  # 当前测试结果放在哪个主文件夹下 默认runs/detect
        name='exp',  # 当前测试结果放在run/detect下的文件名  默认是exp  =>  run/detect/exp
        exist_ok=False,  # 是否存在当前文件 默认False 一般是 no exist-ok 连用  所以一般都要重新创建文件夹
        line_thickness=3,  # bounding box thickness (pixels)   画框的框框的线宽  默认是 3
        hide_labels=False,  # 画出的框框是否需要隐藏label信息 默认False
        hide_conf=False,  # 画出的框框是否需要隐藏conf信息 默认False
        half=False,  # 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False
        dnn=False,  # use OpenCV DNN for ONNX inference  不使用
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # 是否保存预测后的图片 默认nosave=False 所以只要传入的文件地址不是以.txt结尾 就都是要保存预测后的图片的
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)  # 判断suffix[1:]表示后缀是以jpg格式
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))  # 判断地址是不是网络流地址
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    # 是否是使用webcam 网页数据 一般是Fasle  因为我们一般是使用图片流LoadImages(可以处理图片/视频流文件)
    if is_url and is_file:  # 如果是网络流地址，就会根据该地址去下载
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # 检查当前Path(project) / name是否存在 如果存在就新建新的save_dir 默认exist_ok=False 需要重建
    # 将原先传入的名字扩展成新的save_dir 如runs/detect/exp存在 就扩展成 runs/detect/exp1
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # 如果需要save txt就新建save_dir / 'labels' 否则就新建save_dir
    # 默认save_txt=False 所以这里一般都是新建一个 save_dir(runs/detect/expn)

    # Load model
    device = select_device(device)  # 获取当前主机可用的设备
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)  # 选择后端框架
    stride, names, pt = model.stride, model.names, model.pt  # 读取值
    imgsz = check_img_size(imgsz, s=stride)  # check image size 这个尺寸得是32的倍数

    # Dataloader
    if webcam:  # 没有执行，所以去执行加载图片
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size  每次输入一张图片
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference  推理部分
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup  传入一张图片，让GPU先热身一下
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]  # dt用来存储时间，seen是计数的功能
    for path, im, im0s, vid_cap, s in dataset:
        # 去遍历图片，此时在dataloader中进行的是209到216部分，进行计数，
        # 这里的path是指路径，im是指resize后的图片，im0是指原始图片，vid_cap=None，s是代表打印的信息
        # 以下部分是做预处理
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)  # torch.size=[3,640,480]
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0   所有像素点除以255，是归一化的操作
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim 缺少batch这个尺寸，所以将它扩充一下，变成[1，3,640,480]
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference 做预测
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)  # 模型预测出来的所有检测框，torch.size=[1,18900,85]
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS非极大值抑制
        # Apply NMS  进行NMS
        # conf_thres: 置信度阈值
        # iou_thres: iou阈值
        # classes: 是否只保留特定的类别 默认为None
        # agnostic_nms: 进行nms是否也去除不同类别之间的框
        # max_det: 每张图片的最大目标个数 默认1000
        # pred: [num_obj, 6] = [5, 6]   这里的预测信息pred还是相对于 img_size(640) 的
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions 把所有的检测框画到原图中
        for i, det in enumerate(pred):  # per image  i：每个batch的信息，det:表示5个检测框的信息
            seen += 1  # seen是一个计数的功能
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg 存储路径+图片名
            txt_path = str(save_dir / 'labels' / p.stem) + (
                '' if dataset.mode == 'image' else f'_{frame}')  # im.txt默认不存
            s += '%gx%g ' % im.shape[2:]  # 输出信息  图片shape (w, h)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh 获得原图的宽和高的大小
            imc = im0.copy() if save_crop else im0  # for save_crop  是否要将检测的物体进行裁剪
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))  # 定义绘图工具
            if len(det):  # 判断有没有框
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()  # scale_coords坐标映射功能

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # 默认是不执行的
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image 执行这里
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:  # 默认False，不执行
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()  # 返回画好的图片
            if view_img:
                if p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results 打印结果
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)