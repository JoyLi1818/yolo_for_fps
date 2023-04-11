import threading
import win32api
import ghub_mouse as ghub
from math import *

lock_smooth = 5  # lock平滑系数；越大越平滑，最低1.0
lock_sen = 1  # lock幅度系数；若在桌面试用请调成1，在游戏中(csgo)则为灵敏度
lock_choice = [0, 1, 2, 3]  # 目标选择；可自行决定锁定的目标，从自己的标签中选
lock_tag = [1, 0, 3, 2]  # 对应标签；缺一不可，自己按以下顺序对应标签，ct_head ct_body t_head t_body


def lock_thread(aims, mouse, top_x, top_y, len_x, len_y):
    mouse_pos_x, mouse_pos_y = mouse.position
    aims_copy = aims
    k = 5 * (1 / lock_smooth)
    if len(aims_copy):
        dist_list = []
        for det in aims_copy:
            _, x_c, y_c, _, _, _ = det
            dist = (len_x * float(x_c) + top_x - mouse_pos_x) ** 2 + (len_y * float(y_c) + top_y - mouse_pos_y) ** 2
            dist_list.append(dist)
        det = aims_copy[dist_list.index(min(dist_list))]  # 取最近的目标
        tag, x_center, y_center, width, height, conf = det
        x_center, width = len_x * float(x_center) + top_x, len_x * float(width)
        y_center, height = len_y * float(y_center) + top_y, len_y * float(height)
        # print(x_center,y_center)  # 打印目标在屏幕上的坐标
        # rel_x = int(k / lock_sen * atan((mouse_pos_x - x_center) / 640) * 640)    # atan()返回x的反正切弧度值
        # rel_x = int(atan((mouse_pos_x - x_center) / 640) * 640)
        rel_x = int(x_center - mouse_pos_x)
        # if int(tag) == 0:
        #     # rel_y = int(k / lock_sen * atan((mouse_pos_y - y_center) / 640) * 640)
        #     rel_y = int(atan((mouse_pos_y - y_center) / 640) * 640)
        #     ghub.mouse_xy(-rel_x, -rel_y)
        if int(tag) == 0:
            # rel_y = int(k / lock_sen * atan((mouse_pos_y - y_center) / 640) * 640)
            # rel_y = int(atan((mouse_pos_y - y_center) / 640) * 640)
            rel_y = int(y_center - mouse_pos_y)
            # ghub.mouse_xy(-rel_x, -rel_y)
            ghub.mouse_xy(rel_x, rel_y)


def lock(aims, mouse, top_x, top_y, len_x, len_y):
    # 创建锁人线程
    lock = threading.Thread(target=lock_thread, args=(aims, mouse, top_x, top_y, len_x, len_y))
    lock.start()
