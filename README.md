# 通过YOLOv5实现FPS游戏的微自瞄

# 原理：
        1、实时屏幕截图，将图片喂给yolo，yolo返回bbox
        2、将bbox转化为屏幕坐标，方便鼠标控制
        
# 注意事项：
        1、鼠标驱动用的是罗技驱动，版本是2021.3.9205
