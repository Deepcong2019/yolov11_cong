#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:cong
@time: 2024/11/27 17:08:03
"""
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('./runs/detect/train4/weights/best.pt')
    model.predict(source=r'D:\ultralytics\VOCdevkit\VOC2007\images\test',
                  imgsz=640,
                  device='0',
                  )
