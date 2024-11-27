#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:cong
@time: 2024/11/27 16:10:54
"""
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO(r'yolo11s.yaml')
    model = YOLO("yolo11s.pt")
    model.train(data=r'card.yaml',
                imgsz=640,
                epochs=20,
                single_cls=False,
                batch=16,
                workers=0,
                device='0',
                optimizer='SGD',
                )