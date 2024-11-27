#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:cong
@time: 2024/11/27 17:16:55
"""
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/train4/weights/best.pt')
    model.val(data='card.yaml',
              imgsz=640,
              batch=16,
              split='val',
              workers=0,
              device='0',
              )
