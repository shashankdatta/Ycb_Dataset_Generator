#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from multiprocessing.connection import wait
from random import shuffle
import shutil
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import yaml
import glob
import os
import time
import PIL as Image


def make_1080p(cap):
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p(cap):
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p(cap):
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

def main():
    '''
      # python train.py --img 416 --batch 16 --epochs 500 --data '../data.yaml' --cfg ./models/custom_yolov5x.yaml --weights '' --name yolov5x_results --cache
      # python -m torch.distributed.run --nproc_per_node 2 train.py --batch 64 --data coco.yaml --weights yolov5s.pt --device 0,1
      # Final Command: python -m torch.distributed.run --nproc_per_node 2 train.py --img 416 --batch 16 --epochs 500 --data '../data.yaml' --cfg ./models/custom_yolov5x.yaml --weights '' --name yolov5x_results --device 0,1 --cache
    '''

    # model =  torch.hub.load('ultralytics/yolov5', 'yolov5x6')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='../yolov5/runs/train/yolov5s6_results/weights/best.pt', force_reload=True)

    # model.conf = 0.70  # confidence threshold (0-1)
    # model.iou = 0.65  # NMS IoU threshold (0-1)
    # model.classes = [0]  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs

    # for img in glob.glob1(f'./testImgs', '*.jpg'):
    #     results = model(f'./testImgs/{img}')
    #     results.show()
    #     time.sleep(3)

    # ReturnedArray = results.display(pprint=True,labels=True,crop=True)
    # for jsonObject in ReturnedArray:
    #   strSplit = str.split(jsonObject['label'])
    #   confidence = strSplit[1][2:]
    #   classification = strSplit[0]
    #   print('Detected Element Class: {} | Confidence Threshold: {}%'.format(str.capitalize(classification), confidence))
    #   print('Detected Element: {} | Confidence Threshold: {}'.format(jsonObject['label'], jsonObject['conf']))

    cap = cv2.VideoCapture(-1)
    # change_res(cap, 512, 640)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()

        # Make detections 
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(imgRGB)

        imgBGR = cv2.cvtColor(np.squeeze(results.render()), cv2.COLOR_RGB2BGR)
        
        # cv2.imwrite('./03.jpg', frame)
        # exit()

        cv2.imshow('Live Detection', imgBGR)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def takePictures(amt, cntDwn):
    cap = cv2.VideoCapture(-1)
    i = 0

    shutil.rmtree('testImgs')
    os.mkdir('testImgs')

    while i < amt:
        ret, frame = cap.read()

        cv2.imshow('Live Detection', np.squeeze(frame))

        for j in range(cntDwn,0,-1):
            print(f"{j}", end="\r", flush=True)    
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
    
        cv2.imwrite(f'./testImgs/{i}.jpg', frame)
        i+=1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    # takePictures(25, 5)