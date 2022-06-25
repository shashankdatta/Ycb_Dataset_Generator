#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import yaml        

def main():
    '''
      # python train.py --img 416 --batch 16 --epochs 500 --data '../data.yaml' --cfg ./models/custom_yolov5x.yaml --weights '' --name yolov5x_results --cache
      # python -m torch.distributed.run --nproc_per_node 2 train.py --batch 64 --data coco.yaml --weights yolov5s.pt --device 0,1
      # Final Command: python -m torch.distributed.run --nproc_per_node 2 train.py --img 416 --batch 16 --epochs 500 --data '../data.yaml' --cfg ./models/custom_yolov5x.yaml --weights '' --name yolov5x_results --device 0,1 --cache
    '''

    # model =  torch.hub.load('ultralytics/yolov5', 'yolov5x6')
    model =  torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt', force_reload=True)

    model.conf = 0.80  # confidence threshold (0-1)
    model.iou = 0.70  # NMS IoU threshold (0-1)
    # model.classes = [0]  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs

    # ReturnedArray = results.display(pprint=True,labels=True,crop=True)
    # for jsonObject in ReturnedArray:
    #   strSplit = str.split(jsonObject['label'])
    #   confidence = strSplit[1][2:]
    #   classification = strSplit[0]
    #   print('Detected Element Class: {} | Confidence Threshold: {}%'.format(str.capitalize(classification), confidence))
    #   print('Detected Element: {} | Confidence Threshold: {}'.format(jsonObject['label'], jsonObject['conf']))

    cap = cv2.VideoCapture(-1)
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Make detections 
        results = model(frame)
        
        cv2.imshow('Live Detection', np.squeeze(results.render()))
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()   

if __name__ == "__main__":
    main()