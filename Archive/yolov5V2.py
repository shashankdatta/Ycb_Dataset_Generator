import numpy as np
import PIL as Image
import shutil
import torch
import cv2
import glob
import os
import argparse

def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("cmd", help="Choose to execute 'local' or 'live' image detection or 'picture' for clicking test pictures using src", type=str)
    parser.add_argument("src", help="Image capture device id for opencv", type=int)    
    parser.add_argument("model_weight", help="Path to your custom trained model's weight", type=str)
    parser.add_argument("--mode", dest='mode', help="Load default yolov5s6 model: '0' or custom model: '1'", default=0, type=int)    
    parser.add_argument("--local-path", dest='local_path', help="Path to your to local directory with source images in JPG or PNG format", default='./testImgs', type=str)
    parser.add_argument("--count-down", dest='count_down', help="Number of seconds to pause between each picture", default=5, type=int)
    parser.add_argument("--pic-amount", dest='amount', help="Number of pictures to click", default=25, type=int)
    
    args = parser.parse_args()
    # print(args)

    if(args.mode):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.model_weight, force_reload=True) # Loading Custom Trained Model
    else:
        model =  torch.hub.load('ultralytics/yolov5', 'yolov5s6') # Loading Given Model

    # Additional Configuration Values For The Model
    # model.conf = 0.70  # confidence threshold (0-1)
    # model.iou = 0.65  # NMS IoU threshold (0-1)
    # model.classes = [0]  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs

    if (args.cmd == 'live'):
        liveDetection(model, args.src)
    elif (args.cmd == 'local'):
        testModelLocal(model, args.local_path)
    elif (args.cmd == 'picture'):
        takePictures(args.amount, args.count_down, args.src)

    
def liveDetection(model, src):
    '''Live camera detection'''
    
    cap = cv2.VideoCapture(src)
    
    # Change Resoltuion (EXPERIMENTAL)
    # make_480p(cap)
    
    while cap.isOpened():
        ret, frame = cap.read()
        # print('Resolution: ' + str(frame.shape[1]) + ' x ' + str(frame.shape[0]))
        
        # Make detections 
        # Open CV reads input as BGR, but yolo model needs RGB
        
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB

        results = model(imgRGB) # Dectections made on RGB

        imgBGR = cv2.cvtColor(np.squeeze(results.render()), cv2.COLOR_RGB2BGR) # Convert RGB to BGR for corrected color visual output

        cv2.imshow('Live Detection', imgBGR)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def testModelLocal(model, pathToTestImages):
    '''Run Model On Local Test Images'''

    types = ('*.jpg', '*.png') # the tuple of file types
    files_grabbed = []

    for file_type in types:
        files_grabbed.extend(glob.glob1(pathToTestImages, file_type))

    for img in files_grabbed:
        results = model(f'{pathToTestImages}/{img}')
        
        # results.show() # Yolo Way To Display Images
        
        resultingImg = cv2.cvtColor(np.squeeze(results.render()), cv2.COLOR_RGB2BGR)
        
        # Custom display window setup
        cv2.namedWindow('Offline Image Detection', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('Offline Image Detection', resultingImg)

        # Image resized to fit the image display window
        cv2.resizeWindow('Offline Image Detection', 640, 640) 

        if cv2.waitKey(3000) & 0xFF == ord('q'): # Window stays open for 5 seconds
            break
    cv2.destroyAllWindows()

def takePictures(amt, cntDwn, src):
    '''Take pictures for local testing'''
    
    cap = cv2.VideoCapture(src)
    i = 0

    shutil.rmtree('./testImgs', ignore_errors=True)
    os.mkdir('./testImgs')

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