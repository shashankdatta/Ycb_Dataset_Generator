import glob, os, time, torch
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

def main():
    ycb_download_location = f'{os.getcwd()}/models/ycb'
    for object_name in os.listdir(ycb_download_location):
        object_class = 0
        i = 0
        for mask_name, img_name in zip(sorted(os.listdir(f'{ycb_download_location}/{object_name}/masks'), key=parse), 
            glob.glob(f'{ycb_download_location}/{object_name}/*.jpg')):

            print(mask_name, img_name)

            # img = cv.imread(f'{os.getcwd()}/masks/{imgName}', -1)
            # print(imgName)
            # imgs = cv.cvtColor(img.copy(), cv.COLOR_BGR2RGB)
            # imgRGB = cv.cvtColor(imgs, cv.COLOR_RGB2GRAY)
            # thresh = cv.threshold(imgRGB, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

            # ROI_number = 0
            # contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # print(len(contours))
            # contours = contours[0] if len(contours) == 2 else contours[1]

            # x,y,w,h = cv.boundingRect(contours[len(contours) - 1])

            ## Drawing Rectangle Bounding Box On Images
            # cv.rectangle(imgs, (x - 30, y - 30), (x + w + 65, y + h + 65), (255,0,0), 5)
            # print([x - 30, y - 30, x + w + 30, y + h + 30])
            
            # print(img.shape)
            # w_img = img.shape[1]
            # h_img = img.shape[0]
            
            # normalizedBBoxCoordinates = normalize_bbox(0, x - 35, y - 35, w + 80, h + 90, w_img, h_img)
            # normalizedBBoxCoordinates = normalize_bbox(object_class, x, y, w, h, w_img, h_img) 

            # print (" ".join(map(str, normalizedBBoxCoordinates)))
            # print(normalize_bbox_data_to_im(bbox, rows, cols))
            
            ## for c in contours:
            #     x,y,w,h = cv.boundingRect(c)
            #     print([x,y,w,h])
            #     cv.rectangle(imgs, (x - 25, y - 25), (x + w + 25, y + h + 25), (255,0,0), 5)

            ## For Matplotlib Plots:
            # fig = plt.figure(figsize = (7, 7))
            # ax = fig.add_subplot(111)
            # ax.imshow(imgs)
            # plt.show(block=False)
            
            ## For cv Image View:
            # i += 1
            # img = cv.resize(imgs, (960, 540))
            # cv.imshow(f"Image {i}", img)
            
            # if (i == 3):
            #     break

            ## For Matplotlib Plots:
            # plt.pause(5)
            # time.sleep(5)
            # plt.close('all')

            ## For cv Image View:
            # cv.waitKey(1000) 
            # cv.destroyAllWindows()
            object_class += 1

def parse(fname):
    prefix, n1, n2 = fname.split('_')
    return (prefix, int(n1))

def normalize_bbox(label_index, xmin, ymin,
    w, h, w_img, h_img):
    xcenter = (xmin + w/2) / w_img
    ycenter = (ymin + h/2) / h_img
    w = w / w_img
    h = h / h_img
    return [label_index, xcenter, ycenter, w, h]

if __name__ == "__main__":
    main()