import time, os
from matplotlib import pyplot as plt
import cv2 as cv

ycb_download_location = f'{os.getcwd()}/models/ycb'
masks_directory_location = f'{ycb_download_location}/001_chips_can/masks'

def maskParseFilter(fname):
    prefix, n1, n2 = fname.split('_')
    return (prefix, int(n1))

# This function allows us to create a descending sorted list of contour areas.
def get_contour_areas(contours):

    ## Method 1:
    # all_areas= []

    # for cnt in contours:
    #     area = cv.contourArea(cnt)
    #     all_areas.append(area)

    ## Method 2:
    contours = list(map(cv.contourArea, contours))

    # print(all_areas)
    # print(contours)

    return contours

for mask_name in sorted(os.listdir(masks_directory_location), key=maskParseFilter):
    img = cv.imread(f'{masks_directory_location}/{mask_name}', -1)
    # img = cv.imread('./6399621.jpg', -1)
    imgs = cv.cvtColor(img.copy(), cv.COLOR_BGR2RGB)
    imgRGB = cv.cvtColor(imgs, cv.COLOR_RGB2GRAY)
    thresh = cv.threshold(imgRGB, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    ROI_number = 0
    contours, her = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # cnt_area = get_contour_areas(contours)   

    sorted_contours = sorted(contours, key=cv.contourArea, reverse= True)
    
    x,y,w,h = cv.boundingRect(sorted_contours[0])
    
    cv.rectangle(imgs, (x, y), (x + w, y + h), (255,0,0), 15)

    # To draw all the contours in an image:
    # cv.drawContours(imgs, contours, -1, (0,255,0), 3)

    # To draw an individual contour, say 4th contour:
    # cv.drawContours(imgs, contours, 3, (0,255,0), 3)

    # But most of the time, below method will be useful:
    # cnt = contours[4]
    # cv.drawContours(imgs, [cnt], 0, (0,255,0), 100)

    img = cv.resize(imgs, (960, 540))
    # fig = plt.figure(figsize = (7, 7))

    # ax = fig.add_subplot(111)
    # ax.imshow(imgs)
    # plt.show(block=False)

    # plt.pause(9)

    # plt.close('all')

    cv.imshow(f"{mask_name}", img)
    cv.waitKey(500) 
    cv.destroyAllWindows()
