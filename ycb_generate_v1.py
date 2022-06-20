import time
from matplotlib import pyplot as plt
import cv2 as cv

img = cv.imread('./N1_0_mask.pbm', -1)
# img = cv.imread('./6399621.jpg', -1)
imgs = cv.cvtColor(img.copy(), cv.COLOR_BGR2RGB)
imgRGB = cv.cvtColor(imgs, cv.COLOR_RGB2GRAY)
thresh = cv.threshold(imgRGB, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]


# _, ctrs, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# boxes = []
# for ctr in ctrs:
#     x, y, w, h = cv.boundingRect(ctr)
#     boxes.append([x, y, w, h])

# for box in boxes:
#     top_left     = (box[0], box[1])
#     bottom_right = (box[0] + box[2], box[1] + box[3])
#     cv.rectangle(img, top_left, bottom_right, (0,255,0), 2)


ROI_number = 0
contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# print(len(contours))
contours = contours[0] if len(contours) == 2 else contours[1]


x,y,w,h = cv.boundingRect(contours[len(contours) - 1])
print([x,y,w,h])
cv.rectangle(imgs, (x - 25, y - 25), (x + w + 25, y + h + 25), (255,0,0), 5)

# for c in contours:
#     x,y,w,h = cv.boundingRect(c)
#     print([x,y,w,h])
#     cv.rectangle(imgs, (x - 25, y - 25), (x + w + 25, y + h + 25), (255,0,0), 5)

# To draw all the contours in an image:
# cv.drawContours(imgs, contours, -1, (0,255,0), 3)

# To draw an individual contour, say 4th contour:
# cv.drawContours(imgs, contours, 3, (0,255,0), 3)

# But most of the time, below method will be useful:
# cnt = contours[4]
# cv.drawContours(imgs, [cnt], 0, (0,255,0), 100)

# img = cv.resize(img, (960, 540))
fig = plt.figure(figsize = (7, 7))

ax = fig.add_subplot(111)
ax.imshow(imgs)
plt.show(block=False)

plt.pause(9)

plt.close('all')

# cv.imshow("Img", img)
# cv.waitKey(0) 
# cv.destroyAllWindows()
