import glob, os, time, torch, uuid, shutil, inspect
from tkinter.ttk import Treeview 
import download_ycb_dataset as download_ycb
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv


def main():
    ycb_download_location = f'{os.getcwd()}/models/ycb'
    train_folder_location = f'{os.getcwd()}/models/train'
    object_class = 0
    objects_array = []
    
    labels_folder_path = os.path.join(train_folder_location,"labels")
    images_folder_path = os.path.join(train_folder_location,"images")
    
    ## Download All The Needed Models:
    download_ycb.main()
    
    ## For Making Labels Folder For Each Object:
    if os.path.exists(labels_folder_path):
        shutil.rmtree(labels_folder_path)

    if os.path.exists(images_folder_path):
        shutil.rmtree(images_folder_path)

    os.makedirs(labels_folder_path, exist_ok=False)
    os.makedirs(images_folder_path, exist_ok=False)

    for object_name in os.listdir(ycb_download_location):
        objects_array.append(object_name)

        # i = 0
        masks_directory_location = f'{ycb_download_location}/{object_name}/masks' 
        
        ## For Removing 'Poses' Folder And 'calibration.h5' File:
        poses_folder_location = f'{ycb_download_location}/{object_name}/poses'
        calibration_file_location = f'{ycb_download_location}/{object_name}/calibration.h5'
        
        if os.path.exists(poses_folder_location):
            shutil.rmtree(poses_folder_location)

        if os.path.exists(calibration_file_location):
            os.remove(calibration_file_location)

        for mask_name, img_name in zip(sorted(os.listdir(masks_directory_location), key=maskParseFilter), 
            sorted(glob.glob1(f'{ycb_download_location}/{object_name}', '*.jpg'), key=imgParseFilter)):
            
            # print(mask_name, img_name)

            img = cv.imread(f'{masks_directory_location}/{mask_name}', -1)
            # print(imgName)
            img_rgb = cv.cvtColor(img.copy(), cv.COLOR_BGR2RGB)
            img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
            thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

            ROI_number = 0

            contours, hierarchy  = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            sorted_contours = sorted(contours, key=cv.contourArea, reverse= True)

            x,y,w,h = cv.boundingRect(sorted_contours[0])

            ## Drawing Rectangle Bounding Box On Images
            # cv.rectangle(img_rgb, (x, y), (x + w, y + h), (255,0,0), 5)
            # cv.rectangle(img_rgb, (x - 30, y - 30), (x + w + 65, y + h + 65), (255,0,0), 5)
            # print([x - 30, y - 30, x + w + 30, y + h + 30])
            
            # print(img.shape)
            w_img = img.shape[1]
            h_img = img.shape[0]
            
            normalizedBBoxCoordinates = normalize_bbox(object_class, x - 10, y - 10, w + 10, h + 10, w_img, h_img) 
            # normalizedBBoxCoordinates = normalize_bbox(0, x - 35, y - 35, w + 80, h + 90, w_img, h_img)

            # print (" ".join(map(str, normalizedBBoxCoordinates)))
            
            ## For Making bBox Coordinates text file:
            file_unique_id = uuid.uuid1()
            img_name_modified = img_name.replace('.', '_')
            file_path = os.path.join(labels_folder_path, f'{object_name}_{img_name_modified}.{file_unique_id}.txt')
            
            with open(file_path, "w") as file:
                file.write(" ".join(map(str, normalizedBBoxCoordinates)))
            
            ## For Renaming Images' Name:
            old_img_name = f'{ycb_download_location}/{object_name}/{img_name}'
            img_name_split = img_name.split('.')
            new_img_name = f'{images_folder_path}/{object_name}_{img_name_modified}.{file_unique_id}.{img_name_split[1]}'

            # os.rename(old_img_name, new_img_name)
            shutil.move(old_img_name, new_img_name)

            ## For Matplotlib Plots:
            # fig = plt.figure(figsize = (7, 7))
            # ax = fig.add_subplot(111)
            # ax.imshow(img_rgb)
            # plt.show(block=False)
            
            ## For cv Image View:
            # i += 1
            # img = cv.resize(img_rgb, (960, 540))
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
    generate_data_yaml(objects_array)
    shutil.rmtree(ycb_download_location)

def generate_data_yaml(objects_array):
    models_folder_location = f'{os.getcwd()}/models'    
    data_yaml_filepath = f"{models_folder_location}/data.yaml"
    object_classes = len(objects_array)
    with open(data_yaml_filepath, "w") as file:
        file.write(inspect.cleandoc(f'''train: ../train/images
            val: ../train/images\n
            nc: {object_classes}
            names: {objects_array}'''))

def maskParseFilter(fname):
    prefix, n1, n2 = fname.split('_')
    return (prefix, int(n1))

def imgParseFilter(fname):
    prefix, n1 = fname.split('_')
    return (prefix, int(n1.split('.')[0]))

def normalize_bbox(label_index, xmin, ymin,
    w, h, w_img, h_img):
    xcenter = (xmin + w/2) / w_img
    ycenter = (ymin + h/2) / h_img
    w = w / w_img
    h = h / h_img
    return [label_index, xcenter, ycenter, w, h]

if __name__ == "__main__":
    main()