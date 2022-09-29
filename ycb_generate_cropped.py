from matplotlib import pyplot as plt
import glob, os, time, torch, uuid, shutil, inspect, ruamel.yaml, json, download_ycb_dataset as download_ycb, numpy as np, cv2 as cv

def main():
    ycb_download_location = f'{os.getcwd()}/models/ycb'

    object_class = 0
    objects_dict = []

    longest_min = 150 
    longest_max = 800
    
    ## Download All The Needed Models:
    download_ycb.main()

    for object_name in os.listdir(ycb_download_location):
        objects_dict.append(objects_dict_new(object_name, longest_min, longest_max))
        
        images_folder_path = f'{ycb_download_location}/{object_name}/images'
        masks_directory_location = f'{ycb_download_location}/{object_name}/masks' 
            
        # i = 0
        
        ## For Removing 'Poses' Folder And 'calibration.h5' File:
        poses_folder_location = f'{ycb_download_location}/{object_name}/poses'
        calibration_file_location = f'{ycb_download_location}/{object_name}/calibration.h5'
        
        if os.path.exists(poses_folder_location):
            shutil.rmtree(poses_folder_location)

        if os.path.exists(calibration_file_location):
            os.remove(calibration_file_location)

        if os.path.exists(images_folder_path):
            shutil.rmtree.remove(images_folder_path)

        os.mkdir(images_folder_path)
        
        for mask_name, img_name in zip(sorted(os.listdir(masks_directory_location), key=maskParseFilter), 
            sorted(glob.glob1(f'{ycb_download_location}/{object_name}', '*.jpg'), key=imgParseFilter)):

            mask_img = cv.imread(f'{masks_directory_location}/{mask_name}', -1)
            
            img_rgb = cv.cvtColor(mask_img.copy(), cv.COLOR_BGR2RGB)
            img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
            thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

            ROI_number = 0

            contours, hierarchy  = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            sorted_contours = sorted(contours, key=cv.contourArea, reverse= True)

            X,Y,W,H = cv.boundingRect(sorted_contours[0])

            X = X - 40
            Y = Y - 40

            W = W + 70
            H = H + 70
            
            old_img_name = f'{ycb_download_location}/{object_name}/{img_name}'
            org_img = cv.imread(old_img_name, -1)

            cropped_mask_image = mask_img[Y:Y+H, X:X+W]
            cropped_org_image = org_img[Y:Y+H, X:X+W]

            ## Drawing Rectangle Bounding Box On Images
            # cv.rectangle(img_rgb, (x, y), (x + w, y + h), (255,0,0), 5)
            # cv.rectangle(img_rgb, (x - 30, y - 30), (x + w + 65, y + h + 65), (255,0,0), 5)
            # print([x - 30, y - 30, x + w + 30, y + h + 30])

            ## For Renaming Images' Name:
            split_img_name = img_name.split('.')
            split_mask_name = mask_name.split('.')
            # print(split_mask_name)

            new_mask_name = f'{split_img_name[0]}.{split_mask_name[1]}'
            
            cv.imwrite(f'{images_folder_path}/{img_name}', cropped_org_image)
            cv.imwrite(f'{masks_directory_location}/{new_mask_name}', cropped_mask_image)
            
            os.remove(old_img_name)
            os.remove(f'{masks_directory_location}/{mask_name}')
            
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
    objects_dict = dict(enumerate(objects_dict))
    with open(f"{ycb_download_location}/objects_dict_json.json", "w") as outfile:
        json.dump(objects_dict, outfile, sort_keys=True, indent=4)
    write_custom_yolo_yaml(num_classes=len(objects_dict))  # Can be done manually too

def objects_dict_new(object_name, longest_min, longest_max):
    return {'folder': object_name, 'longest_min': longest_min, 'longest_max': longest_max}

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

def write_custom_yolo_yaml(num_classes):
   model_version = input("Please input your yolov5 model version (e.g s, s6, x, ...): ")
   yolov5_models_directory = "../Object Detection Files/yolov5/models/"
   model_version = str.strip(model_version).replace(" ", "")
   pre_existing_yaml_location = find_files(f'yolov5{model_version}.yaml', f'{yolov5_models_directory}')[0]
   
   if (pre_existing_yaml_location == []):
      print("Error: Yolo Provided Yaml Config File Not Found")
      exit(1)

   yaml = ruamel.yaml.YAML()
   yaml.preserve_quotes = True
   yaml.default_flow_style = False
   
   # Read Pre-existing Yaml Config File:
   with open(f'{pre_existing_yaml_location}') as yamlFile:
      yamlFileContents = yaml.load(yamlFile)
      yaml.indent(offset=1)
   
   with open(yolov5_models_directory + f"/custom_yolov5{model_version}.yaml", 'w') as custom_yaml_file:
      yamlFileContents['nc'] = num_classes
      yaml.dump( yamlFileContents, custom_yaml_file)
      shutil.move('./models/ycb/objects_dict_json.json', './data/objects_dict_json.json')

def find_files(filename, search_path):
   result = []

  # Walking top-down from the root
   for root, dir, files in os.walk(search_path):
      if filename in files:
         result.append(os.path.join(root, filename))
   return result

if __name__ == "__main__":
    main()
