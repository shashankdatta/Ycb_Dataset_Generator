import glob, os, time, torch, uuid, shutil, inspect
import ruamel.yaml as yaml
from unittest import main
from tkinter.ttk import Treeview 
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

yolov5_models_directory = "./yolov5/models/"

def find_files(filename, search_path):
   result = []

  # Wlaking top-down from the root
   for root, dir, files in os.walk(search_path):
      if filename in files:
         result.append(os.path.join(root, filename))
   return result

def writeCustomYaml(model_version, num_classes):
   model_version = str.strip(model_version).replace(" ", "")
   pre_existing_yaml_location = find_files(f'yolov5{model_version}.yaml', f'{yolov5_models_directory}')[0]
   
   if (pre_existing_yaml_location == []):
      print("Error: Yolo Provided Yaml Config File Not Found")
      exit(1)

   # Read Pre-existing Yaml Config File:
   with open(f'{pre_existing_yaml_location}') as yamlFile:
      yamlFileContents = yaml.load(yamlFile, Loader=yaml.RoundTripLoader)

   with open(yolov5_models_directory + f"/custom_yolov5{model_version}.yaml", 'w') as custom_yaml_file:
      yamlFileContents['nc'] = num_classes
      yaml.dump(yamlFileContents, custom_yaml_file, Dumper=yaml.RoundTripDumper)

if __name__ == "__main__":
   writeCustomYaml("s6", 300)