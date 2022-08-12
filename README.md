# Ycb_Dataset_Generator
## Installation

### Commands to be executed in order:
1. ```git clone``` [Ycb_Dataset_Generator](https://github.com/VT-Collab/Ycb_Dataset_Generator.git)
2. ```cd ./Ycb_Dataset_Generator```
4. Setup and install ```Python 3.7.12``` or ```Python 3.8.0``` using [Pyenv](https://realpython.com/intro-to-pyenv/). **CREATE** and **ACTIVATE** a new python environment named '**YCB_generator**' running Python 3.7.12 or Python 3.8.0. (This step is required to successfully meet all the requirements)
    * ```pyenv virtualenv 3.7.12 YCB_generator``` (Install python and create python virtual environment)
    * (Only for the initial setup) ```pyenv local YCB_generator``` (Activate python virtual environment)
    * ```pyenv versions``` (Check if the correct virtual environment with intended python version is active) 
3. ```pip install -r ./Requirements/requirements.txt```
4. ```python ./ycb_generate_cropped.py``` (Generates cropped images & masks of the YCB object(s))
5. ```mv ./models/ycb/* ./data``` (To use cropped objects from previous step in dataset generatation)
6. Open ***generator_for_yolov5.ipynb*** -> run all the cells => generates ```100 train```, ```50 valid```, ```5 test``` 640x640 images with labels (yolo bounding box co-ordinates) by default.
7. Follow next instructions on [Ycb_Yolov5_Trainer](https://github.com/VT-Collab/Ycb_Yolov5_Trainer.git) 

### Possible changes needed to run the notebook:
* Packages import configuration can be different different machines:
![image](https://user-images.githubusercontent.com/68425706/184435407-15dcaf1a-8c89-4be3-82e6-d56b3e73d640.png)

### Custom dataset generation config:
* You can change the name of the custom dataset, number of images generated per split, andd add new split.
![image](https://user-images.githubusercontent.com/68425706/184442187-a4640d8c-4c72-4046-a4b1-b0de7be340c2.png)

### Resize dataset images
* Recommended image sizes: ```640x640``` OR ```416x416```
![image](https://user-images.githubusercontent.com/68425706/184442086-41e810f8-a338-437e-ab8b-ccf11fcc835a.png)

