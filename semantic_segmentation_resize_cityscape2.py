'''
This code is used to convert the cityscape dataset into the following format:
1. The images are cropped and resized to the specified size.
2. The groundTruth are converted to the specified color space.
3. The labelIds are converted to the specified index space
4. The images and groundTruth are saved in the specified path.

id          class                color
0      background        [  0,  0,  0]
1           human        [220, 20, 60]
2            pole        [153,153,153]
3            road        [128, 64,128]
4   traffic light        [250,170, 30]
5    traffic sign        [220,220,  0]
6         vehicle        [  0,  0,142]

'''

from re import L
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
from openvino.inference_engine import IECore
import argparse
import os
argparse = argparse.ArgumentParser()
import random
from skimage.measure import approximate_polygon, find_contours

sys.path.append("./utils")
from notebook_utils import segmentation_map_to_image
from notebook_utils import download_ir_model

import json
import base64
from PIL import Image
import io
import yaml

from collections import namedtuple

from labels import labels
from labels_roadsurvey import labelsRoadsurvey


print(labelsRoadsurvey[0].Id, labelsRoadsurvey[0].color, labelsRoadsurvey[0].name)
print(labelsRoadsurvey[1].Id, labelsRoadsurvey[1].color, labelsRoadsurvey[1].name)
print(labelsRoadsurvey[2].Id, labelsRoadsurvey[2].color, labelsRoadsurvey[2].name)
print(labelsRoadsurvey[3].Id, labelsRoadsurvey[3].color, labelsRoadsurvey[3].name)
print(labelsRoadsurvey[4].Id, labelsRoadsurvey[4].color, labelsRoadsurvey[4].name)
print(labelsRoadsurvey[5].Id, labelsRoadsurvey[5].color, labelsRoadsurvey[5].name)
print(labelsRoadsurvey[6].Id, labelsRoadsurvey[6].color, labelsRoadsurvey[6].name)
#'''

argparse.add_argument(
    "-i", "--path", type=str, help="path to input directory"
)

argparse.add_argument(
    "-tv", "--type", type=str, help="path to input image folder containing the images"
)

argparse.add_argument(
    "-c", "--class", type=str, help="relative path of the city folder"
)


args = vars(argparse.parse_args())
path = os.path.join(os.getcwd(), "{}".format(args["path"]))
## input arguments can be changed as per need
input_path = "cityscapes/"
image_path = "leftImg8bit/"
gTruth_path = "gtFine/"
output_path = "cityscape_output/"

set_no = os.path.join("{}".format(args["type"]))
city_no = os.path.join("{}".format(args["class"]))

input
input_image_path = os.path.join(path,input_path,image_path,set_no,city_no)
input_gtruth_path = os.path.join(path,input_path,gTruth_path,set_no,city_no)

output_image_path = os.path.join(path,output_path,image_path,set_no,city_no)
output_ground_truth = os.path.join(path,output_path,gTruth_path,set_no,city_no)

print("INFO: Input image path: {}".format(input_image_path))
print("INFO: gTruth path: {}".format(input_gtruth_path ))
print("INFO: output image path: {}".format(output_image_path))
print("INFO: output gTruth path: {}".format(output_ground_truth))


try:
    os.makedirs(output_image_path, exist_ok=True)
    print("Directory created")
except OSError as e:
    print("Directory created")

try:
    os.makedirs(output_ground_truth , exist_ok=True)
    print("Directory created")
except:
    print("Directory created")

alpha = 0.3
###################################################
for root, sub_dirs, files in os.walk(input_image_path):
    for file in files:
        if(file.endswith(".png")):
         orig = cv2.imread(os.path.join(root,file)) # Read the image using opencv format
         image = orig#cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
         
         gTruth_file_color = file.split("_")[0]+"_"+file.split("_")[1] +"_"+file.split("_")[2]+"_gtFine_color.png"
         gTruth_file_labelId =file.split("_")[0]+"_"+ file.split("_")[1]+ "_"+file.split("_")[2]+"_gtFine_labelIds.png"

         gTruth_file_path_color = os.path.join(input_gtruth_path,gTruth_file_color)
         gTruth_file_path_labelID = os.path.join(input_gtruth_path,gTruth_file_labelId)

         print("Image: {} processed".format(gTruth_file_path_labelID))
         print("Image: {} processed".format(file))
         gTruth_mask = cv2.imread(gTruth_file_path_color)
         labelID_mask = cv2.imread(gTruth_file_path_labelID)
         labelID_mask = cv2.cvtColor(labelID_mask,cv2.COLOR_BGR2GRAY)

         gTruth_mask = cv2.cvtColor(gTruth_mask,cv2.COLOR_BGR2RGB)
         image_resized = cv2.resize(image[0:1024,384:1280+384],dsize=(720,576),interpolation=cv2.INTER_NEAREST)
         gTruth_mask_resized = cv2.resize(gTruth_mask[0:1024,384:1280+384],dsize=(720,576),interpolation=cv2.INTER_NEAREST)
         labelID_mask_resized  = cv2.resize(labelID_mask[0:1024,384:1280+384],dsize=(720,576),interpolation=cv2.INTER_NEAREST)  
         #######################################################
         
         obj_mask = np.zeros((576,720), dtype=np.uint8)
         obj_color_mask= np.zeros((576,720,3), dtype=np.uint8)
         final_label = np.zeros((576,720), dtype=np.uint8)
         final_mask = np.zeros((576,720,3), dtype=np.uint8)

         for label in labels: #i is the label id from cityscapes
            if label.name == 'person': # human=1
                obj_mask  = ((labelID_mask_resized == label.id)*labelsRoadsurvey[1].Id).astype(np.uint8)
                temp  = ((labelID_mask_resized == label.id)*255).astype(np.uint8)
                obj_color_mask_temp= ((gTruth_mask_resized == label.color)*labelsRoadsurvey[1].color).astype(np.uint8)
                temp_3 = np.stack((temp,)*3,axis=-1)
                obj_color_mask = cv2.bitwise_and(obj_color_mask_temp, temp_3)
            elif label.name == 'rider': # human=1
                obj_mask  = ((labelID_mask_resized == label.id)*labelsRoadsurvey[1].Id).astype(np.uint8)
                temp  = ((labelID_mask_resized == label.id)*255).astype(np.uint8)
                obj_color_mask_temp= ((gTruth_mask_resized == label.color)*labelsRoadsurvey[1].color).astype(np.uint8)
                temp_3 = np.stack((temp,)*3,axis=-1)
                obj_color_mask = cv2.bitwise_and(obj_color_mask_temp, temp_3)
            elif label.name == 'pole': #pole=2
                obj_mask  = ((labelID_mask_resized == label.id)*labelsRoadsurvey[2].Id).astype(np.uint8)
                temp  = ((labelID_mask_resized == label.id)*255).astype(np.uint8)
                obj_color_mask_temp= ((gTruth_mask_resized == label.color)*labelsRoadsurvey[2].color).astype(np.uint8)
                temp_3 = np.stack((temp,)*3,axis=-1)
                obj_color_mask = cv2.bitwise_and(obj_color_mask_temp, temp_3)
            elif label.name == 'road': #3
                obj_mask  = ((labelID_mask_resized == label.id)*labelsRoadsurvey[3].Id).astype(np.uint8)
                temp  = ((labelID_mask_resized == label.id)*255).astype(np.uint8)
                obj_color_mask_temp= ((gTruth_mask_resized == label.color)*labelsRoadsurvey[3].color).astype(np.uint8)
                temp_3 = np.stack((temp,)*3,axis=-1)
                obj_color_mask = cv2.bitwise_and(obj_color_mask_temp, temp_3)
            elif label.name == 'traffic light': #4
                obj_mask  = ((labelID_mask_resized == label.id)*labelsRoadsurvey[4].Id).astype(np.uint8)
                temp  = ((labelID_mask_resized == label.id)*255).astype(np.uint8)
                obj_color_mask_temp= ((gTruth_mask_resized == label.color)*labelsRoadsurvey[4].color).astype(np.uint8)
                temp_3 = np.stack((temp,)*3,axis=-1)
                obj_color_mask = cv2.bitwise_and(obj_color_mask_temp, temp_3)
            elif label.name == 'traffic sign': #5
                obj_mask  = ((labelID_mask_resized == label.id)*labelsRoadsurvey[5].Id).astype(np.uint8)
                temp  = ((labelID_mask_resized == label.id)*255).astype(np.uint8)
                obj_color_mask_temp= ((gTruth_mask_resized == label.color)*labelsRoadsurvey[5].color).astype(np.uint8)
                temp_3 = np.stack((temp,)*3,axis=-1)
                obj_color_mask = cv2.bitwise_and(obj_color_mask_temp, temp_3)
            elif label.name == 'car': #6
                obj_mask  = ((labelID_mask_resized == label.id)*labelsRoadsurvey[6].Id).astype(np.uint8)
                temp  = ((labelID_mask_resized == label.id)*255).astype(np.uint8)
                obj_color_mask_temp= ((gTruth_mask_resized == label.color)*labelsRoadsurvey[6].color).astype(np.uint8)
                temp_3 = np.stack((temp,)*3,axis=-1)
                obj_color_mask = cv2.bitwise_and(obj_color_mask_temp, temp_3)
            elif label.name == 'truck': #6
                obj_mask  = ((labelID_mask_resized == label.id)*labelsRoadsurvey[6].Id).astype(np.uint8)
                temp  = ((labelID_mask_resized == label.id)*255).astype(np.uint8)
                obj_color_mask_temp= ((gTruth_mask_resized == label.color)*labelsRoadsurvey[6].color).astype(np.uint8)
                temp_3 = np.stack((temp,)*3,axis=-1)
                obj_color_mask = cv2.bitwise_and(obj_color_mask_temp, temp_3)
            elif label.name == 'bus': #6
                obj_mask  = ((labelID_mask_resized == label.id)*labelsRoadsurvey[6].Id).astype(np.uint8)
                temp  = ((labelID_mask_resized == label.id)*255).astype(np.uint8)
                obj_color_mask_temp= ((gTruth_mask_resized == label.color)*labelsRoadsurvey[6].color).astype(np.uint8)
                temp_3 = np.stack((temp,)*3,axis=-1)
                obj_color_mask = cv2.bitwise_and(obj_color_mask_temp, temp_3)
            elif label.name == 'caravan': #6
                obj_mask  = ((labelID_mask_resized == label.id)*labelsRoadsurvey[6].Id).astype(np.uint8)
                temp  = ((labelID_mask_resized == label.id)*255).astype(np.uint8)
                obj_color_mask_temp= ((gTruth_mask_resized == label.color)*labelsRoadsurvey[6].color).astype(np.uint8)
                temp_3 = np.stack((temp,)*3,axis=-1)
                obj_color_mask = cv2.bitwise_and(obj_color_mask_temp, temp_3)
            elif label.name == 'trailer': #6
                obj_mask  = ((labelID_mask_resized == label.id)*labelsRoadsurvey[6].Id).astype(np.uint8)
                temp  = ((labelID_mask_resized == label.id)*255).astype(np.uint8)
                obj_color_mask_temp= ((gTruth_mask_resized == label.color)*labelsRoadsurvey[6].color).astype(np.uint8)
                temp_3 = np.stack((temp,)*3,axis=-1)
                obj_color_mask = cv2.bitwise_and(obj_color_mask_temp, temp_3)
            else:
                obj_mask =  np.zeros((576,720), dtype=np.uint8)
                obj_color_mask = np.zeros((576,720,3), dtype=np.uint8)

            final_label= final_label + obj_mask
            final_mask = final_mask  + obj_color_mask
         #######################################################
         #final_mask = cv2.cvtColor(final_mask,cv2.COLOR_RGB2BGR)
         cv2.imwrite(os.path.join(output_image_path,file),image_resized)
         cv2.imwrite(os.path.join(output_ground_truth,gTruth_file_color),final_mask)
         cv2.imwrite(os.path.join(output_ground_truth,gTruth_file_labelId),final_label)
         #color = (0, 0, 255)
         #cv2.putText(final_image , "Testing", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
         #cv2.imshow(file, final_mask)
         
         #k = cv2.waitKey(0)
         #cv2.destroyAllWindows()
         #if k == 27: 
            #cv2.destroyAllWindows()
            #break
###################################################

#'''