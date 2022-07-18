#This code is used to segment the images into road and non-road regions.
#The images need to be in the folder provided as input_image_path.
#The ground_truth is the folder where the segmented images will be saved.
#The mask_image_path is the folder where the mask images will be saved.
#The combined_root is the folder where the combined channel_image will be saved.
#The crop_root is the folder where the cropped input images will be saved.

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

def init_labels():

    # Read labels
    with open("function.yaml", 'rb') as function_file:
        functionconfig = yaml.safe_load(function_file)

    labels_spec = functionconfig['metadata']['annotations']['spec']
    labels = {item['id']: item['name'] for item in json.loads(labels_spec)}
    return labels

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

labels = init_labels()
print("INFO: label Initalized...")
print(labels[0])
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

         gTruth_mask = cv2.imread(gTruth_file_path_color)
         labelID_mask = cv2.imread(gTruth_file_path_labelID)
         labelID_mask = cv2.cvtColor(labelID_mask,cv2.COLOR_BGR2GRAY)

         gTruth_mask = cv2.cvtColor(gTruth_mask,cv2.COLOR_BGR2RGB)

         image_resized = cv2.resize(image[0:1024,384:1280+384],dsize=(720,576),interpolation=cv2.INTER_NEAREST)
         gTruth_mask_resized = cv2.resize(gTruth_mask[0:1024,384:1280+384],dsize=(720,576),interpolation=cv2.INTER_NEAREST)
         labelID_mask_resized  = cv2.resize(labelID_mask[0:1024,384:1280+384],dsize=(720,576),interpolation=cv2.INTER_NEAREST)
         #print("Image: {} processed".format(image_resized.shape))
         #print("Image: {} processed".format(gTruth_mask_resized.shape))
         #print("Image: {} processed".format(labelID_mask_resized.shape))
         
         temp = np.zeros((576,720,3), dtype=np.uint8)
         obj_mask = np.zeros((576,720,34), dtype=np.uint8)
         obj_image = np.zeros((576,720,3,34), dtype=np.uint8)
         final_image = np.zeros((576,720,3), dtype=np.uint8)

         for i in range(0,33):
           obj_mask[:,:,i] = ((labelID_mask_resized == i)*255).astype(np.uint8)
           bgr_obj_mask = np.stack((obj_mask[:,:,i],)*3,axis=-1)
           obj_image[:,:,:,i] = cv2.bitwise_and(image_resized, bgr_obj_mask)
           if i == 24:
                temp = obj_image[:,:,:,i]
                temp = cv2.GaussianBlur(temp,(25,25),0)
                temp = cv2.GaussianBlur(temp,(25,25),0)
                obj_image[:,:,:,i] = cv2.bitwise_and(temp, bgr_obj_mask)
           elif i==25:
                temp = obj_image[:,:,:,i]
                temp = cv2.GaussianBlur(temp,(25,25),0)
                temp = cv2.GaussianBlur(temp,(25,25),0)
                obj_image[:,:,:,i] = cv2.bitwise_and(temp, bgr_obj_mask)
           elif i==26:
                temp = obj_image[:,:,:,i]
                temp = cv2.GaussianBlur(temp,(25,25),0)
                temp = cv2.GaussianBlur(temp,(25,25),0)
                obj_image[:,:,:,i] = cv2.bitwise_and(temp, bgr_obj_mask)
           elif i==27:
                temp = obj_image[:,:,:,i]
                temp = cv2.GaussianBlur(temp,(25,25),0)
                temp = cv2.GaussianBlur(temp,(25,25),0)
                obj_image[:,:,:,i] = cv2.bitwise_and(temp, bgr_obj_mask)
           elif i==28:
                temp = obj_image[:,:,:,i]
                temp = cv2.GaussianBlur(temp,(25,25),0)
                temp = cv2.GaussianBlur(temp,(25,25),0)
                obj_image[:,:,:,i] = cv2.bitwise_and(temp, bgr_obj_mask)
           elif i==29:
                temp = obj_image[:,:,:,i]
                temp = cv2.GaussianBlur(temp,(25,25),0)
                temp = cv2.GaussianBlur(temp,(25,25),0)
                obj_image[:,:,:,i] = cv2.bitwise_and(temp, bgr_obj_mask)
           final_image = final_image + obj_image[:,:,:,i]
           
         print("Image: {} processed".format(file))
         cv2.imwrite(os.path.join(output_image_path,file),final_image)
         #color = (0, 0, 255)
         #cv2.putText(final_image , "Testing", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
         #cv2.imshow(file, final_image)
         #k = cv2.waitKey(0)
         #cv2.destroyAllWindows()
         #if k == 27: 
         #   cv2.destroyAllWindows()
         #   break
###################################################