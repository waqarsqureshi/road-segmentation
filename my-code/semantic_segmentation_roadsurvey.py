#This code is used to segment the images into different regions.
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

labels = init_labels()
print("INFO: label Initalized...")

set_no = os.path.join("{}".format(args["type"]))
city_no = os.path.join("{}".format(args["class"]))
output_path = path+set_no+"_out/"

input_path = os.path.join(path,set_no,city_no)
output_path = os.path.join(path,output_path,city_no)
print("INFO: Input path: {}".format(input_path))
print("INFO: Output path: {}".format(output_path))



try:
    os.makedirs(output_path, exist_ok=True)
    print("Directory created")
except OSError as e:
    print("Directory created")

alpha = 0.3

####################################################
ie = IECore()

net = ie.read_network( model="./model/semantic-segmentation-adas-0001.xml", weights="./model/semantic-segmentation-adas-0001.bin")
device = "MULTI:CPU,GPU" if "GPU" in ie.available_devices else "CPU"
print(device)
# The GPU is running slower than CPU as of now.
exec_net = ie.load_network(net, 'CPU')

output_layer_ir = next(iter(exec_net.outputs))
input_layer_ir = next(iter(exec_net.input_info))
print("Model Loaded")
output_layer_ir = next(iter(exec_net.outputs))
input_layer_ir = next(iter(exec_net.input_info))

kernel = np.ones((25,25),np.uint8)

###################################################
for root, sub_dirs, files in os.walk(input_path):
    for file in files:
        if(file.endswith(".jpg")):
         #file = "4A18RAR351A 0000.340 1.jpg"   # just for sample test
         orig = cv2.imread(os.path.join(root,file)) # Read the image using opencv format
         
         image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
         image_h, image_w, _ = orig.shape # can the shape of the image be obtained
         ##########################################################################
         N, C, H, W = net.input_info[input_layer_ir].tensor_desc.dims
         #OpenCV resize expects the destination size as (width, height)
         resized_image = cv2.resize(orig, (W, H))# the input to the model is a BGR image
         input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0) 
         result = exec_net.infer(inputs={input_layer_ir: input_image})
         result_ir = result[output_layer_ir]
         labelID_mask  = result_ir[0,0,:,:]
         obj_mask = np.zeros((H,W,34), dtype=np.uint8)
         obj_image = np.zeros((H,W,3,34), dtype=np.uint8)
         final_image = np.zeros((H,W,3), dtype=np.uint8)
       
         for i in range(0,20):
           obj_mask[:,:,i] = ((labelID_mask == i)*255).astype(np.uint8)
           bgr_obj_mask = np.stack((obj_mask[:,:,i],)*3,axis=-1)
           obj_image[:,:,:,i] = cv2.bitwise_and(resized_image, bgr_obj_mask)
           if i == 11:
                temp = obj_image[:,:,:,i]
                #temp = cv2.GaussianBlur(temp,(25,25),0)
                temp= cv2.dilate(temp,kernel,iterations = 1)
                obj_image[:,:,:,i] = cv2.bitwise_and(temp, bgr_obj_mask)
           elif i==12:
                temp = obj_image[:,:,:,i]
                #temp = cv2.GaussianBlur(temp,(25,25),0)
                temp = cv2.dilate(temp,kernel,iterations = 1)
                obj_image[:,:,:,i] = cv2.bitwise_and(temp, bgr_obj_mask)
           elif i==13:
                temp = obj_image[:,:,:,i]
                #temp = cv2.GaussianBlur(temp,(25,25),0)
                temp = cv2.dilate(temp,kernel,iterations = 1)
                obj_image[:,:,:,i] = cv2.bitwise_and(temp, bgr_obj_mask)
           elif i==14:
                temp = obj_image[:,:,:,i]
                #temp = cv2.GaussianBlur(temp,(25,25),0)
                temp = cv2.dilate(temp,kernel,iterations = 1)
                obj_image[:,:,:,i] = cv2.bitwise_and(temp, bgr_obj_mask)
           elif i==15:
                temp = obj_image[:,:,:,i]
                #temp = cv2.GaussianBlur(temp,(25,25),0)
                temp = cv2.dilate(temp,kernel,iterations = 1)
                obj_image[:,:,:,i] = cv2.bitwise_and(temp, bgr_obj_mask)
           elif i==17:
                temp = obj_image[:,:,:,i]
                #temp = cv2.GaussianBlur(temp,(25,25),0)
                temp = cv2.dilate(temp,kernel,iterations = 1)
                obj_image[:,:,:,i] = cv2.bitwise_and(temp, bgr_obj_mask)
           final_image = final_image + obj_image[:,:,:,i]
           
         print("Image: {} processed".format(file))
         resized_image = cv2.resize(final_image, (image_w, image_h)) #resized color mask
         color = (150, 150, 50)
         cv2.putText(resized_image , "PMS ROAD SURVEY", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
         cv2.imwrite(os.path.join(output_path,file),resized_image)
        '''
         cv2.imshow(file, final_image)
         k = cv2.waitKey(0)
         cv2.destroyAllWindows()
         if k == 27: 
          cv2.destroyAllWindows()
          break
        '''
###################################################