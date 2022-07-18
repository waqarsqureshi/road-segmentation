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
from __future__ import print_function, absolute_import, division
from re import L
import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.inference_engine import IECore
import argparse, os, glob, sys,random, json, base64,io
argparse = argparse.ArgumentParser()
from skimage.measure import approximate_polygon, find_contours

sys.path.append("./utils")
from notebook_utils import segmentation_map_to_image
from notebook_utils import download_ir_model

from PIL import Image
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
cityscapesPath = path
## input arguments can be changed as per need
input_path = "cityscapes/"
image_path = "leftImg8bit/"
gTruth_path = "gtFine/"
output_path = "cityscape_output/"

set_no = os.path.join("{}".format(args["type"]))
city_no = os.path.join("{}".format(args["class"]))
searchFine   = os.path.join( cityscapesPath , "gtFine"   , "*" , "*" , "*_gt*_color.png" )
searchImage = os.path.join( cityscapesPath , "leftImg8bit"   , "*" , "*" , "*_leftImg8bit.png" )
# search files
filesFine = glob.glob( searchFine )
filesImage = glob.glob( searchImage)

filesFine.sort()
filesImage.sort()

files = filesFine
#print(files)
#print((filesImage))
# quit if we did not find anything
if not files:
    printError( "Did not find any files. Please consult the README." )
# a bit verbose
print("Processing {} files".format(len(files)))
progress = 0
print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')


alpha = 0.3
###################################################
for file in filesImage:
    if(file.endswith(".png")):
        orig = cv2.imread(os.path.join(file)) # Read the image using opencv format
        image = orig#cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        dst = file.replace("_leftImg8bit.png","_leftImg8bit.jpg")
        image_resized = cv2.resize(image[0:1024,384:1280+384],dsize=(720,576),interpolation=cv2.INTER_NEAREST)
        dst_out = os.path.join("/",dst.split("/")[1],dst.split("/")[2],output_path,dst.split("/")[4],dst.split("/")[5],dst.split("/")[6])
        #print("INFO: output image path: {}".format(dst_out))
        
        try:
            os.makedirs(dst_out , exist_ok=True)
        except:
            print("Directory created") 

        cv2.imwrite(os.path.join(dst_out,dst.split("/")[7]),image_resized)

        progress += 1
        print("\rProgress: {:>3} %".format( progress * 100 / len(filesImage) ), end=' ')
        sys.stdout.flush()

print("Processing {} files".format(len(files)))
progress = 0
print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')

for file in files:

        gTruth_file_path_labelId  = file.replace("_gtFine_color.png","_gtFine_labelIds.png")
        gTruth_file_path_color = os.path.join(file)
        gTruth_file_path_labelId = os.path.join(gTruth_file_path_labelId)
        gTruth_file_path_color_out = os.path.join("/",file.split("/")[1],file.split("/")[2],output_path,file.split("/")[4],file.split("/")[5],file.split("/")[6])
        gTruth_file_path_labelId_out = os.path.join("/",file.split("/")[1],file.split("/")[2],output_path,file.split("/")[4],file.split("/")[5],file.split("/")[6])
        try:
            os.makedirs(gTruth_file_path_color_out, exist_ok=True)
        except:
            print("Directory created")  
        try:
            os.makedirs(gTruth_file_path_labelId, exist_ok=True)
        except:
            print("Directory created")          
        #print("Image: {} processed".format(gTruth_file_path_labelID))
        #print("Image: {} processed".format(file))
        
        gTruth_mask = cv2.imread(gTruth_file_path_color)
        labelID_mask = cv2.imread(gTruth_file_path_labelId)
        labelID_mask = cv2.cvtColor(labelID_mask,cv2.COLOR_BGR2GRAY)

        gTruth_mask = cv2.cvtColor(gTruth_mask,cv2.COLOR_BGR2RGB)
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
            elif label.name== 'car': #6
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
        #final_mask = no converversion to BGR required
        cv2.imwrite(os.path.join(gTruth_file_path_color_out,gTruth_file_path_color.split("/")[7]),final_mask)
        cv2.imwrite(os.path.join(gTruth_file_path_labelId_out,gTruth_file_path_labelId .split("/")[7]),final_label)
        progress += 1
        print("\rProgress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
        sys.stdout.flush()
        #######################################################
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