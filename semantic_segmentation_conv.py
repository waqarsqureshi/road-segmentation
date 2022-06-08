#This code is used to segment the images into road and non-road regions.
#The images need to be in the folder provided as input_image_path.
#The segmented_image_path is the folder where the segmented images will be saved.
#The mask_image_path is the folder where the mask images will be saved.
#The combined_root is the folder where the combined channel_image will be saved.
#The crop_root is the folder where the cropped input images will be saved.

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
import os
argparse = argparse.ArgumentParser()


argparse.add_argument(
    "-i", "--path", type=str, help="path to input directory"
)

argparse.add_argument(
    "-tv", "--type", type=str, help="path to input image folder containing the images"
)

argparse.add_argument(
    "-c", "--class", type=str, help="relative path of the class folder"
)


args = vars(argparse.parse_args())
path = os.path.join(os.getcwd(), "{}".format(args["path"]))
## input arguments can be changed as per need
input_path = ""
segmented_image_path = "segmented/"
mask_image_path = "masked/"

set_no = os.path.join("{}".format(args["type"]))
class_no = os.path.join("{}".format(args["class"]))

input_image_path = os.path.join(path,input_path,set_no,class_no)
segmented_image_path = os.path.join(path,segmented_image_path,set_no,class_no)
mask_image_path = os.path.join(path,mask_image_path,set_no,class_no)

print("INFO: Input path: {}".format(input_image_path))
print("INFO: Output path: {}".format(segmented_image_path))
#print("INFO: Mask path: {}".format(mask_image_path))


try:
    os.makedirs(segmented_image_path, exist_ok=True)
    print("Directory created")
except OSError as e:
    print("Directory created")
'''
try:
    os.makedirs(mask_image_path, exist_ok=True)
    print("Directory created")
except:
    print("Directory created")

'''


###################################################
for root, sub_dirs, files in os.walk(input_image_path):
    for file in files:
      if file.endswith(".jpg_labelTrainIds.png"):
        orig = cv2.imread(os.path.join(root, file)) # Read the image using opencv format
        image_gray = np.zeros((orig.shape[0], orig.shape[1], 1), np.uint8)
        image_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY);
        image_gray2= np.zeros((orig.shape[0], orig.shape[1], 1),dtype=np.uint8)
        '''
        image_gray2[image_gray2==0]=2 # convert all the background to 2
        image_gray2[image_gray==1]=2 # background are 2
        image_gray2[image_gray==2]=0 # road pixels are 0
        image_gray2[image_gray==3]=3 # sky pixels are 3
        image_gray2[image_gray==4]=1 # sidewalk pixels are 1
        '''
        #########################
        #if commented above then uncomment the below line
        image_gray2 = image_gray
        #########################

        image_h, image_w, = image_gray.shape # can the shape of the image be obtained
        print("INFO: Image shape: {}".format(image_gray.shape))
        
        #H,W = height, width
        H, W = 1024,2048
        # if you want to resize the image mask
        resized_image = cv2.resize(image_gray2, (W, H),0,0,interpolation = cv2.INTER_NEAREST)# resize the image
       
        # Save the image after cropping
        cv2.imwrite(os.path.join(segmented_image_path , file),image_gray2) #this is the 1 channel
        #cv2.imwrite(os.path.join(mask_image_path  , file),resized_image ) # this is 1 channel resized image
        #####################################################
        print("Image {} processed".format(file))
        #color = (0, 0, 255)
        #cv2.putText(bgr_road_image, "Testing", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        '''
        cv2.imshow(file, image_gray2)
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if k == 27: 
            cv2.destroyAllWindows()
            break
        '''
###################################################
