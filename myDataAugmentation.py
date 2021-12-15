import cv2
import random
import os
import numpy as np
import pathlib
import cv2
import PIL
import PIL.Image
import time
import datetime
import argparse

argparse = argparse.ArgumentParser()

argparse.add_argument(
    "-i", "--inputDir", type=str, help="path to input directory of images"
)

argparse.add_argument(
    "-o", "--outDir", type=str, help="path to output directory of images"
)

args = vars(argparse.parse_args())

input_data_dir = os.path.join(os.getcwd(), "{}".format(args["inputDir"]))
output_data_dir  = os.path.join(os.getcwd(), "{}".format(args["outDir"]))

#input_data_dir = "/media/waqar/Data/PMS/exp-new-class/training"
#output_data_dir = "/media/waqar/Data/PMS/exp-new-class/new-training"

input_data_dir = pathlib.Path(input_data_dir)
output_data_dir = pathlib.Path(output_data_dir)

print("Input Training-Directory: ",input_data_dir)
print("Output Training-Directory: ",output_data_dir)
total_class = 3

try:
    os.makedirs(output_data_dir, exist_ok=True)
    print("New Directory created")
except OSError as e:
    print("The Directory already exists...")
count = 0
countT = 0
root, class_n, files = next(os.walk(input_data_dir))
for c in class_n:
    new_root = os.path.join(output_data_dir, c)
    try:
        os.mkdir(new_root)
    except OSError as e:
        print("The sub Directory already exists...")
    for dirs in os.walk(os.path.join(root, c)):
        for files in dirs:
            for filename in files:
                if(filename.endswith(".jpg")):
                    path = os.path.join(root, c, filename)
                    orig = cv2.imread(path)
                    img = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
                    img_height, img_width, _ = img.shape[0], img.shape[1], img.shape[2]
                    M1 = cv2.getRotationMatrix2D((img_width/2, img_height/2), 1, 1)
                    M2 = cv2.getRotationMatrix2D((img_width/2, img_height/2), -1, 1)
                    img_r1 = cv2.warpAffine(img, M1, (img_width, img_height))
                    img_r2 = cv2.warpAffine(img, M2, (img_width, img_height))
                    img_h = cv2.flip(img, 1)
                    img_v = cv2.flip(img, 0)
                    img_hv = cv2.flip(img, -1)
                    
                    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                    img_zoom = cv2.resize(img, (int(img_width*1.50), int(img_height*1.50)))
                    img_zoom = img_zoom[250:550,0:700]
                    img_guassian = cv2.GaussianBlur(img, (5,5), 0)
                    img_contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)

                    new_filename_r1 = filename.split(".")[0]+ filename.split(".")[1] + "_"  + str('r1') + ".jpg"
                    new_filename_r2 = filename.split(".")[0]+ filename.split(".")[1] + "_" + str('r2') + ".jpg"
                    new_filename_h = filename.split(".")[0]+ filename.split(".")[1] + "_" + str('hf') + ".jpg"
                    new_filename_v = filename.split(".")[0]+ filename.split(".")[1] + "_"  + str('vf') + ".jpg"
                    new_filename_hv = filename.split(".")[0]+ filename.split(".")[1] + "_" + str('hvf') + ".jpg"
                    
                    new_filename_lab = filename.split(".")[0]+ filename.split(".")[1] + "_" + str('lab') + ".jpg"
                    new_filename_zoom = filename.split(".")[0]+ filename.split(".")[1] + "_" + str('zoom') + ".jpg"
                    new_filename_guassian = filename.split(".")[0]+ filename.split(".")[1] + "_" + str('g') + ".jpg"
                    new_filename_rgb = filename.split(".")[0]+ filename.split(".")[1] + "_" + str('rgb') + ".jpg"
                    new_filename_contrast = filename.split(".")[0]+ filename.split(".")[1] + "_" + str('c') + ".jpg"
                    new_filename_orig = filename.split(".")[0]+ filename.split(".")[1] + "_" + str('orig') + ".jpg"
                    
                    new_path_h = os.path.join(new_root, new_filename_h)
                    new_path_r1 = os.path.join(new_root, new_filename_r1)
                    new_path_r2= os.path.join(new_root, new_filename_r2)
                    new_path_v = os.path.join(new_root, new_filename_v)
                    new_path_hv = os.path.join(new_root, new_filename_hv)
                    new_path_lab = os.path.join(new_root, new_filename_lab)
                    new_path_zoom = os.path.join(new_root, new_filename_zoom)
                    new_path_guassian = os.path.join(new_root, new_filename_guassian)
                    new_path_rgb = os.path.join(new_root, new_filename_rgb)
                    new_path_contrast = os.path.join(new_root, new_filename_contrast)
                    new_path_orig = os.path.join(new_root, new_filename_orig)
                    cv2.imwrite(new_path_r1, img_r1)
                    cv2.imwrite(new_path_r2, img_r2)
                    cv2.imwrite(new_path_h, img_h)
                    cv2.imwrite(new_path_v, img_v)
                    cv2.imwrite(new_path_hv, img_hv)
                    
                    cv2.imwrite(new_path_lab, img_lab)
                    cv2.imwrite(new_path_zoom, img_zoom)
                    cv2.imwrite(new_path_guassian, img_guassian)
                    cv2.imwrite(new_path_rgb, img)
                    cv2.imwrite(new_path_orig, orig)
                    cv2.imwrite(new_path_contrast, img_contrast)
                    count = count + 11;
                    print("Processed: ", filename)
    countT = countT + count
    print('class: ',c, 'Total files: ',len(files), 'Initial Images in class: ', count,'Total Images in class: ',countT)
    count = 0
    countf = 0
print('Total classes: ',c,'Total Images: ', countT)


