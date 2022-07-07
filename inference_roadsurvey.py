# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import print_function, absolute_import, division
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette, get_classes

import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse, os, glob, sys,random, json, base64,io

input_path = "/home/pms/mmsegmentation/data/roadsurvey/test/"
image_path = "leftImg8bit/"
gTruth_path = "gtFine/"
output_path = "roadsurvey_output/"
segImg_path = "segImg/"
augImg_path = "augImg/"
cropImg_path = "cropImg/"
#####################################
def result_segImage(result,
                    palette=None,
                    CLASS_NAMES=None):
        """Draw `result` over `img`.

        """
        seg = result[0]
        if palette is None:
            state = np.random.get_state()
            np.random.seed(42)
            # random palette
            palette = np.random.randint(0, 255, size=(len(CLASS_NAMES), 3))
            np.random.set_state(state)

        palette = np.array(palette)
        assert palette.shape[0] == len(CLASS_NAMES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            if(label == 0):
                continue
            if(label == 1):
                continue
            if(label == 2):
                continue
            if(label == 3): # road
                color_seg[seg == label, :] = [255,255,255] #color
            if(label == 4): # 
                continue
            if(label == 5): #
                continue
            if(label == 6): #
                continue
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        return color_seg
#####################################

def main() :
    parser = argparse.ArgumentParser(description='RoadSurvey inference')
    parser.add_argument("-path", "--path", type=str, help="path to input test directory", default=input_path)
    parser.add_argument("-config", "--config", type=str, help="path to config file", default="/home/pms/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_40k_roadsurvey.py")
    parser.add_argument("-checkpoint", "--checkpoint", type=str, help="path to checkpoint file", default="/home/pms/mmsegmentation/work_dirs/deeplabv3plus_r50-d8_512x512_40k_roadsurvey/latest.pth")
    args = vars(parser.parse_args())
    path = os.path.join(os.getcwd(), "{}".format(args["path"]))
    config = os.path.join(os.getcwd(), "{}".format(args["config"]))
    checkpoint = os.path.join(os.getcwd(), "{}".format(args["checkpoint"]))
    roadsurveyPath = path
    print("roadsurveyPath: ", roadsurveyPath)
    # build the model from a config file and a checkpoint file
    model = init_segmentor(config, checkpoint , device='cuda:0')
    # inference the model on a given image
    searchImage = os.path.join(roadsurveyPath , "*" , "*_leftImg8bit.jpg")
    filesImage = glob.glob( searchImage)
    filesImage.sort()
    files = filesImage
    if not files:
        print( "Did not find any files. Please consult the README." )
        sys.exit(1)
    # a bit verbose
    print("Processing {} files".format(len(files)))
    progress = 0
    print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
    for file in files:
        orig = cv2.imread(os.path.join(file)) # Read the image using opencv format
        image = orig#cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        result = inference_segmentor(model, image)
        mask = result_segImage(result, palette=get_palette('roadsurvey'), CLASS_NAMES=get_classes('roadsurvey'))
        segImage = cv2.bitwise_and(orig, mask)
        imgAug = cv2.merge((orig[:,:,0],segImage[:,:,1],segImage[:,:,2]))
        dst = file
        seg_out = os.path.join("/",dst.split("/")[1],dst.split("/")[2],output_path,dst.split("/")[4],dst.split("/")[5],dst.split("/")[6],segImg_path,dst.split("/")[7])
        aug_out = os.path.join("/",dst.split("/")[1],dst.split("/")[2],output_path,dst.split("/")[4],dst.split("/")[5],dst.split("/")[6],augImg_path,dst.split("/")[7])
        crop_out = os.path.join("/",dst.split("/")[1],dst.split("/")[2],output_path,dst.split("/")[4],dst.split("/")[5],dst.split("/")[6],cropImg_path,dst.split("/")[7])

        try:
            os.makedirs(seg_out, exist_ok=True)
            os.makedirs(aug_out, exist_ok=True)
            os.makedirs(crop_out, exist_ok=True)
        except OSError as e:
            print("Directory created")
        cv2.imwrite(os.path.join(seg_out,dst.split("/")[8]),segImage[230:560,0:700])
        cv2.imwrite(os.path.join(aug_out,dst.split("/")[8]),imgAug[230:560,0:700])
        cv2.imwrite(os.path.join(crop_out,dst.split("/")[8]),orig[230:560,0:700])
        #k = cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #if k == 27: 
           #cv2.destroyAllWindows()
           #break
        #cv2.destroyAllWindows()

        progress += 1
        print("\rProgress: {:>3} %".format( progress * 100 / len(filesImage) ), end=' ')
        sys.stdout.flush()
    print("\nDone!")
if __name__ == '__main__':
    main()
