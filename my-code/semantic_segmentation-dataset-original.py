import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
from openvino.inference_engine import IECore
import argparse
import os
argparse = argparse.ArgumentParser()

sys.path.append("./utils")
from notebook_utils import segmentation_map_to_image
from notebook_utils import download_ir_model

argparse.add_argument(
    "-i", "--images", type=str, help="path to input directory of images"
)

argparse.add_argument(
    "-o", "--output", type=str, help="path to output directory of images"
)

argparse.add_argument(
    "-m", "--mask", type=str, help="path to mask directory of images"
)

argparse.add_argument(
    "-c", "--cropped", type=str, help="path to original cropped image directory of images"
)

argparse.add_argument(
    "-mo", "--combined", type=str, help="path to output combined-image directory of images"
)

args = vars(argparse.parse_args())

data_path = os.path.join(os.getcwd(), "{}".format(args["images"]))
out_root = os.path.join(os.getcwd(), "{}".format(args["output"]))
mask_root = os.path.join(os.getcwd(), "{}".format(args["mask"]))
combined_root = os.path.join(os.getcwd(), "{}".format(args["combined"]))
cropped_root = os.path.join(os.getcwd(), "{}".format(args["cropped"]))


ie = IECore()

net = ie.read_network( model="./model/road-segmentation-adas-0001.xml")
device = "MULTI:CPU,GPU" if "GPU" in ie.available_devices else "CPU"
print(device)
# The GPU is running slower than CPU as of now.
exec_net = ie.load_network(net, 'CPU')

output_layer_ir = next(iter(exec_net.outputs))
input_layer_ir = next(iter(exec_net.input_info))
print("Model Loaded")



output_layer_ir = next(iter(exec_net.outputs))
input_layer_ir = next(iter(exec_net.input_info))
###################################################

for root, sub_dirs, files in os.walk(data_path):
    for file in files:
        orig = cv2.imread(os.path.join(root, file))
        image_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY);
        ###################################################
        # apply equilization to each channel of color image
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(15,15))
        image_gray_eq_B= clahe.apply(orig[:,:,0])
        image_gray_eq_G= clahe.apply(orig[:,:,1])
        image_gray_eq_R= clahe.apply(orig[:,:,2])
        bgr_image_eq = cv2.merge((image_gray_eq_B, image_gray_eq_G, image_gray_eq_R))
        ###################################################
        image_h, image_w, _ = orig.shape
        #h,w, r = orig.shape
        # N,C,H,W = batch size, number of channels, height, width
        N, C, H, W = net.input_info[input_layer_ir].tensor_desc.dims
        # OpenCV resize expects the destination size as (width, height)
        resized_image = cv2.resize(orig, (W, H))# the input to the model is a BGR image
        # reshape to network input shape
        input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0) 

        result = exec_net.infer(inputs={input_layer_ir: input_image})
        result_ir = result[output_layer_ir]
        ###################################################
        # Prepare data for visualization
        segmentation_mask = np.argmax(result_ir, axis=1)
        # Define colormap, each color represents a class
        colormap = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]])
        ###################################################
        # Use function from notebook_utils.py to transform mask to an RGB image
        mask = segmentation_map_to_image(segmentation_mask, colormap)
        resized_mask = cv2.resize(mask, (image_w, image_h)) #resized color mask
        ###################################################
        # make road and road+line mask from orginal masks and apply to itensity image to get the image_with_mask
        road_mask = resized_mask[:, :, 0] #this is the road channel
        road_line_mask = resized_mask[:, :, 2]+ resized_mask[:,:,0] #this is the road and line channel
        road_gray = cv2.bitwise_and(image_gray, road_mask)
        road_line_gray = cv2.bitwise_and(image_gray, road_line_mask)
        ###################################################
        # apply the mask to the color equalized image or normal bgr image choose the one you want to apply the mask to and comment the other one
        bgr_road_mask = np.stack((road_mask,)*3,axis=-1)
        #bgr_road_image = cv2.bitwise_and(bgr_image_eq,bgr_road_mask)
        bgr_road_image = cv2.bitwise_and(orig,bgr_road_mask)
        ###################################################
        # Define the transparency of the segmentation mask on the photo
        alpha = 0.3
        # Create image after removing the background obtained through masking
        #image_with_mask = cv2.addWeighted(resized_mask, alpha,orig, 1 - alpha, 0)
        image_with_mask = cv2.merge((road_gray,road_line_gray,image_gray))
        ###################################################
        # Save the image after cropping
        cv2.imwrite(os.path.join(combined_root, file),image_with_mask[230:560,0:700]) #this is the image with mask
        cv2.imwrite(os.path.join(cropped_root, file),orig[230:560,0:700]) # this is the original cropped image
        cv2.imwrite(os.path.join(out_root, file),bgr_road_image[230:560,0:700]) #[250:550,0:700]
        #cv2.imwrite(os.path.join(mask_root, file),resized_mask[230:560,0:700])
        print("Image {} processed".format(file))
        #color = (0, 0, 255)
        #cv2.putText(bgr_road_image, "Testing", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        #cv2.imshow(file, bgr_road_image)
        #k = cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #if k == 27: 
        #    cv2.destroyAllWindows()
        #    break
###################################################
