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
    "-m", "--mask", type=str, help="path to output  mask directory of images"
)

argparse.add_argument(
    "-mo", "--combined", type=str, help="path to output combined-image directory of images"
)

args = vars(argparse.parse_args())

data_path = os.path.join(os.getcwd(), "{}".format(args["images"]))
out_root = os.path.join(os.getcwd(), "{}".format(args["output"]))
mask_root = os.path.join(os.getcwd(), "{}".format(args["mask"]))
combined_root = os.path.join(os.getcwd(), "{}".format(args["combined"]))


ie = IECore()

net = ie.read_network( model="./model/road-segmentation-adas-0001.xml")
device = "MULTI:CPU,GPU" if "GPU" in ie.available_devices else "CPU"
print(device)
# The GPU is running slower than CPU
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
        
        rgb_image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        #gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        image_h, image_w, _ = orig.shape
        #h,w, r = orig.shape
        # N,C,H,W = batch size, number of channels, height, width
        N, C, H, W = net.input_info[input_layer_ir].tensor_desc.dims
        # OpenCV resize expects the destination size as (width, height)
        resized_image = cv2.resize(orig, (W, H))
        # reshape to network input shape
        input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0) 

        result = exec_net.infer(inputs={input_layer_ir: input_image})
        result_ir = result[output_layer_ir]

        # Prepare data for visualization
        segmentation_mask = np.argmax(result_ir, axis=1)
        # Define colormap, each color represents a class
        colormap = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]])

        # Use function from notebook_utils.py to transform mask to an RGB image
        mask = segmentation_map_to_image(segmentation_mask, colormap)
        resized_mask = cv2.resize(mask, (image_w, image_h))
        road_mask = resized_mask[:, :, 0] #this is the road channel
        #convert back to RGB
        color_road_mask = np.stack((road_mask,)*3,axis=-1)

        # Create image with mask put on
        rgb_masked_image = cv2.bitwise_and(rgb_image, color_road_mask)

        # Define the transparency of the segmentation mask on the photo
        alpha = 0.3
        # Create image after removing the background obtained through masking
        image_with_mask = cv2.addWeighted(resized_mask, alpha, rgb_image, 1 - alpha, 0)
        rgb_masked_image = cv2.cvtColor(rgb_masked_image, cv2.COLOR_RGB2BGR)
        resized_mask_opencv = cv2.cvtColor(resized_mask, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_root, file),rgb_masked_image)
        cv2.imwrite(os.path.join(mask_root, file),resized_mask_opencv)
        cv2.imwrite(os.path.join(combined_root, file),image_with_mask)
         
        #color = (0, 0, 255)
        #cv2.putText(orig, "Testing", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        #cv2.imshow(file, resized_mask)
        #k = cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #if k == 27: 
        #    cv2.destroyAllWindows()
        #    break
###################################################

