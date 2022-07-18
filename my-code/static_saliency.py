# Packages Import
import argparse
import cv2
import os


# We first build the argument parser, then parse args...
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Path to input image")
parser.add_argument("-o", "--save_to", required=True, help="Path to save image excluding the saliencyType")
parser.add_argument("-d", "--diff", required=True, help="Differentiator")
parser.add_argument("-a", "--here", required=False, help="Ignores SaveTo and uses CWD to save output")
args = vars(parser.parse_args())

if args["here"] == "True":
    args["save_to"] = os.getcwd() + "/output/"



# Load our input image
img = cv2.imread(args["image"])
di = args["diff"]

# Initialise OpenCV's static saliency SPECTRAL RESIDUAL DETECTOR and compute saliency map
#saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
#(success, saliencyMap) = saliency.computeSaliency(img)
#saliencyMap = (saliencyMap * 255).astype("uint8")
#cv2.imshow("Image", img)
#cv2.imshow("Output", saliencyMap)
#cv2.imwrite(args["save_to"] + "/{}_lowfi_image_saliency.png".format(di), saliencyMap)
#cv2.waitKey(0)

# Initialise the more fine-grained saliency detector and compute the saliencyMap
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(img)
print(saliencyMap)
clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
cl1 = clahe.apply((saliencyMap * 255).astype("uint8"))
cl2 = 255 - cl1

# If we want a *binary* map to use for contour processing,
# computing convex hulls, extract bounding boxes, etc... we can 
# additionally threshold the saliency map.

#threshMap = cv2.threshold((saliencyMap * 255).astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Show Images
cv2.imshow("Image", img)
cv2.imshow("Output", cl2)
cv2.imshow("SaliencyMap", saliencyMap)
#cv2.imshow("Thresholds", threshMap)
#impath = args["save_to"] + "/{}_image_original.png".format(di)
salpath = args["save_to"] + "/{}_image_saliency.png".format(di)
threshpath = args["save_to"] + "/{}_image_thresh.png".format(di)
#cv2.imwrite(impath, img)
cv2.imwrite(salpath, (saliencyMap * 255).astype("uint8"))
cv2.imwrite(threshpath, cl2)
cv2.waitKey(0)
