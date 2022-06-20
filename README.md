# road-segmentation
road-segmentation using openvino library - install openvino and tensorflow
The semantic-segmentation-roadsurvey takes a set of images from the folder, use openvino semnatic segmentation model to
find different object in the image (total 20) and dilate car and person
The semantic-segmentation-cityscape do the same thing for cityscape dataset and use guassian blur
the utility file is used in road-segmentation.py to get the segmentation mask.
you can download the model from openvino and keep it in the model folder

#Command for changing city scape dataset
python3 ./semantic_segmentation_resize_cityscape.py -i /home/pms -tv train -c aachen

python3 ./semantic_segmentation_resize_cityscape.py -i /home/pms/cityscapes
