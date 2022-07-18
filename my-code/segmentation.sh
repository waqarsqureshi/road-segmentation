python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/input/train/1 -o /home/pms/pms-dataset/data-set/output/1 -c /home/pms/pms-dataset/data-set/crop/1 -mo /home/pms/pms-dataset/data-set/combined/1 

python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/input/train/2 -o /home/pms/pms-dataset/data-set/output/2 -c /home/pms/pms-dataset/data-set/crop/2 -mo /home/pms/pms-dataset/data-set/combined/2 

python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/input/train/3 -o /home/pms/pms-dataset/data-set/output/3 -c /home/pms/pms-dataset/data-set/crop/3 -mo /home/pms/pms-dataset/data-set/combined/3

python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/input/train/4 -o /home/pms/pms-dataset/data-set/output/4 -c /home/pms/pms-dataset/data-set/crop/4 -mo /home/pms/pms-dataset/data-set/combined/4

python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/input/train/5 -o /home/pms/pms-dataset/data-set/output/5 -c /home/pms/pms-dataset/data-set/crop/5 -mo /home/pms/pms-dataset/data-set/combined/5 

python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/input/train/6 -o /home/pms/pms-dataset/data-set/output/6 -c /home/pms/pms-dataset/data-set/crop/6 -mo /home/pms/pms-dataset/data-set/combined/6 

python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/input/train/7 -o /home/pms/pms-dataset/data-set/output/7 -c /home/pms/pms-dataset/data-set/crop/7 -mo /home/pms/pms-dataset/data-set/combined/7 

python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/input/train/8 -o /home/pms/pms-dataset/data-set/output/8 -c /home/pms/pms-dataset/data-set/crop/8 -mo /home/pms/pms-dataset/data-set/combined/8 

python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/input/train/9 -o /home/pms/pms-dataset/data-set/output/9 -c /home/pms/pms-dataset/data-set/crop/9 -mo /home/pms/pms-dataset/data-set/combined/9 

python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/input/train/10 -o/home/pms/pms-dataset/data-set/output/10 -c /home/pms/pms-dataset/data-set/crop/10 -mo /home/pms/pms-dataset/data-set/combined/10

python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/ -tv train -c 1
python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/ -tv train -c 2
python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/ -tv train -c 3
python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/ -tv train -c 4
python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/ -tv train -c 5
python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/ -tv train -c 6
python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/ -tv train -c 7
python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/ -tv train -c 8
python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/ -tv train -c 9
python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/ -tv train -c 10

python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/ -tv val -c 1
python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/ -tv val -c 2
python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/ -tv val -c 3
python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/ -tv val -c 4
python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/ -tv val -c 5
python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/ -tv val -c 6
python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/ -tv val -c 7
python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/ -tv val -c 8
python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/ -tv val -c 9
python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/data-set/ -tv val -c 10

python3 ./semantic_segmentation.py -i /home/pms/pms-dataset/stretch-dataset -tv test -c "2 A"

python3 ./semantic_segmentation_conv.py -i /home/pms/mmsegmentation/data/roadsurvey/gTruth -tv train -c 3-new

python3 ./semantic_segmentation_conv.py -i /home/pms/test -tv mask -c 1

python3 ./semantic_segmentation_test_countour.py -i /home/pms/pms-dataset/data-set-10class -tv train -c 2

#Command for changing city scape dataset
python3 ./semantic_segmentation_resize_cityscape.py -i /home/pms -tv train -c aachen
#or
python3 ./semantic_segmentation_resize_cityscape.py -i /home/pms/cityscapes

#command for blurring the dataset
python3 ./semantic_segmentation_roadsurvey.py -i /home/pms/-tv RoadSections -c Urban


          