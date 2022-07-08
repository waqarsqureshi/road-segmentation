from collections import namedtuple

'''
id          name                color
0      background        [  0,  0,  0]
1           human        [220, 20, 60]
2            pole        [153,153,153]
3            road        [128, 64,128]
4   traffic light        [250,170, 30]
5    traffic sign        [220,220,  0]
6         vehicle        [  0,  0,142]
'''

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'Id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.
    'color'       , # The color of this label
    ] )

labelsRoadsurvey = [
    #       name                     id    color(BGR)
    Label(  'background'            , 0 ,  (  0,  0,  0) ),
    Label(  'human'                 , 1 ,  ( 60, 20,220) ),
    Label(  'pole'                  , 2 ,  (153,153,153) ),
    Label(  'road'                  , 3 ,  (128, 64,128) ),
    Label(  'traffic light'         , 4 ,  ( 30,170,250) ),
    Label(  'traffic sign'          , 5 ,  (  0,220,220) ),
    Label(  'vehicle'               , 6 ,  (142,  0,  0) )
]