import numpy as np
import openslide
from openslide.deepzoom import DeepZoomGenerator
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.image as im
import pandas as pd
import sys
import datetime

# Read in metadata info

dir_path = '/scratch/nc2433/myjupyter/'
# metadata_path = 'sub_metadata/sub_metadata' + sys.argv[1] + '.csv'
metadata_path = 'demo_metadata.csv'

metadata = pd.read_csv(dir_path + metadata_path)

# preprocess images

# determine whether a slice should be kept
def white_ratio(image):
    width = image.shape[0]
    height = image.shape[1]
    image_reshape = image.reshape(width*height,image.shape[2])
    image_reshape_unique = np.unique(image_reshape,axis=0)
    return len(image_reshape_unique)/len(image_reshape)

# dimension for slices
# pixels = 1024
pixels = 224
# tiles_dir = dir_path + 'sliced_tiles/tiles' + sys.argv[1] +'/'
tiles_dir = dir_path + 'sliced_tiles/demo_tiles/'

for k in range(len(metadata)):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S  ') + 'Start slicing ' + str(k) + '-th image...')
#     file_id = metadata.loc[k]['File ID']
#     file_name = metadata.loc[k]['File Name']
    file_id = metadata.iloc[k]['File ID']
    file_name = metadata.iloc[k]['File Name']
    img_path = dir_path + 'data/' + file_id + '/' + file_name
    
    slide = openslide.open_slide(img_path)
    
    data_gen = DeepZoomGenerator(slide, tile_size=pixels, overlap=0, limit_bounds=False)

    
    [w,h] = slide.dimensions
    num_h = int(np.ceil(h/pixels))
    num_w = int(np.ceil(w/pixels))
    level = data_gen.level_count - 1

    for i in range(num_h):
        for j in range(num_w):
            img = np.array(data_gen.get_tile(level, (j,i)))
            if white_ratio(img) > 0.1:
                seq_num = str(i) + '_' + str(j)
                im.imsave(tiles_dir + file_id + '_' + seq_num + '.png', img)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S  ') + 'End slicing ' + str(k) + '-th image.')
                
