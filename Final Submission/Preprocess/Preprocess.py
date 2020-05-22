import pandas as pd
import numpy as np
import os
import sys
import openslide
import matplotlib
import matplotlib.pyplot as plt


number = sys.argv[1]
print('df',number)


def white_ratio(image):
    width = image.shape[0]
    height = image.shape[1]
    image_reshape = image.reshape(width*height,image.shape[2])
    image_reshape_unique = np.unique(image_reshape,axis=0)
    return len(image_reshape)/len(image_reshape_unique)
    

imgage_folder = '/scratch/cz2064/myjupyter/BDML/Project/Data/data/'
New_image_folder = '/scratch/cz2064/myjupyter/BDML/Project/Data/data_subimages/'
df_path = '/scratch/cz2064/myjupyter/BDML/Project/metadata/Preprocess_sub_dataframe/df' + str(number) +'.csv'
df = pd.read_csv(df_path, sep=',')

for i in df.index:
	print i 
	file_id = df.loc[i,'File ID']
	file_name = df.loc[i,'File Name']
	sample_image = openslide.open_slide(imgage_folder+file_id+'/'+file_name)
	level_dimensions = sample_image.level_dimensions
	sample_dimensions = sample_image.level_dimensions[0]
	sub_images_path = New_image_folder+file_id
	if not os.path.exists(sub_images_path):
		os.mkdir(sub_images_path)
	pixels = 2048
	valid_subsamples_count = 0
	valid_subsamples = []
	for i in range(int(sample_dimensions[0]/pixels)):
		for j in range(int(sample_dimensions[1]/pixels)):
			sub_image = sample_image.read_region((pixels*i,pixels*j),0,(pixels,pixels))
			sub_image_array = np.array(sub_image)
			if white_ratio(sub_image_array) < 100:
				subsample_file_name = 'subsample_'+ str(i) + '_' + str(j) +'.jpg'
				save_file = sub_images_path + '/' + subsample_file_name
				matplotlib.image.imsave(save_file, sub_image_array)
				valid_subsamples_count = valid_subsamples_count + 1
				valid_subsamples.append(subsample_file_name)
	summary_file = sub_images_path + '/' + 'subsamples_list.txt'
	f = open(summary_file,"w")
	information_1 = 'level_dimensions:' + str(level_dimensions) +'\n'
	information_2 = 'Valid samples:'+ str(valid_subsamples_count) +'\n'
	f.write(information_1)
	f.write(information_2)
	for i in valid_subsamples:
		i = i + '\n'
		f.write(i)
	f.close()