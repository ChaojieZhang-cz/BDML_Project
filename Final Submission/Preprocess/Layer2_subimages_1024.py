
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import os
import sys
import openslide
import matplotlib
import matplotlib.pyplot as plt

imgage_folder = '/scratch/cz2064/myjupyter/BDML/Project/Data/data/'
New_image_folder = '/scratch/cz2064/myjupyter/BDML/Project/Data/data_layer2_subimages/'
df_path = '/scratch/cz2064/myjupyter/BDML/Project/Data/metadata.tsv'
df = pd.read_csv(df_path, sep='\t')
print('Layer2 Preprocess')


# In[2]:

df


# In[ ]:




# In[ ]:

for i in df.index:
    if i%10 == 0:
        print i,
    file_id = df.loc[i,'File ID']
    file_name = df.loc[i,'File Name']
    sample_image = openslide.open_slide(imgage_folder+file_id+'/'+file_name)
    level_dimensions = sample_image.level_dimensions
    level0_dimensions = sample_image.level_dimensions[0]
    sample_dimensions = sample_image.level_dimensions[1]
    sub_images_path = New_image_folder+file_id
    if not os.path.exists(sub_images_path):
        os.mkdir(sub_images_path)
    
    pixels = 1024
    subsamples_list = []
    for i in range(int(sample_dimensions[0]/pixels)+1):
        for j in range(int(sample_dimensions[1]/pixels)+1):
            if i == int(sample_dimensions[0]/pixels):
                strat = level0_dimensions[0]-pixels*4
            else:
                strat = pixels*4*i
            if j == int(sample_dimensions[1]/pixels):
                end = level0_dimensions[1]-pixels*4
            else:
                end = pixels*4*j
                
            sub_image = sample_image.read_region((strat,end),1,(pixels,pixels))
            sub_image_array = np.array(sub_image)
            subsample_file_name = 'subsample_'+ str(i) + '_' + str(j) +'.jpg'
            save_file = sub_images_path + '/' + subsample_file_name
            matplotlib.image.imsave(save_file, sub_image_array)
            subsamples_list.append(subsample_file_name)
            
    summary_file = sub_images_path + '/' + 'subsamples_list.txt'
    f = open(summary_file,"w")
    information_1 = 'level_dimensions:' + str(level_dimensions) +'\n'
    information_2 = 'Layer 2' +'\n'
    f.write(information_1)
    f.write(information_2)
    for i in subsamples_list:
        i = i + '\n'
        f.write(i)
    f.close()


# In[ ]:




# In[ ]:



