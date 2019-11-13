# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:24:34 2019

@author: Nikos
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array
import pandas as pd

#%% find the path for all images

#the data is pre-split into training and validation
#but we'll further split the validation set into 50/50% valid-test later

path = './data/maps/train/'
fnames = os.listdir(path)
fnames_tr = [path+f for f in fnames]

path = './data/maps/val/'
fnames = os.listdir(path)
fnames_val = [path+f for f in fnames]

fnames_val, fnames_ts = train_test_split(fnames_val, test_size=0.5, random_state=1)

#save filenames to disk
path = './data/'

pd.Series(fnames_tr).to_csv(path+'fnames_tr.csv',index=False,header=None)
pd.Series(fnames_val).to_csv(path+'fnames_val.csv',index=False,header=None)
pd.Series(fnames_ts).to_csv(path+'fnames_ts.csv',index=False,header=None)

#%% load image filenames from disk

path = './data/'

fnames_tr = [f[0] for f in pd.read_csv(path+'fnames_tr.csv',header=None).values.tolist()]#need to unpack list of lists
fnames_val = [f[0] for f in pd.read_csv(path+'fnames_val.csv',header=None).values.tolist()]#need to unpack list of lists
fnames_ts = [f[0] for f in pd.read_csv(path+'fnames_ts.csv',header=None).values.tolist()]#need to unpack list of lists


#%% plot a single image as a sanity check

fname = fnames_tr[0]
img = img_to_array(load_img(fname,color_mode='rgb'))/255#normalize 8bit image to 0-1

print(img.shape)#(600, 1200, 3), height x width x channels

img_sat = img[:,:int(img.shape[1]/2),:]
img_map = img[:,int(img.shape[1]/2):,:]

alpha=0.8
fig, axes = plt.subplots(2,2,figsize=(10,10))

ax=axes[0,0]
ax.set_title('Satellite Image')
ax.set_xlim(0,600)
ax.set_ylim(0,600)
ax.set_xticks(np.arange(0,601,100))
ax.set_yticks(np.arange(0,601,100))
ax.imshow(img_sat)

ax=axes[0,1]
ax.set_title('Map Image')
ax.set_xlim(0,600)
ax.set_ylim(0,600)
ax.set_xticks(np.arange(0,601,100))
ax.set_yticks(np.arange(0,601,100))
ax.imshow(img_map)

ax=axes[1,0]
ax.set_title('Satellite Image Histogram')
ax.set_ylim(0,150000)
ax.hist(img_sat[:,:,0].flatten(),bins=20,color='red',alpha=alpha)
ax.hist(img_sat[:,:,1].flatten(),bins=20,color='green',alpha=alpha)
ax.hist(img_sat[:,:,2].flatten(),bins=20,color='blue',alpha=alpha)

ax=axes[1,1]
ax.set_title('Map Image Histogram')
ax.set_ylim(0,150000)
ax.hist(img_map[:,:,0].flatten(),bins=20,color='red',alpha=alpha)
ax.hist(img_map[:,:,1].flatten(),bins=20,color='green',alpha=alpha)
ax.hist(img_map[:,:,2].flatten(),bins=20,color='blue',alpha=alpha)

plt.savefig('./figures/example_image.png',dpi=100)

#%% convert the above into a function

def load_split_normalize_img(fname):
    '''
    Loads an image corresponding to a path given by the filename (fname).
    Then it splits the image in half, the left part is the satellite and
    the right part corresponds to the map portion of the image.
    Additionally, each image is normalized to 0-1.
    '''
    from keras.preprocessing.image import load_img, img_to_array
    img = img_to_array(load_img(fname,color_mode='rgb'))/255#normalize 8bit image to 0-1
    img_sat = img[:,:int(img.shape[1]/2),:]#get satellite image (left half)
    img_map = img[:,int(img.shape[1]/2):,:]#get map image (right half)
    return (img_sat, img_map)

#load the same image as above and do a sanity check
fname = fnames_tr[0]
img_sat, img_map = load_split_normalize_img(fname)

alpha=0.8
fig, axes = plt.subplots(2,2,figsize=(10,10))

ax=axes[0,0]
ax.set_title('Satellite Image')
ax.set_xlim(0,600)
ax.set_xticks(np.arange(0,601,100))
ax.imshow(img_sat)

ax=axes[0,1]
ax.set_title('Map Image')
ax.set_xlim(0,600)
ax.set_xticks(np.arange(0,601,100))
ax.imshow(img_map)

ax=axes[1,0]
ax.set_title('Satellite Image Histogram')
ax.set_ylim(0,150000)
ax.hist(img_sat[:,:,0].flatten(),bins=20,color='red',alpha=alpha)
ax.hist(img_sat[:,:,1].flatten(),bins=20,color='green',alpha=alpha)
ax.hist(img_sat[:,:,2].flatten(),bins=20,color='blue',alpha=alpha)

ax=axes[1,1]
ax.set_title('Map Image Histogram')
ax.set_ylim(0,150000)
ax.hist(img_map[:,:,0].flatten(),bins=20,color='red',alpha=alpha)
ax.hist(img_map[:,:,1].flatten(),bins=20,color='green',alpha=alpha)
ax.hist(img_map[:,:,2].flatten(),bins=20,color='blue',alpha=alpha)

#%% load the data
#
## We will leave this part out since the dataset is large.
## We will only load images on-the-fly using the filenames 
#
#S_tr = np.zeros((len(fnames_tr),600,600,3))#satellite
#M_tr = np.zeros((len(fnames_tr),600,600,3))#map
#
#for i in np.arange(len(fnames_tr)):
#    print('loading image',i+1,'of',len(fnames_tr))
#    fname = fnames_tr[i]
#    img = img_to_array(load_img(fname,color_mode='rgb'))/255#normalize 8bit image to 0-1
#    S_tr[i,:,:,:] = img[:,:int(img.shape[1]/2),:]
#    M_tr[i,:,:,:] = img[:,int(img.shape[1]/2):,:]
#
#S_val = np.zeros((len(fnames_val),600,600,3))#satellite
#M_val = np.zeros((len(fnames_val),600,600,3))#map
#
#for i in np.arange(len(fnames_val)):
#    print('loading image',i+1,'of',len(fnames_val))
#    fname = fnames_val[i]
#    img = img_to_array(load_img(fname,color_mode='rgb'))/255#normalize 8bit image to 0-1
#    S_val[i,:,:,:] = img[:,:int(img.shape[1]/2),:]
#    M_val[i,:,:,:] = img[:,int(img.shape[1]/2):,:]
#
#S_ts = np.zeros((len(fnames_ts),600,600,3))#satellite
#M_ts = np.zeros((len(fnames_ts),600,600,3))#map
#
#for i in np.arange(len(fnames_ts)):
#    print('loading image',i+1,'of',len(fnames_ts))
#    fname = fnames_ts[i]
#    img = img_to_array(load_img(fname,color_mode='rgb'))/255#normalize 8bit image to 0-1
#    S_ts[i,:,:,:] = img[:,:int(img.shape[1]/2),:]
#    M_ts[i,:,:,:] = img[:,int(img.shape[1]/2):,:]
#
##% save numpy arrays to disk
#
#np.save('./data/S_tr.npy',S_tr)
#np.save('./data/M_tr.npy',M_tr)
#
#np.save('./data/S_val.npy',S_val)
#np.save('./data/M_val.npy',M_val)
#
#np.save('./data/S_ts.npy',S_ts)
#np.save('./data/M_ts.npy',M_ts)

#%%






















