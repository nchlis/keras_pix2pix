# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:33:31 2019

@author: Nikos
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import pandas as pd

from keras.models import load_model

path = './data/'
fnames_ts = [f[0] for f in pd.read_csv(path+'fnames_ts.csv',header=None).values.tolist()]#need to unpack list of lists

##get the shape of the data by loading a single image
#IMG_SHAPE = list(img_to_array(load_img(fnames_tr[0],color_mode='rgb',target_size=(400,400))).shape)#convert to list to change width
#IMG_SHAPE[1]=int(IMG_SHAPE[1]/2)#calculate shape after splitting to satellite and map images (halve the width)
#IMG_SHAPE = tuple(IMG_SHAPE)#switch the shape back to a tuple

def load_split_normalize_img(fname, target_size=(256,int(2*256))):
    '''
    Loads an image corresponding to a path given by the filename (fname).
    Then it splits the image in half, the left part is the satellite and
    the right part corresponds to the map portion of the image.
    Additionally, each image is normalized to 0-1.
    By default resizes to 400x400 to make the image dimensions UNET-friendly
    '''
    from keras.preprocessing.image import load_img, img_to_array
#    img = img_to_array(load_img(fname,color_mode='rgb',target_size=target_size))/255#normalize 8bit image to [0,1]
    img = (img_to_array(load_img(fname,color_mode='rgb',target_size=target_size))-127.5)/127.5#normalize 8bit image to [-1,+1]
    img_sat = img[:,:int(img.shape[1]/2),:]#get satellite image (left half)
    img_map = img[:,int(img.shape[1]/2):,:]#get map image (right half)
    return (img_sat, img_map)

def tanh_to_sigmoid_range(x):
    '''
    converts an array x from [-1,1] to [0,1]
    useful for plotting images that have already
    been normalized to the [-1,1] range
    '''
    return((x+1)/2)

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """
    Add a vertical color bar to an image plot.
    https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    """
    from mpl_toolkits import axes_grid1
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

#get the shape of the data by loading a single image
IMG_SHAPE = load_split_normalize_img(fnames_ts[0])[0].shape #(400, 400, 3)

#initialize empty matrices to load images
n_images = 20 #how many test images to load
X_ts = np.zeros((n_images,)+IMG_SHAPE)
Y_ts = np.zeros((n_images,)+IMG_SHAPE)
for i in np.arange(n_images):
    X_ts[i,:,:,:], Y_ts[i,:,:,:] = load_split_normalize_img(fnames_ts[i])

#%%
#modelname = 'gen_pix2pix_RF2.0_disRF2.0_batchSize1_earlyStoppingFalse_last'
#modelname = 'gen_pix2pix_RF1.0_disRF1.0_batchSize1_earlyStoppingFalse_ep50';epoch=50
#modelname = 'gen_pix2pix_RF1.0_disRF1.0_batchSize1_earlyStoppingFalse_ep100';epoch=100
#modelname = 'gen_pix2pix_RF1.0_disRF1.0_batchSize1_earlyStoppingFalse_ep150';epoch=150
#modelname = 'gen_pix2pix_RF1.0_disRF1.0_batchSize1_earlyStoppingFalse_last';epoch=200

#modelname = 'gen_pix2pix_RF1.0_disRF1.0_batchSize16_earlyStoppingFalse_ep50';epoch='50, batch_size=16'
#modelname = 'gen_pix2pix_RF1.0_disRF1.0_batchSize16_earlyStoppingFalse_ep100';epoch='100, batch_size=16'
#modelname = 'gen_pix2pix_RF1.0_disRF1.0_batchSize16_earlyStoppingFalse_ep150';epoch='150, batch_size=16'
#modelname = 'gen_pix2pix_RF1.0_disRF1.0_batchSize16_earlyStoppingFalse_last';epoch='200, batch_size=16'
#modelname = 'gen_pix2pix_RF1.0_disRF1.0_batchSize1_earlyStoppingFalse_bestVal'
    
#modelname = 'gen_pix2pix_RF0.5_disRF0.5_batchSize1_earlyStoppingFalse_ep50';epoch='50, half #filters'
#modelname = 'gen_pix2pix_RF0.5_disRF0.5_batchSize1_earlyStoppingFalse_ep100';epoch='100, half #filters'
#modelname = 'gen_pix2pix_RF0.5_disRF0.5_batchSize1_earlyStoppingFalse_ep150';epoch='150, half #filters'
#modelname = 'gen_pix2pix_RF0.5_disRF0.5_batchSize1_earlyStoppingFalse_last';epoch='200, half #filters'
##
modelname = 'gen_pix2pix_RF2.0_disRF2.0_batchSize1_earlyStoppingFalse_ep50';epoch='50, double #filters'
modelname = 'gen_pix2pix_RF2.0_disRF2.0_batchSize1_earlyStoppingFalse_ep100';epoch='100, double #filters'
modelname = 'gen_pix2pix_RF2.0_disRF2.0_batchSize1_earlyStoppingFalse_ep150';epoch='150, double #filters'
modelname = 'gen_pix2pix_RF2.0_disRF2.0_batchSize1_earlyStoppingFalse_last';epoch='200, double #filters'
#    
#modelname = 'gen_pix2pix_RF0.5_disRF0.5_batchSize1_earlyStoppingFalse_ep100'#a bit before ep 200 something goes wrong

model = load_model('./trained_models/'+modelname+'.hdf5')
Y_ts_hat = model.predict(X_ts)
M=np.abs(tanh_to_sigmoid_range(Y_ts)-tanh_to_sigmoid_range(Y_ts_hat))
mae_pixel=np.mean(M.flatten())#pixel level mae on the test subset

#%% only print a subset of the images and the corresponding epoch, to show training progression

n_images=3
fig, axes = plt.subplots(n_images,4,figsize=(4*4,n_images*4))
idx=[9,10,11]
for i in range(3):
    print(i)
    ax=axes[i,0]
    ax.set_title('Satellite')
    ax.imshow(tanh_to_sigmoid_range(X_ts[idx[i],:,:,:]))
    #ax.set_xticks(np.arange(0,401,100))
    #ax.set_yticks(np.arange(0,401,100))
    
    ax=axes[i,1]
    ax.set_title('Map - True')
    ax.imshow(tanh_to_sigmoid_range(Y_ts[idx[i],:,:,:]))
    #ax.set_xticks(np.arange(0,401,100))
    #ax.set_yticks(np.arange(0,401,100))
    
    ax=axes[i,2]
    ax.set_title('Map - Predicted')
    ax.imshow(tanh_to_sigmoid_range(Y_ts_hat[idx[i],:,:,:]))
    #ax.set_xticks(np.arange(0,401,100))
    #ax.set_yticks(np.arange(0,401,100))
    
    ax=axes[i,3]
    ax.set_title('Map - Error: '+str(np.round(M[idx[i],:,:,:].mean(axis=-1).flatten().mean(),3)))#averaged over channels and pixels
    im = ax.imshow(M[idx[i],:,:,:].mean(axis=-1),cmap='gray',vmin=0,vmax=1)
    #ax.set_xticks(np.arange(0,401,100))
    #ax.set_yticks(np.arange(0,401,100))
    add_colorbar(im)
    plt.suptitle('Epoch:'+str(epoch),fontsize=40)
plt.savefig('./figures/cropped_'+modelname+'_MAE_'+str(np.round(mae_pixel,3))+'.png',bbox_inches='tight',dpi=100)

#%% specify and evaluate model

#modelname = 'pix2pix_sigmoidOut_genRF'+str(np.round(resize_factor_gen,3))+'_disRF'+str(np.round(resize_factor_dis,3))
#model = load_model('./trained_models/'+modelname+'.hdf5')
#Y_ts_hat = model.predict(X_ts)
#M=np.abs(Y_ts-Y_ts_hat)
#mae_pixel=np.mean(M.flatten())#pixel level mae on the test subset

#%%
##%% evaluate pix2pix loss
#
#modelname = 'pix2pix_genRF0.5_disRF0.5'
#model = load_model('./trained_models/pix2pix_genRF0.5_disRF0.5.hdf5')
#Y_ts_hat = model.predict(X_ts)
#M=np.abs(tanh_to_sigmoid_range(Y_ts)-tanh_to_sigmoid_range(Y_ts_hat))
#mae_pixel=np.mean(M.flatten())#pixel level mae on the test subset
#
##%% evaluate pix2pix loss
#
#modelname = 'pix2pix_genRF1.0_disRF1.0'
#model = load_model('./trained_models/pix2pix_genRF1.0_disRF1.0.hdf5')
#Y_ts_hat = model.predict(X_ts)
#M=np.abs(tanh_to_sigmoid_range(Y_ts)-tanh_to_sigmoid_range(Y_ts_hat))
#mae_pixel=np.mean(M.flatten())#pixel level mae on the test subset
#
##%% evaluate pix2pix loss
#
#modelname = 'pix2pix_genRF1.5_disRF1.5'
#model = load_model('./trained_models/pix2pix_genRF1.5_disRF1.5.hdf5')
#Y_ts_hat = model.predict(X_ts)
#M=np.abs(tanh_to_sigmoid_range(Y_ts)-tanh_to_sigmoid_range(Y_ts_hat))
#mae_pixel=np.mean(M.flatten())#pixel level mae on the test subset
#
##%% evaluate pix2pix loss
#
#modelname = 'pix2pix_genRF2.0_disRF2.0'
#model = load_model('./trained_models/pix2pix_genRF2.0_disRF2.0.hdf5')
#Y_ts_hat = model.predict(X_ts)
#M=np.abs(tanh_to_sigmoid_range(Y_ts)-tanh_to_sigmoid_range(Y_ts_hat))
#mae_pixel=np.mean(M.flatten())#pixel level mae on the test subset

#%% output in [-1,+1]
fig, axes = plt.subplots(n_images,4,figsize=(4*4,n_images*4))
for i in range(n_images):
    print(i)
    ax=axes[i,0]
    ax.set_title('Satellite')
    ax.imshow(tanh_to_sigmoid_range(X_ts[i,:,:,:]))
    #ax.set_xticks(np.arange(0,401,100))
    #ax.set_yticks(np.arange(0,401,100))
    
    ax=axes[i,1]
    ax.set_title('Map - True')
    ax.imshow(tanh_to_sigmoid_range(Y_ts[i,:,:,:]))
    #ax.set_xticks(np.arange(0,401,100))
    #ax.set_yticks(np.arange(0,401,100))
    
    ax=axes[i,2]
    ax.set_title('Map - Predicted')
    ax.imshow(tanh_to_sigmoid_range(Y_ts_hat[i,:,:,:]))
    #ax.set_xticks(np.arange(0,401,100))
    #ax.set_yticks(np.arange(0,401,100))
    
    ax=axes[i,3]
    ax.set_title('Map - Error: '+str(np.round(M[i,:,:,:].mean(axis=-1).flatten().mean(),3)))#averaged over channels and pixels
    im = ax.imshow(M[i,:,:,:].mean(axis=-1),cmap='gray')
    #ax.set_xticks(np.arange(0,401,100))
    #ax.set_yticks(np.arange(0,401,100))
    add_colorbar(im)
    
plt.savefig('./figures/'+modelname+'_MAE_'+str(np.round(mae_pixel,3))+'.png',bbox_inches='tight',dpi=100)

#%% plot loss curves
#based on: https://matplotlib.org/gallery/api/two_scales.html

df = pd.read_csv('./trained_models/pix2pix_RF1.0_disRF1.0_batchSize1_earlyStoppingFalse.csv')
#df = pd.read_csv('./trained_models/pix2pix_RF0.5_disRF0.5_batchSize1_earlyStoppingFalse.csv')
# Create some mock data
t = df.epoch
data1 = df.dis_loss_total
data2 = df.gan_loss_total

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Discriminator loss', color=color)
ax1.plot(t, data1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Generator loss', color=color)  # we already handled the x-label with ax1
ax2.plot(t, data2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('./figures/loss_pix2pix_RF1.0_disRF1.0_batchSize1_earlyStoppingFalse.png',bbox_inches='tight',dpi=100)
plt.show()

#%% plot loss curves, also validation

df = pd.read_csv('./trained_models/pix2pix_RF1.0_disRF1.0_batchSize1_earlyStoppingFalse.csv')
#df = pd.read_csv('./trained_models/pix2pix_RF0.5_disRF0.5_batchSize1_earlyStoppingFalse.csv')
# Create some mock data
t = df.epoch
fig, ax1 = plt.subplots()

color = plt.cm.tab20(0)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Generator total loss')
ax1.plot(t, df.gan_loss_total, color=color, label='G loss-train')
#ax1.tick_params(axis='y', labelcolor=color)
ax1.plot(t, df.val_gan_loss_total, color=plt.cm.tab20(1), label='G loss-valid')
ax1.legend()
#fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('./figures/val_lossTotal_pix2pix_RF1.0_disRF1.0_batchSize1_earlyStoppingFalse.png',bbox_inches='tight',dpi=100)
plt.show()

df = pd.read_csv('./trained_models/pix2pix_RF1.0_disRF1.0_batchSize1_earlyStoppingFalse.csv')
#df = pd.read_csv('./trained_models/pix2pix_RF0.5_disRF0.5_batchSize1_earlyStoppingFalse.csv')
# Create some mock data
t = df.epoch
fig, ax1 = plt.subplots()

color = plt.cm.tab20(0)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Generator MAE loss')
ax1.plot(t, df.gan_loss_mae, color=color, label='G loss-train')
#ax1.tick_params(axis='y', labelcolor=color)
ax1.plot(t, df.val_gan_loss_mae, color=plt.cm.tab20(1), label='G loss-valid')
ax1.legend()
#fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('./figures/val_lossMAE_pix2pix_RF1.0_disRF1.0_batchSize1_earlyStoppingFalse.png',bbox_inches='tight',dpi=100)
plt.show()

df = pd.read_csv('./trained_models/pix2pix_RF1.0_disRF1.0_batchSize1_earlyStoppingFalse.csv')
#df = pd.read_csv('./trained_models/pix2pix_RF0.5_disRF0.5_batchSize1_earlyStoppingFalse.csv')
# Create some mock data
t = df.epoch
fig, ax1 = plt.subplots()

color = plt.cm.tab20(0)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Generator BCE loss')
ax1.plot(t, df.gan_loss_bce, color=color, label='G loss-train')
#ax1.tick_params(axis='y', labelcolor=color)
ax1.plot(t, df.val_gan_loss_bce, color=plt.cm.tab20(1), label='G loss-valid')
ax1.legend()
#fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('./figures/val_lossBCE_pix2pix_RF1.0_disRF1.0_batchSize1_earlyStoppingFalse.png',bbox_inches='tight',dpi=100)
plt.show()

#%% plot loss curves for half #filters model to showcase the peak/problem near the end of training
#based on: https://matplotlib.org/gallery/api/two_scales.html

df = pd.read_csv('./trained_models/pix2pix_RF0.5_disRF0.5_batchSize1_earlyStoppingFalse.csv')
#df = pd.read_csv('./trained_models/pix2pix_RF0.5_disRF0.5_batchSize1_earlyStoppingFalse.csv')
# Create some mock data
t = df.epoch
data1 = df.dis_loss_total
data2 = df.gan_loss_total

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Discriminator loss', color=color)
ax1.plot(t, data1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Generator loss', color=color)  # we already handled the x-label with ax1
ax2.plot(t, data2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('./figures/loss_pix2pix_RF0.5_disRF0.5_batchSize1_earlyStoppingFalse.png',bbox_inches='tight',dpi=100)
plt.show()

