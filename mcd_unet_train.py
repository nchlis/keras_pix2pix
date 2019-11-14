# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 16:04:13 2018

@author: N.Chlis
"""
#if used on a non-GUI server ######
#import matplotlib
#matplotlib.use('Agg')
###################################

import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
#import h5py

from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dense
#from keras.layers import Flatten
from keras.layers import AveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Conv2DTranspose
from keras.layers.merge import concatenate #Concatenate (capital C) not working 
#from keras.utils.vis_utils import plot_model
from keras.layers import Dropout

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
import time
import skimage.transform

def rotateT(X,angle):
    #rotate image tensor, TF order, single channel
    X_rot = np.zeros_like(X)
    #repeat for every channel
    for ch in np.arange(X.shape[-1]):
        #print('channel',ch)
        #repeat for every image
        for i in np.arange(X.shape[0]):
            #print('image',i)
            X_rot[i,:,:,ch] = skimage.transform.rotate(X[i,:,:,ch],angle=angle,resize=False,preserve_range=True,mode='edge')
    return(X_rot)

def shiftT(X,dx,dy):
    #rotate image tensor, TF order, single channel
    X_shift = np.zeros_like(X)
    #repeat for every image
    tform = skimage.transform.SimilarityTransform(translation=(dx, dy))
    for i in np.arange(X.shape[0]):
        #print('image',i)
        X_shift[i,:,:,:] = skimage.transform.warp(X[i,:,:,:],tform,mode='edge')
    return(X_shift)

#% load image filenames from disk
import pandas as pd

path = './data/'

fnames_tr = [f[0] for f in pd.read_csv(path+'fnames_tr.csv',header=None).values.tolist()]#need to unpack list of lists
fnames_val = [f[0] for f in pd.read_csv(path+'fnames_val.csv',header=None).values.tolist()]#need to unpack list of lists
fnames_ts = [f[0] for f in pd.read_csv(path+'fnames_ts.csv',header=None).values.tolist()]#need to unpack list of lists

##get the shape of the data by loading a single image
#IMG_SHAPE = list(img_to_array(load_img(fnames_tr[0],color_mode='rgb',target_size=(400,400))).shape)#convert to list to change width
#IMG_SHAPE[1]=int(IMG_SHAPE[1]/2)#calculate shape after splitting to satellite and map images (halve the width)
#IMG_SHAPE = tuple(IMG_SHAPE)#switch the shape back to a tuple

def load_split_normalize_img(fname, target_size=(400,int(2*400))):
    '''
    Loads an image corresponding to a path given by the filename (fname).
    Then it splits the image in half, the left part is the satellite and
    the right part corresponds to the map portion of the image.
    Additionally, each image is normalized to 0-1.
    By default resizes to 400x400 to make the image dimensions UNET-friendly
    '''
    from keras.preprocessing.image import load_img, img_to_array
    img = img_to_array(load_img(fname,color_mode='rgb',target_size=target_size))/255#normalize 8bit image to 0-1
    img_sat = img[:,:int(img.shape[1]/2),:]#get satellite image (left half)
    img_map = img[:,int(img.shape[1]/2):,:]#get map image (right half)
    return (img_sat, img_map)

#get the shape of the data by loading a single image
IMG_SHAPE = load_split_normalize_img(fnames_tr[0])[0].shape #(400, 400, 3)

#%%
def aug_generator(fnames=None,
                  batch_size=4,
                  flip_axes=['x','y'],
                  rotation_angles=[5,15]):
                  #noise_gaussian_mean=0,
                  #noise_gaussian_var=1e-2):
                  #noise_snp_amount=0.05):
    '''
    Loads a number of batch_size images from the disk according to the path in fnames.
    It then splits the image into its satellite and map parts, normalizes to 0-1,
    applies augmentation and returns the batch matrices.
    '''
    batch_size=batch_size#recommended batch size    
    Ndatapoints = len(fnames)
    #Naugmentations=4 #original + flip, rotation, noise_gaussian, noise_snp
    
    while(True):
        #print('start!')
        ix_randomized = np.random.choice(Ndatapoints,size=Ndatapoints,replace=False)
        ix_batches = np.array_split(ix_randomized,int(Ndatapoints/batch_size))
        
        for b in range(len(ix_batches)):
            #print('step',b,'of',len(ix_batches))
            ix_batch = ix_batches[b]
            current_batch_size=len(ix_batch)
            #print('size of current batch',current_batch_size)
            #print(ix_batch)
            
                    #initialize batch matrices
            batch_shape = (current_batch_size,)+IMG_SHAPE
            X_batch = np.zeros(batch_shape)
            Y_batch = np.zeros(batch_shape)
#            X_batch = X_raw[ix_batch,:,:,:].copy()#.copy() to leave original unchanged
#            Y_batch = Y_raw[ix_batch,:,:,:].copy()#.copy() to leave original unchanged
            
            #now do augmentation on images and masks
            #iterate over each image in the batch
            for img in range(current_batch_size):
                #print('current_image',img,': ',ix_batch[img])
                #print(ix_batch)
                #load the images X_batch = satellite image, Y_batch = map image
                X_batch[img,:,:,:], Y_batch[img,:,:,:] = load_split_normalize_img(fnames[ix_batch[img]]) 
                
                do_aug=np.random.choice([True, False],size=1)[0]#50-50 chance
                if do_aug == True:
                    #print('flipping',img)
                    flip_axis_selected = np.random.choice(flip_axes,1,replace=False)[0]
                    if flip_axis_selected == 'x':
                        flip_axis_selected = 1
                    else: # 'y'
                        flip_axis_selected = 0
                    #flip an axis
                    X_batch[img,:,:,:] = np.flip(X_batch[img,:,:,:],axis=flip_axis_selected)
                    Y_batch[img,:,:,:] = np.flip(Y_batch[img,:,:,:],axis=flip_axis_selected)
                    #print('Flip on axis',flip_axis_selected)
                
                do_aug=np.random.choice([True, False],size=1)[0]#50-50 chance
                if do_aug == True:
                    #print('rotating',img)
                    rotation_angle_selected = np.random.uniform(low=rotation_angles[0],high=rotation_angles[1],size=1)[0]
                    #rotate the image
                    X_batch[img,:,:,:] = rotateT(np.expand_dims(X_batch[img,:,:,:],axis=0),angle=rotation_angle_selected)
                    Y_batch[img,:,:,:] = rotateT(np.expand_dims(Y_batch[img,:,:,:],axis=0),angle=rotation_angle_selected)
                    #print('Rotate angle',rotation_angle_selected)
            yield(X_batch,Y_batch)
            #print('step end after',b,'of',len(ix_batches))

#%% sanity check

#X_batch[0,:,:,:], Y_batch[0,:,:,:] = load_split_normalize_img(fnames_tr[0])            
#fig, axes = plt.subplots(2,2,figsize=(10,10))
#ax=axes[0,0]
#ax.imshow(X_batch[0,:,:,:])
#ax=axes[0,1]
#ax.imshow(Y_batch[0,:,:,:])
#ax=axes[1,0]
#ax.imshow(X_batch[1,:,:,:])
#ax=axes[1,1]
#ax.imshow(Y_batch[1,:,:,:])
#plt.show()


#%% initialize the generator
gen_train = aug_generator(fnames_tr,batch_size=10,flip_axes=['x','y'])


#plot the augmentations
X_batch, Y_batch = next(gen_train)
Nbatch=len(X_batch)

#%
fig, axes = plt.subplots(Nbatch,2,figsize=(2*6,Nbatch*6))
for i in range(Nbatch):
    print(i)
    axes[i,0].imshow(X_batch[i,:,:,:],cmap='gray')
    axes[i,1].imshow(Y_batch[i,:,:,:],cmap='gray')
    
#%% set-up the UNET model

#model parameters
bnorm_axis = -1
#filter sizes of the original model
nfilters = np.array([64, 128, 256, 512, 1024])
drop_rate=0.0
drop_train=True
#if the dropout rate is zero, there's no dropout
if drop_rate==0.0:
    drop_train=False

#downsize the UNET for this example.
#the smaller network is faster to train
#and produces excellent results on the dataset at hand
nfilters = (nfilters/8).astype('int')

#input
input_tensor = Input(shape=IMG_SHAPE, name='input_tensor')

####################################
# encoder (contracting path)
####################################
#encoder block 0
e0 = Conv2D(filters=nfilters[0], kernel_size=(3,3), padding='same')(input_tensor)
e0 = BatchNormalization(axis=bnorm_axis)(e0)
e0 = Activation('relu')(e0)
e0 = Conv2D(filters=nfilters[0], kernel_size=(3,3), padding='same')(e0)
e0 = BatchNormalization(axis=bnorm_axis)(e0)
e0 = Activation('relu')(e0)

#encoder block 1
e1 = MaxPooling2D((2, 2))(e0)
e1 = Conv2D(filters=nfilters[1], kernel_size=(3,3), padding='same')(e1)
e1 = BatchNormalization(axis=bnorm_axis)(e1)
e1 = Activation('relu')(e1)
e1 = Conv2D(filters=nfilters[1], kernel_size=(3,3), padding='same')(e1)
e1 = BatchNormalization(axis=bnorm_axis)(e1)
e1 = Activation('relu')(e1)

#encoder block 2
e2 = Dropout(drop_rate)(e1, training = drop_train)
e2 = MaxPooling2D((2, 2))(e2)
e2 = Conv2D(filters=nfilters[2], kernel_size=(3,3), padding='same')(e2)
e2 = BatchNormalization(axis=bnorm_axis)(e2)
e2 = Activation('relu')(e2)
e2 = Conv2D(filters=nfilters[2], kernel_size=(3,3), padding='same')(e2)
e2 = BatchNormalization(axis=bnorm_axis)(e2)
e2 = Activation('relu')(e2)

#encoder block 3
e3 = Dropout(drop_rate)(e2, training = drop_train)
e3 = MaxPooling2D((2, 2))(e3)
e3 = Conv2D(filters=nfilters[3], kernel_size=(3,3), padding='same')(e3)
e3 = BatchNormalization(axis=bnorm_axis)(e3)
e3 = Activation('relu')(e3)
e3 = Conv2D(filters=nfilters[3], kernel_size=(3,3), padding='same')(e3)
e3 = BatchNormalization(axis=bnorm_axis)(e3)
e3 = Activation('relu')(e3)

#encoder block 4
e4 = Dropout(drop_rate)(e3, training = drop_train)
e4 = MaxPooling2D((2, 2))(e4)
e4 = Conv2D(filters=nfilters[4], kernel_size=(3,3), padding='same')(e4)
e4 = BatchNormalization(axis=bnorm_axis)(e4)
e4 = Activation('relu')(e4)
e4 = Conv2D(filters=nfilters[4], kernel_size=(3,3), padding='same')(e4)
e4 = BatchNormalization(axis=bnorm_axis)(e4)
e4 = Activation('relu')(e4)
#e4 = MaxPooling2D((2, 2))(e4)

####################################
# decoder (expansive path)
####################################

#decoder block 3
d3 = Dropout(drop_rate)(e4, training = drop_train)
d3=UpSampling2D((2, 2),)(d3)
d3=concatenate([e3,d3], axis=-1)#skip connection
d3=Conv2DTranspose(nfilters[3], (3, 3), padding='same')(d3)
d3=BatchNormalization(axis=bnorm_axis)(d3)
d3=Activation('relu')(d3)
d3=Conv2DTranspose(nfilters[3], (3, 3), padding='same')(d3)
d3=BatchNormalization(axis=bnorm_axis)(d3)
d3=Activation('relu')(d3)

#decoder block 2
d2 = Dropout(drop_rate)(d3, training = drop_train)
d2=UpSampling2D((2, 2),)(d2)
d2=concatenate([e2,d2], axis=-1)#skip connection
d2=Conv2DTranspose(nfilters[2], (3, 3), padding='same')(d2)
d2=BatchNormalization(axis=bnorm_axis)(d2)
d2=Activation('relu')(d2)
d2=Conv2DTranspose(nfilters[2], (3, 3), padding='same')(d2)
d2=BatchNormalization(axis=bnorm_axis)(d2)
d2=Activation('relu')(d2)

#decoder block 1
d1=UpSampling2D((2, 2),)(d2)
d1=concatenate([e1,d1], axis=-1)#skip connection
d1=Conv2DTranspose(nfilters[1], (3, 3), padding='same')(d1)
d1=BatchNormalization(axis=bnorm_axis)(d1)
d1=Activation('relu')(d1)
d1=Conv2DTranspose(nfilters[1], (3, 3), padding='same')(d1)
d1=BatchNormalization(axis=bnorm_axis)(d1)
d1=Activation('relu')(d1)

#decoder block 0
d0=UpSampling2D((2, 2),)(d1)
d0=concatenate([e0,d0], axis=-1)#skip connection
d0=Conv2DTranspose(nfilters[0], (3, 3), padding='same')(d0)
d0=BatchNormalization(axis=bnorm_axis)(d0)
d0=Activation('relu')(d0)
d0=Conv2DTranspose(nfilters[0], (3, 3), padding='same')(d0)
d0=BatchNormalization(axis=bnorm_axis)(d0)
d0=Activation('relu')(d0)

#output
#out_class = Dense(1)(d0)
out_class = Conv2D(3, (1, 1), padding='same')(d0)#RGB output
out_class = Activation('sigmoid',name='output')(out_class)

#create and compile the model
model=Model(inputs=input_tensor,outputs=out_class)
#model.compile(loss={'output':'binary_crossentropy'},
#              metrics={'output':'accuracy'},
#              optimizer='adam')
model.compile(loss={'output':'mae'}, optimizer='adam')

#%%
print(model.summary())
#plot_model(model, to_file='unet_model.png', show_shapes=True, show_layer_names=True)

#%% train the model
#filepath = 'mcd_unet_31M_drop'+str(drop_rate)
#filepath = 'mcd_unet_31M_drop'+str(drop_rate)+'_mae'
#filepath = 'mcd_unet_div8_495K_drop'+str(drop_rate)
filepath = 'mcd_unet_div8_495K_drop'+str(drop_rate)+'_mae'

#save the model when val_loss improves during training
checkpoint = ModelCheckpoint('./trained_models/'+filepath+'.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
#save training progress in a .csv
csvlog = CSVLogger('./trained_models/'+filepath+'_train_log.csv',append=True)
#stop training if no improvement has been seen on val_loss for a while
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20)
batch_size=4

#initialize the training generator
gen_train = aug_generator(fnames_tr,batch_size=10,flip_axes=['x','y'],rotation_angles=[5,15])
#split the array and see how many splits there are to determine #steps
steps_per_epoch_tr = len(np.array_split(np.zeros(len(fnames_tr)),int(len(fnames_tr)/batch_size)))

#initialize the training generator
gen_val = aug_generator(fnames_val,batch_size=10,flip_axes=['x','y'],rotation_angles=[5,15])
#split the array and see how many splits there are to determine #steps
steps_per_epoch_val = len(np.array_split(np.zeros(len(fnames_val)),int(len(fnames_val)/batch_size)))


#actually do the training
model.fit_generator(generator=gen_train,
                    steps_per_epoch=steps_per_epoch_tr,#the generator internally goes over the entire dataset in one iteration
                    validation_data=gen_val,
                    validation_steps=steps_per_epoch_val,
                    epochs=200,
                    verbose=2,
                    initial_epoch=0,
                    callbacks=[checkpoint, csvlog, early_stopping])







