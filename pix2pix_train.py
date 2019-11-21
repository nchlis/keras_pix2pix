# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 16:04:13 2018

@author: N.Chlis

based on pix2pix code found in:
https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/
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
from keras.models import load_model

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

def load_split_normalize_img(fname, target_size=(256,int(2*256))):
    '''
    Loads an image corresponding to a path given by the filename (fname).
    Then it splits the image in half, the left part is the satellite and
    the right part corresponds to the map portion of the image.
    Additionally, each image is normalized to 0-1.
    By default resizes to 400x400 to make the image dimensions UNET-friendly
    '''
    from keras.preprocessing.image import load_img, img_to_array
    #img = img_to_array(load_img(fname,color_mode='rgb',target_size=target_size))/255#normalize 8bit image to [0,1]
    img = (img_to_array(load_img(fname,color_mode='rgb',target_size=target_size))-127.5)/127.5#normalize 8bit image to [-1,+1]
    img_sat = img[:,:int(img.shape[1]/2),:]#get satellite image (left half)
    img_map = img[:,int(img.shape[1]/2):,:]#get map image (right half)
    return (img_sat, img_map)

#get the shape of the data by loading a single image
IMG_SHAPE = load_split_normalize_img(fnames_tr[0])[0].shape #(400, 400, 3)

from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.layers import LeakyReLU
from keras.layers import Concatenate
from collections import OrderedDict 

def tanh_to_sigmoid_range(x):
    '''
    converts an array x from [-1,1] to [0,1]
    useful for plotting images that have already
    been normalized to the [-1,1] range
    '''
    return((x+1)/2)
    
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

#%% initialize the generator
#gen_train = aug_generator(fnames_tr,batch_size=10,flip_axes=['x','y'])
#
#
##plot the augmentations
#X_batch, Y_batch = next(gen_train)
#Nbatch=len(X_batch)

#%% print a few example images
#fig, axes = plt.subplots(Nbatch,2,figsize=(2*6,Nbatch*6))
#for i in range(Nbatch):
#    print(i)
#    axes[i,0].imshow(tanh_to_sigmoid_range(X_batch[i,:,:,:]))
#    axes[i,1].imshow(tanh_to_sigmoid_range(Y_batch[i,:,:,:]))

#%%
#print(model.summary())
#plot_model(model, to_file='unet_model.png', show_shapes=True, show_layer_names=True)

#%%
# define the discriminator model
def get_discriminator(image_shape,resize_factor=1.0):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=image_shape)
	# target image input
	in_target_image = Input(shape=image_shape)
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	# C64
	d = Conv2D(int(64*resize_factor), (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(int(128*resize_factor), (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(int(256*resize_factor), (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(int(512*resize_factor), (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(int(512*resize_factor), (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model
 

 
# define the standalone generator model
def get_generator(image_shape=None,resize_factor=1.0):

    # define an encoder block
    def encoder_block(layer_in, n_filters, batchnorm=True):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # add downsampling layer
        g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
        # conditionally add batch normalization
        if batchnorm:
            g = BatchNormalization()(g, training=True)
        # leaky relu activation
        g = LeakyReLU(alpha=0.2)(g)
        return g
    
    # define a decoder block
    def decoder_block(layer_in, skip_in, n_filters, dropout=True):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # add upsampling layer
        g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
        # add batch normalization
        g = BatchNormalization()(g, training=True)
        # conditionally add dropout
        if dropout:
            g = Dropout(0.5)(g, training=True)
        # merge with skip connection
        g = Concatenate()([g, skip_in])
        # relu activation
        g = Activation('relu')(g)
        return g
    
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)#256x256
    # encoder model
    e1 = encoder_block(in_image, int(64*resize_factor), batchnorm=False)#128x128
    e2 = encoder_block(e1, int(128*resize_factor))#64x64
    e3 = encoder_block(e2, int(256*resize_factor))#32x32
    e4 = encoder_block(e3, int(512*resize_factor))#16x16
    e5 = encoder_block(e4, int(512*resize_factor))#8x8
    e6 = encoder_block(e5, int(512*resize_factor))#4x4
    e7 = encoder_block(e6, int(512*resize_factor))#2x2
    # bottleneck, no batch norm and relu
    b = Conv2D(int(512/resize_factor), (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)#1x1
    b = Activation('relu')(b)
    # decoder model
    d1 = decoder_block(b, e7, int(512*resize_factor))
    d2 = decoder_block(d1, e6, int(512*resize_factor))
    d3 = decoder_block(d2, e5, int(512*resize_factor))
    d4 = decoder_block(d3, e4, int(512*resize_factor), dropout=False)
    d5 = decoder_block(d4, e3, int(256*resize_factor), dropout=False)
    d6 = decoder_block(d5, e2, int(128*resize_factor), dropout=False)
    d7 = decoder_block(d6, e1, int(64*resize_factor), dropout=False)
    # output
    g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model
 
# define the combined generator and discriminator model, for updating the generator
def get_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model

#%% setup the models

#resize_factor_gen  = 2.0
#resize_factor_dis  = 2.0

#resize_factor_gen  = 1.5
#resize_factor_dis  = 1.5

#resize_factor_gen  = 1.0
#resize_factor_dis  = 1.0
    
resize_factor_gen  = 0.5
resize_factor_dis  = 0.5

#resize_factor_gen  = 0.25
#resize_factor_dis  = 0.25

#%% set up model training parameters

batch_size=1#define the batch size
#batch_size=16#define the batch size
#batch_size=32#define the batch size

n_epochs=200#total number of epochs to train for
save_epochs=50#save a model instance periodically every 'save_epochs'
early_stopping=False
max_patience = n_epochs#epochs without improvement to wait, only matters if early_stopping==True
resume_training=False#resume training an existing model

#generator to get training data
gen_train = aug_generator(fnames_tr,batch_size=batch_size,flip_axes=['x','y'],rotation_angles=[5,15])

#generator to get validation data
gen_val = aug_generator(fnames_val,batch_size=batch_size,flip_axes=['x','y'],rotation_angles=[5,15])

# calculate the number of batches per training epoch
batches_per_epoch = int(len(fnames_tr) / batch_size)
batches_per_epoch_val = int(len(fnames_val) / batch_size)

#filepath to save the model
filepath = 'pix2pix_RF'+str(np.round(resize_factor_gen,3))+'_disRF'+str(np.round(resize_factor_dis,3))+'_batchSize'+str(batch_size)+'_earlyStopping'+str(early_stopping)

#%% train the model

#for testing purposes
#n_epochs=5
#batches_per_epoch=10
#print('WARNING, LEFT MANUAL NUMBER OF EPOCHS AND/OR BATCHES PER EPOCH')

if resume_training == True:
    print('resuming training')
    df=pd.read_csv('./trained_models/'+filepath+'.csv')
    epoch_start=df['epoch'].values[-1]
    epoch_end=epoch_start+n_epochs
    #create lists to keep statistics
    list_epoch=df['epoch'].values.tolist()
    list_epoch_duration=df['epoch_duration'].values.tolist()#in seconds
    #training statistics
    list_dis_loss_real=df['dis_loss_real'].values.tolist()
    list_dis_loss_fake=df['dis_loss_fake'].values.tolist()
    list_dis_loss_total=df['dis_loss_total'].values.tolist()
    list_gan_loss_bce=df['gan_loss_bce'].values.tolist()
    list_gan_loss_mae=df['gan_loss_mae'].values.tolist()
    list_gan_loss_total=df['gan_loss_total'].values.tolist()
    #validation statistics
    list_val_gan_loss_bce=df['gan_loss_bce'].values.tolist()
    list_val_gan_loss_mae=df['val_gan_loss_mae'].values.tolist()
    list_val_gan_loss_total=df['val_gan_loss_total'].values.tolist()
    #initialize the best validation loss
    best_val_loss = df['val_gan_loss_mae'].min()
    #load the models
    gen_model = get_generator(image_shape=IMG_SHAPE, resize_factor=resize_factor_gen)
    dis_model = get_discriminator(image_shape=IMG_SHAPE, resize_factor=resize_factor_dis)
    gen_model.load_weights('./trained_models/'+'gen_'+filepath+'_last.hdf5')
    dis_model.load_weights('./trained_models/'+'dis_'+filepath+'_last.hdf5')
    gan_model = get_gan(g_model=gen_model, d_model=dis_model, image_shape=IMG_SHAPE)
    # unfreeze the discriminator
    dis_model.trainable=True
    dis_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
    
else:
    print('training the model from scratch')
    epoch_start=0
    epoch_end=n_epochs
    #create lists to keep statistics
    list_epoch=[]
    list_epoch_duration=[]#in seconds
    #training statistics
    list_dis_loss_real=[]
    list_dis_loss_fake=[]
    list_dis_loss_total=[]
    list_gan_loss_bce=[]
    list_gan_loss_mae=[]
    list_gan_loss_total=[]
    #validation statistics
    list_val_gan_loss_bce=[]
    list_val_gan_loss_mae=[]
    list_val_gan_loss_total=[]
    #initialize the best validation loss
    best_val_loss = np.Inf
    
    gen_model = get_generator(image_shape=IMG_SHAPE, resize_factor=resize_factor_gen)
    dis_model = get_discriminator(image_shape=IMG_SHAPE, resize_factor=resize_factor_dis)
    gan_model = get_gan(g_model=gen_model, d_model=dis_model, image_shape=IMG_SHAPE)
    # unfreeze
    dis_model.trainable=True
    dis_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])

print(gen_model.summary())
print(dis_model.summary())
print(gan_model.summary())

# determine the output square shape of the discriminator
patch_shape = dis_model.output_shape[1]
#get real targets for the discriminator
T_batch_real = np.ones((batch_size, patch_shape, patch_shape, 1))
#get fake targets for the discriminator
T_batch_fake = np.zeros((batch_size, patch_shape, patch_shape, 1))
#initialize the patience parameter
patience=10
#actually start training for a fixed number of epochs
#only the model instances that improve validation performance are saved to disk
for e in np.arange(epoch_start,epoch_end):
    start=time.time()
    print()#print an empty line for readability
    print('epoch',e+1,'of',epoch_start+n_epochs)
    
    #initialize epoch statistics for the training set
    dis_loss_real=np.zeros(batches_per_epoch)
    dis_loss_fake=np.zeros(batches_per_epoch)
    gan_loss_total=np.zeros(batches_per_epoch)
    gan_loss_bce=np.zeros(batches_per_epoch)
    gan_loss_mae=np.zeros(batches_per_epoch)
    
    #train on the training set
    for b in range(batches_per_epoch): 
        #print('--batch',b+1,'of',batches_per_epoch)
        #get true image pairs and labels
        X_batch, Y_batch = next(gen_train)
        #if the dimensions don't fit, redo the targets
        #the batch size of the generator might vary if it does not divide perfectly the total number of datapoints
        #that is, if len(fnames_tr) % batch_size != 0
        if(T_batch_real.shape[0])!=(X_batch.shape[0]):
            #print('T_batch_real.shape:',T_batch_real.shape)
            #print('X_batch.shape:',X_batch.shape)
            #get real targets for the discriminator
            T_batch_real = np.ones((X_batch.shape[0], patch_shape, patch_shape, 1))
            #get fake targets for the discriminator
            T_batch_fake = np.zeros((X_batch.shape[0], patch_shape, patch_shape, 1))
        #train discriminator on true samples
        dis_loss_real[b] = dis_model.train_on_batch([X_batch, Y_batch], T_batch_real)
        
        #get fake image pairs and labels
        #X_batch_fake = gen_model.predict(X_batch)
        Y_batch_fake = gen_model.predict(X_batch)
        #train discriminator on fake samples
        #dis_loss_fake[b] = dis_model.train_on_batch([X_batch_fake, Y_batch], T_batch_fake)
        dis_loss_fake[b] = dis_model.train_on_batch([X_batch, Y_batch_fake], T_batch_fake)
        
        #train the gan
        #Note: the generator is only indirectly trained through the gan
        gan_loss_total[b], gan_loss_bce[b], gan_loss_mae[b] = gan_model.train_on_batch(X_batch, [T_batch_real, Y_batch])
    
    #initialize epoch statistics for the validation set
    val_gan_loss_bce=np.zeros(batches_per_epoch_val)
    val_gan_loss_mae=np.zeros(batches_per_epoch_val)
    val_gan_loss_total=np.zeros(batches_per_epoch_val)
    
    #print('validating...')
    #evaluate on the validation set
    for b in range(batches_per_epoch_val):
        X_batch, Y_batch = next(gen_val)
        #if the dimensions don't fit, redo the targets
        #the batch size of the generator might vary if it does not divide perfectly the total number of datapoints
        #that is, if len(fnames_tr) % batch_size != 0
        if(T_batch_real.shape[0])!=(X_batch.shape[0]):
            #print('T_batch_real.shape:',T_batch_real.shape)
            #print('X_batch.shape:',X_batch.shape)
            #get real targets for the discriminator
            T_batch_real = np.ones((X_batch.shape[0], patch_shape, patch_shape, 1))
        val_gan_loss_total[b], val_gan_loss_bce[b], val_gan_loss_mae[b] = gan_model.test_on_batch(X_batch, [T_batch_real, Y_batch])
        
    end=time.time()
    #keep statistics
    #training statistics
    list_epoch.append(e+1)
    list_epoch_duration.append(end-start)
    list_dis_loss_real.append(dis_loss_real.mean())
    list_dis_loss_fake.append(dis_loss_fake.mean())
    list_dis_loss_total.append(dis_loss_real.mean()+dis_loss_fake.mean())
    list_gan_loss_total.append(gan_loss_total.mean())
    list_gan_loss_bce.append(gan_loss_bce.mean())
    list_gan_loss_mae.append(gan_loss_mae.mean())
    #validation statistics    
    list_val_gan_loss_bce.append(val_gan_loss_bce.mean())
    list_val_gan_loss_mae.append(val_gan_loss_mae.mean())
    list_val_gan_loss_total.append(val_gan_loss_total.mean())
    
    #save model if it improves validation set performance
    #val_loss = val_gan_loss_total.mean()
    val_loss = val_gan_loss_mae.mean()
    print('done in',str(np.round(end-start,2))+'s')
    #print(str(np.round(end-start,2))+'s, end of epoch',e+1)
    if val_loss < best_val_loss:
        print('val_loss_mae improved from ',np.round(best_val_loss,3),'to',np.round(val_loss,3))
        best_val_loss = val_loss
        print('saving generator to '+'./trained_models/'+'gen_'+filepath+'.hdf5')
        gen_model.save('./trained_models/'+'gen_'+filepath+'_bestVal.hdf5')
        patience=0#reset patience since valid loss improved
    else:
        patience=patience+1
    
    #print statistics for this epoch
    print('loss_mae:',gan_loss_mae.mean(),'val_loss:',val_loss)
    
    #% save training statistics
    od = OrderedDict()
    od['epoch']=list_epoch
    od['epoch_duration']=list_epoch_duration
    od['dis_loss_real']=list_dis_loss_real
    od['dis_loss_fake']=list_dis_loss_fake
    od['dis_loss_total']=list_dis_loss_total
    od['gan_loss_bce']=list_gan_loss_bce
    od['gan_loss_mae']=list_gan_loss_mae
    od['gan_loss_total']=list_gan_loss_total
    od['val_gan_loss_bce']=list_val_gan_loss_bce
    od['val_gan_loss_mae']=list_val_gan_loss_mae
    od['val_gan_loss_total']=list_val_gan_loss_total
    df = pd.DataFrame(od, columns=od.keys())
    df.to_csv('./trained_models/'+filepath+'.csv',index=False)
    
    if (e % save_epochs == 0) and (e>0):
        #save backup of generator
        gen_model.save('./trained_models/'+'gen_'+filepath+'_ep'+str(e)+'.hdf5')
    else:
        #save last model to allow training to resume
        #save the generator
        gen_model.save('./trained_models/'+'gen_'+filepath+'_last.hdf5')
        #save the discriminator
        dis_model.save('./trained_models/'+'dis_'+filepath+'_last.hdf5')
    
    if early_stopping == True:
        #stop if patience limit is exceeded
        if patience > max_patience:
            print('Run out of patience!')
            break


