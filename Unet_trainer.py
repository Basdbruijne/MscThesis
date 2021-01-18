"""
Unet_trainer

An example file for how to import data and set up and train a neural network

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow import keras
import numpy as np
import cupy as cp
from tqdm import tqdm
import matplotlib.pyplot as plt
import aotools
import tensorflow as tf
import numpy
import dfws_solver as solver
from dfws_solver import plot 

tf.config.experimental.set_memory_growth

if __name__ == '__main__': 
    #%% IMPORT DATA
    
    # Set data path and network path
    safe_path = 'Training_Data/set_01_17/'
    network_location = 'NeuralNetworks/Unet_set_01_17/'
    
    # Select how many datasets to import
    threads = 59
    thread_start = 0
 
    # Initialize functions for the normalization of the wavefront
    Pupil_func = cp.array(aotools.functions.zernike.zernikeArray(1, 128), dtype = 'uint8')[0,]
    B = cp.array(aotools.functions.zernike.zernikeArray(3, 128), dtype = 'float32')[1:].reshape([2, 128**2])
    B_inv = cp.linalg.pinv(B)
    
    x_train = np.zeros([0, 128, 128], dtype='float16')
    y_train = np.zeros([0, 128, 128], dtype='float16')
    print('Importing Data')
    try: # Try if there is already an array with corrected data available
        x_train = np.load(safe_path + 'x_train.npy')            
        y_train = np.load(safe_path + 'y_train.npy')
    except:
        for i in tqdm(range(threads)):
            
            # Load data from file
            try:
                data = cp.load(safe_path + 'Data_SH_' + str(i+thread_start) + '.npy')
                data_y_int = cp.load(safe_path + 'Data_phi_' + str(i+thread_start) + '.npy')
            except:
                print('Dataset ' + str(i) + ' not found.')
                continue
            
            # Initialize array for corrected data
            data_x = np.zeros([0, 128, 128], dtype='float16')
            data_y = np.zeros([0, 128, 128], dtype='float16')
            delete = []
            for j in range(data_y_int.shape[0]):
                
                # Normalize x data
                x_curr = cp.array(data[j,])
                y_curr = cp.array(data_y_int[j,])*Pupil_func
                
                # Skip data if the set is somehow empty or corrupted
                if not cp.any(x_curr) or not cp.any(y_curr) or cp.any(cp.isnan(x_curr)) or cp.any(cp.isnan(y_curr)):
                    continue
                
                x_curr -= cp.min(x_curr)
                x_curr /= cp.max(x_curr)
                
                if np.mean(x_curr) > .11:
                    continue
                
                # Remove tip-and tilt modes for existing data
                x = cp.dot(y_curr.reshape([128**2]), B_inv)
                y_curr -= cp.dot(x, B).reshape([128, 128])
                
                # Store corrected data
                data_x = np.append(data_x, cp.asnumpy(x_curr.reshape([1, 128, 128])), axis=0)               
                data_y = np.append(data_y, cp.asnumpy(y_curr.reshape([1, 128, 128])), axis=0)
                
            # Concatenate data with other data sets
            x_train = np.append(x_train, data_x.astype('float16'), axis = 0)
            y_train = np.append(y_train, data_y.astype('float16'), axis = 0)
            
        np.save(safe_path + 'x_train.npy', x_train)            
        np.save(safe_path + 'y_train.npy', y_train)
        
    # Shuffle the data
    shuffled_index = np.arange(x_train.shape[0])
    np.random.shuffle(shuffled_index)
    x_train = x_train[shuffled_index,]
    y_train = y_train[shuffled_index,]
    
    print("Number of valid datasets is ", x_train.shape[0])
    x_train = x_train.reshape(list(x_train.shape)+[1])

    #%% SETUP NETWORK
    inputs = keras.Input((128, 128, 1))
    filter_size = [12, 24, 48, 96, 192]
    
    c1 = keras.layers.Conv2D(filter_size[0], (7, 7), activation='relu', padding='same')(inputs)
    c1 = keras.layers.Conv2D(filter_size[0], (3, 3), activation='relu', padding='same')(c1)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)
    c1 = keras.layers.Conv2D(filter_size[0], (3, 3), activation='relu', padding='same')(c1)
    
    c2 = keras.layers.BatchNormalization()(p1)
    c2 = keras.layers.Conv2D(filter_size[1], (5, 5), activation='relu', padding='same')(c2)
    c2 = keras.layers.Conv2D(filter_size[1], (3, 3), activation='relu', padding='same')(c2)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)
    c2 = keras.layers.Conv2D(filter_size[1], (3, 3), activation='relu', padding='same')(c2)
    
    c3 = keras.layers.Conv2D(filter_size[2], (3, 3), activation='relu', padding='same')(p2)
    c3 = keras.layers.Conv2D(filter_size[2], (3, 3), activation='relu', padding='same')(c3)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)
    c3 = keras.layers.Conv2D(filter_size[2], (3, 3), activation='relu', padding='same')(c3)
    
    c4 = keras.layers.Conv2D(filter_size[3], (3, 3), activation='relu', padding='same')(p3)
    c4 = keras.layers.Conv2D(filter_size[3], (3, 3), activation='relu', padding='same')(c4)
    p4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
    c4 = keras.layers.Conv2D(filter_size[3], (3, 3), activation='relu', padding='same')(c4)
    
    c5 = keras.layers.BatchNormalization()(p4)
    c5 = keras.layers.Conv2D(filter_size[4], (3, 3), activation='relu', padding='same')(c5)
    c5 = keras.layers.Conv2D(filter_size[4], (3, 3), activation='relu', padding='same')(c5)
    
    u6 = keras.layers.Conv2DTranspose(filter_size[3], (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = keras.layers.concatenate([u6, c4])
    c6 = keras.layers.Conv2D(filter_size[3], (3, 3), activation='relu', padding='same')(u6)
    c6 = keras.layers.Conv2D(filter_size[3], (3, 3), activation='relu', padding='same')(c6)
    
    u7 = keras.layers.Conv2DTranspose(filter_size[2], (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = keras.layers.concatenate([u7, c3])
    c7 = keras.layers.Conv2D(filter_size[2], (3, 3), activation='relu', padding='same')(u7)
    c7 = keras.layers.Conv2D(filter_size[2], (3, 3), activation='relu', padding='same')(c7)
    
    u8 = keras.layers.Conv2DTranspose(filter_size[1], (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = keras.layers.concatenate([u8, c2])
    c8 = keras.layers.Conv2D(filter_size[1], (3, 3), activation='relu', padding='same')(u8)
    c8 = keras.layers.Conv2D(filter_size[1], (3, 3), activation='relu', padding='same')(c8)
    
    u9 = keras.layers.Conv2DTranspose(filter_size[0], (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = keras.layers.concatenate([u9, c1])
    c9 = keras.layers.Conv2D(filter_size[0], (3, 3), activation='relu', padding='same')(u9)
    c9 = keras.layers.Conv2D(filter_size[0], (3, 3), activation='relu', padding='same')(c9)
    
    outputs = keras.layers.Conv2D(1, (1, 1), activation='linear')(c9)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    #%% TRAINING

    # Delete all variables stored in GPU memory
    keras.backend.clear_session()
    for variable in dir(): 
        try:
            globals()[variable].device
            del globals()[variable]
        except:
            pass
    
    # Initialize training
    batch_size = 32
    model.compile(optimizer=keras.optimizers.Adam(), loss='mean_absolute_error')
    
    # Start Training
    try:
        for i in range(0, 6):
            training_history = model.fit(x_train.astype('float16'), y_train.astype('float16'), batch_size=batch_size, epochs=10, validation_split=0.01)
            model.save(network_location+'Batch_Size_'+str(batch_size)+'_Loss_'+str(np.round(training_history.history['loss'][-1],2))+'_Val_Loss_'+str(np.round(training_history.history['val_loss'][-1],2)))
    except KeyboardInterrupt:
        model.save(network_location+'my_model_keyboard_interrupt')
