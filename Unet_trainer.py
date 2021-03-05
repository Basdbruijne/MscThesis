"""
Unet_trainer 

An example file for how to import data and set up and train a neural network

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow import keras
from tqdm import tqdm
import numpy as np
import cupy as cp
import aotools
import tensorflow as tf
import numpy
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose, concatenate

tf.config.experimental.set_memory_growth

if __name__ == '__main__': 
    """ 
    Import data 
    
    The training data is not uploaded to the repository, so this section will
    need to be adjusted according to the specific training data that is used.
    """
    
    # Set data path and network path
    safe_path = 'Training_Data/set_02_15/'
    network_location = 'NeuralNetworks/set_02_15/'
    
    # Select how many datasets to import
    threads = 84
    thread_start = 0
 
    # Initialize functions for the normalization of the wavefront
    Pupil_func = cp.array(aotools.functions.zernike.zernikeArray(1, 128), 
                          dtype = 'uint8')[0,]
    B = cp.array(aotools.functions.zernike.zernikeArray(3, 128), 
                 dtype = 'float32')[1:].reshape([2, 128**2])
    B_inv = cp.linalg.pinv(B)
    
    x_train = np.zeros([0, 128, 128], dtype='float16')
    y_train = np.zeros([0, 128, 128], dtype='float16')
    
    print('Importing Data')
    try: 
        # Try if there is already an array with corrected data available
        x_train = np.load(safe_path + 'x_train.npy')            
        y_train = np.load(safe_path + 'y_train.npy')
    except:
        for i in tqdm(range(threads)):
            
            # Load data from file
            try:
                data = cp.load(safe_path + 'Data_SH_' 
                               + str(i+thread_start) + '.npy')
                data_y_int = cp.load(safe_path + 'Data_phi_' 
                                     + str(i+thread_start) + '.npy')
            except:
                print('Dataset ' + str(i) + ' not found.')
                continue
            
            # Initialize array for corrected data
            data_x = np.zeros([data.shape[0], 128, 128], dtype='float16')
            data_y = np.zeros([data.shape[0], 128, 128], dtype='float16')
            delete = []
            for j in range(data_y_int.shape[0]):
                
                # Normalize x data
                x_curr = cp.array(data[j,])
                y_curr = cp.array(data_y_int[j,])*Pupil_func
                
                # Skip data if the set is somehow empty or corrupted
                if (not cp.any(x_curr) or not cp.any(y_curr) 
                or cp.any(cp.isnan(x_curr)) or cp.any(cp.isnan(y_curr))):
                    continue
                
                x_curr -= cp.min(x_curr)
                x_curr /= cp.max(x_curr)
                
                if cp.mean(x_curr) > .027:
                    continue
                
                # Remove tip-and tilt modes for existing data
                x = cp.dot(y_curr.reshape([128**2]), B_inv)
                y_curr -= cp.dot(x, B).reshape([128, 128])
                
                # Store corrected data
                data_x[j,] = cp.asnumpy(x_curr)         
                data_y[j,] = cp.asnumpy(y_curr)
                
            # Concatenate data with other data sets
            index = np.any(data_x, axis=(1,2))
            x_train = np.append(x_train, data_x[index,].astype('float16'),
                                axis = 0)
            y_train = np.append(y_train, data_y[index,].astype('float16'),
                                axis = 0)
        
        # Save the corrected data
        np.save(safe_path + 'x_train.npy', x_train)            
        np.save(safe_path + 'y_train.npy', y_train)
        
    # Shuffle the data
    shuffled_index = np.arange(x_train.shape[0])
    np.random.shuffle(shuffled_index)
    x_train = x_train[shuffled_index,]
    y_train = y_train[shuffled_index,]
    
    print("Number of valid datasets is ", x_train.shape[0])
    x_train = x_train.reshape(list(x_train.shape)+[1])

    """ 
    Setup the network 
    
    For a nice overview of the network, look at section 5-2-1 in the thesis"""
    
    inputs = keras.Input((128, 128, 1))
    filter_size = [12, 24, 48, 96, 192]
    activation = 'relu'
    
    def residual_block(N, inp):
        for ii in range(2):
            c = []
            c.append(Conv2D(4, (7, 7), activation=activation,
                            padding='same')(inp))
            c.append(Conv2D(4, (5, 5), activation=activation, 
                            padding='same')(inp))
            c.append(Conv2D(N-8, (3, 3), activation=activation, 
                            padding='same')(inp))
            inp = (concatenate(c))
        return inp
    
    # Setup the input layers
    p = [inputs]
    cin = [inputs]
    for i in range(len(filter_size)):
        cin.append(p[-1])
        cin[-1] = residual_block(filter_size[i], cin[-1])   
        # The last input layer does not have a pooling layer
        if i < len(filter_size)-1: 
            p.append(MaxPooling2D((2, 2))(cin[-1]))
            p[-1] = BatchNormalization()(p[-1])
            cin[-1] = Conv2D(filter_size[i], (3,3), activation=activation, 
                             padding='same')(cin[-1])
    
    # Setup the output layers
    u = [cin[-1]]
    for i in range(len(filter_size)-1):
        u.append(Conv2DTranspose(filter_size[len(filter_size)-2-i], (2, 2), 
                                 strides=(2, 2), padding='same')(u[-1]))
        u[-1] = concatenate([u[-1], cin[len(filter_size)-1-i]])
        u[-1] = residual_block(filter_size[len(filter_size)-2-i], u[-1])
            
    outputs = keras.layers.Conv2D(1, (1, 1), activation='linear')(u[-1])
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    """ Train the network """

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
    model.compile(optimizer=keras.optimizers.Adam(), 
                  loss='mean_absolute_error')
    
    # Start Training
    try:
        for i in range(0, 10):
            history = model.fit(x_train.astype('float16'), 
                                         y_train.astype('float16'), 
                                         batch_size=batch_size, 
                                         epochs=10, 
                                         validation_split=0.01)
            model.save(network_location+'test3'+'_Batch_Size_'
                       +str(batch_size)+'_Loss_'
                       +str(np.round(history.history['loss'][-1],2))
                       +'_Val_Loss_'
                       +str(np.round(history.history['val_loss'][-1],2)))
    except KeyboardInterrupt:
        model.save(network_location+'my_model_keyboard_interrupt')
