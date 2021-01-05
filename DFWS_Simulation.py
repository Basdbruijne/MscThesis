# DFWS_Simulation
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import DFWS_Simulator as sim
import DFWS_Solver as solver
from DFWS_Solver import plot
import aotools
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
from tensorflow import keras
import numpy as cp
import numpy as np
import time
from tqdm import tqdm
from scipy.ndimage import rotate
# plt.close('all')
#%%
print('Downloading Network')
network_location = 'NeuralNetworks/Unet_set_01_03/'
model = tf.keras.models.load_model(network_location+'Batch_Size_32_Loss_0.7_Val_Loss_0.77')

obj = cp.load('Object_Main.npy')
#%%
print('Make Setup')
setup = sim.DFWS(8, 6, 680, 680, 0, 0, .6)

#%% Choose source of phase screen

# Wavefront from disk:
# setup.wavefront_from_disk(np.random.rand()*2*np.pi, factor = 1, screen = 2)#int(np.around(np.random.rand())+1))

# Wavefront from Kolmogorov
setup.wavefront_kolmogorov(1)

# Wavefront from Zernike
# zCoeffs = np.random.rand(32)-.5
# zCoeffs[0:3] = 0
# setup.wavefront_from_zernike(zCoeffs)
setup.make_psf(no_SH = False, no_main = False, no_main_wavefront = False)

#%% Load object and make image
setup.load_object(obj)
setup.make_image()
setup.wavefront_0 = setup.wavefront
setup.psf_0 = setup.psf
setup.psf_sh_0 = setup.psf_sh

#%% Estimate the wavefront
solver.get_wavefront_modal(setup)
solver.get_wavefront_zonal(setup)
solver.get_wavefront_DLWFS(setup, model)

#%% Deconvolve Image
time0 = time.time()
setup, objectt = solver.deconvolve(setup, mode = 'LR', iterations=50)
time1 = time.time()
print('Deconvolution time = ', time1-time0)
plot(objectt)

#%% Make sure the original wavefront is zero mean for good comparison
pupil = np.array(aotools.functions.pupil.circle(340, 680))
setup.wavefront_0 *= pupil
setup.wavefront_0 -= np.mean(setup.wavefront_0)
setup.wavefront_0 *= pupil

#%% Calculate wavefront reconstrution error
RMSE = np.zeros([4])
RMSE[0] = np.sqrt(np.mean((setup.wavefront_0)**2))
print('RMSE wavefront= ', RMSE[0])
RMSE[1] = np.sqrt(np.mean((setup.wavefront-setup.wavefront_0)**2))
print('RMSE wavefront error = ', RMSE[1])
RMSE[2] = np.sqrt(np.mean((setup.wavefront_modal-setup.wavefront_0)**2))
print('RMSE wavefront error modal= ', RMSE[2])
RMSE[3] = np.sqrt(np.mean((setup.wavefront_zonal-setup.wavefront_0)**2))
print('RMSE wavefront error zonal= ', RMSE[3])

#%% Plot the results
solver.plot_wavefront(setup)  
