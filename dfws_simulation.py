"""
DFWS Simulation

An example file of how to interact with the DFWS Simulator and the DFWS Solver

"""
import os
import DFWS_Simulator as sim
import dfws_solver as solver
from dfws_solver import plot
import aotools
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
import numpy as np
import time
from tqdm import tqdm

# Close all figures
plt.close('all')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#%% Import neural network from file
print('Downloading Network')
network_location = 'NeuralNetworks/Unet_set_01_12/'
model = tf.keras.models.load_model(network_location+'Batch_Size_32_Loss_0.51_Val_Loss_0.51')


#%% Initialize DFWS class
print('Make Setup')
setup = sim.DFWS(10, 6, 680, 680, True, 0, .6)
# Choose source of phase screen

setup.wavefront_kolmogorov(1)
setup.make_psf(no_SH = False, no_main = False, no_main_wavefront = False)
setup.random_object()
setup.make_image()

# store original wavefronts and PSF's
setup.wavefront_0 = setup.wavefront
setup.psf_0 = setup.psf
setup.psf_sh_0 = setup.psf_sh

#%% Estimate the wavefront
solver.get_wavefront_DLWFS(setup, model, test = True, tip_iterations = 3)
setup.remove_ptt('wavefront_0')

# Plot the results of the TIP-algorithm
plot(solver.convert_680_to_128(setup, setup.psf_sh_0))
plot(setup.psf_est)

#%% Deconvolve Image
setup, objectt = solver.deconvolve(setup, mode = 'LR', iterations=25)
setup.object_estimate = objectt
solver.plot_object(setup)

#%% Plot the results
solver.plot_wavefront(setup)


