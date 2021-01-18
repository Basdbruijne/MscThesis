"""
DFWS Simulation

An example file of how to interact with the DFWS Simulator and the DFWS Solver

"""
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


# Disable GPU:
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Close all figures
plt.close('all')
#%% Import neural network from file
print('Downloading Network')
network_location = 'NeuralNetworks/Unet_set_01_15/'
model_Bas = tf.keras.models.load_model(network_location+'Batch_Size_32_Loss_0.47_Val_Loss_0.51')


#%% Initialize DFWS class
print('Make Setup')
setup = sim.DFWS(16, 8, 680, 680, 0, 0, .6)

# Choose source of phase screen
Wavefront_source = 'Kolmogorov' #, 'Zernike' or 'Tubrulence Simulator'

if Wavefront_source == 'Tubrulence Simulator':
    setup.wavefront_from_disk(np.random.rand()*2*np.pi, factor = 1, screen = 2)#int(np.around(np.random.rand())+1))
elif Wavefront_source == 'Kolmogorov':
    setup.wavefront_kolmogorov(1)
elif Wavefront_source == 'Zernike':
    zCoeffs = np.random.rand(32)-.5
    zCoeffs[0:3] = 0
    setup.wavefront_from_zernike(zCoeffs)

# Make setup file
setup.make_psf(no_SH = False, no_main = False, no_main_wavefront = False)

#%% Load object and make image
# Load object from file:
# obj = cp.load('Object_Main.npy')
# im = imageio.imread('ISS.png')

setup.random_object()
setup.make_image()

# store original wavefronts and PSF's
setup.wavefront_0 = setup.wavefront
setup.psf_0 = setup.psf
setup.psf_sh_0 = setup.psf_sh

#%% Estimate the wavefront

time0 = time.time()
solver.get_wavefront_DLWFS(setup, model_Bas, test = True, tip_iterations = 3)
setup.remove_ptt('wavefront_0')
print('RMSE wavefront error DLWFS Bas= ', solver.rmse(setup))
time1 = time.time()
print('Prediction time = ', time1-time0)

# Plot the results of the TIP-algorithm
plot(solver.convert_680_to_128(setup, setup.psf_sh_0))
plot(setup.psf_est)

#%% Deconvolve Image
time0 = time.time()
setup, objectt = solver.deconvolve(setup, mode = 'LR', iterations=50)
setup.object_estimate = objectt
solver.plot_object(setup)
time1 = time.time()
print('Deconvolution time = ', time1-time0)


#%% Make sure the original wavefront is zero mean for good comparison
pupil = np.array(aotools.functions.pupil.circle(340, 680))
setup.wavefront_0 *= pupil
setup.wavefront_0 -= np.mean(setup.wavefront_0)
setup.wavefront_0 *= pupil

#%% Plot the results
solver.plot_wavefront(setup)

#%% Collect data over range of turbulence strengths
if False:
    RMSE = np.zeros([18, 5, 50])
    for i in tqdm(range(18)):
        setup = sim.DFWS(i, 6, 680, 680, 0, 0, .6)
        for ii in tqdm(range(50)):
            setup.wavefront_kolmogorov(1)
            setup.make_psf(no_SH = False, no_main = False, no_main_wavefront = False)
            setup.random_object()
            setup.make_image()
            setup.wavefront_0 = setup.wavefront
            setup.remove_ptt('wavefront_0')
            solver.get_wavefront_DLWFS(setup, model_Bas, test = True, tip_iterations = 3)
            RMSE[i,0, ii] = solver.rmse(setup)
            solver.get_wavefront_DLWFS(setup, model_Hu, test = True, tip_iterations = 3)
            RMSE[i,1, ii] = solver.rmse(setup)
            solver.get_wavefront_DLWFS(setup, model_Bekendam, test = True, tip_iterations = 3)
            RMSE[i,2, ii] = solver.rmse(setup)
            solver.get_wavefront_zonal(setup)
            RMSE[i,3, ii] = solver.rmse(setup)
            solver.get_wavefront_modal(setup)
            RMSE[i,4, ii] = solver.rmse(setup)     

#%% Plot RMSE results from file
if False:
    RMSE = np.load('RMSE.npy')
    RMSE = RMSE[:,[0, 3, 1, 4, 2],]
    names = ['de Bruijne', 'Zonal', 'Hu', 'Modal', 'Bekendam']
    solver.plot_rms_bar(RMSE, names)
    
#%% Temporary Code



