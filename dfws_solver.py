"""
Deconvolution from Wavefront Sensing Solver!

Created by Bas de Bruijne, Sep 09, 2020

For questions, contact BasdBruijne@gmail.com

-------------------------------------------------------------------------------------------------------------------------------------------------------
EXAMPLE OF HOW TO USE THIS FILE


import DFWS_Simulator as sim

import dfws_solver as solver

# Fist, use DFWS_Simulator to generate a DFWS class object:

dfws = sim.DFWS(1, 6, 680, 680, 0, 0, .6)   # Initialise system with diameter
 of 1, 6x6 shack-hartmann sensor and 680x680 pixels of imaging resolution

dfws.random_object()                        # Load Object

dfws.wavefront_kolmogorov(0.2)              # Make D/r0 = 5 turbulent phase 
                                             screen

dfws.make_psf()                             # Adjust the wavefront to the right
                                             size and generate the point spread 
                                             functions

dfws.make_image()                           # Generate the output images

# Now, interact with dfws using the solver:
    
solver.get_wavefront_DLWFS(dfws)            # Use my developed deep learning 
                                              wavefront sensing method to 
                                              retrieve the wavefront

objest_est = solver.deconvolve(dfws)        # Estimate object using PSF loaded 
                                             in dfws (solver.get_wavefront_
                                            DLWFS(dfws) loads the estimated 
                                            psf automatically)

-------------------------------------------------------------------------------------------------------------------------------------------------------
"""

import aotools
import matplotlib.pyplot as plt
import numpy
import warnings
import numpy as np
from numpy import fft
from scipy.ndimage import rotate
import cupy as cp
import threading
from copy import copy
import tensorflow as tf
from  aotools.functions.pupil import circle
from aotools.functions.zernike import zernikeArray
cupy = 0

def free_gpu_memory():
    """
    Frees up GPU memory by making reserved blocks available
    
    Inputs:
    
    None, all variables come from globals
    
    Outputs:
    
    None
    """
    if cupy:
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    
def tonumpy(x):
    """
    Returns a variable from either cupy or numpy to numpy
    
    cupy is the CUDA accelerated version of numpy and will
    
    be used by this class if supported by the hardware
    
    Input:
    
    x: either a numpy or cupy array
    
    Output:
    
    x: a numpy array
    """
    try:
        return cp.asnumpy(x)
    except:
        return x
    
def convolve2(a, b):
    """
    Fourier domain convolution between two matrices
    
    Input:
    
    a, b: two square matrices to be convoluted with each other
    
    Output:
    
    c: the result of the convolution between a and b
    """
    # Make sure matrices are square
    if a.shape[0] != a.shape[1] or b.shape[0] != b.shape[1]:
        raise Exception('Please enter square matrices')
    
    # Add padding to the matrices
    a_pad = np.pad(np.array(a), [0, b.shape[0]-1], mode='constant')
    b_pad = np.pad(np.array(b), [0, a.shape[0]-1], mode='constant')
    
    # Convolve the image and crop the edges
    edge = np.minimum(a.shape[0], b.shape[0])/2
    c = np.fft.ifft2(np.fft.fft2(a_pad)
                     *np.fft.fft2(b_pad))[int(np.floor(edge))
                                          :-int(np.ceil(edge))+1, 
                                          int(np.floor(edge))
                                          :-int(np.ceil(edge))+1]      
    return c

def numpy_or_cupy(cupy_req):
    """
    Check if numpy or cupy should be used and load right libraries
    
    Inputs:
    
    cupy_req [bool]: is cupy required?
    
    Outputs:
    
    None
    """
    global np, fft, rotate, cupy, mempool, pinned_mempool
    if cupy_req:
        import cupy as np  
        from cupy import fft
        from cupyx.scipy.ndimage import rotate
        cupy = 1
        mempool = np.get_default_memory_pool()
        pinned_mempool = np.get_default_pinned_memory_pool()
    else:
        import numpy as np 
        from numpy import fft
        from scipy.ndimage import rotate
        cupy = 0
        
def Downres_Image(setup):
    """
    Reduce the resolution of the image, as to be used for the sh-sensor 
    deconvolutoin
    
    TO BE REMOVED IN FUTURE VERSION
    
    Inputs:
    
    Setup: DFWS class
    
    Outputs:
    
    object_est: Downsampled image
    """
    numpy_or_cupy(setup.cupy_req)
    # Downsizes the main image from sensor to the size of the subaperture
    # images
    downres = np.around(np.arange(0, setup.res, 
                                  setup.res/(setup.res_SH
                                             /setup.N))).astype('int')
    object_est = setup.image[downres, :]
    object_est = object_est[:, downres]
    return object_est


def Run_SH_TIP(setup, iterations = 3, o_0 = None, psf_0 = None, 
                    pupil = None, crop_index = None):
    """
    
    Performs the TIP-alogirithm on the Shack-Hartmann image of setup in 
    order to find the Shack-Hartmann PSF
    
    Based on Wilding et all (2017)
    
    Inputs:
    
    o_0 [optional]: initial estimate of object
    
    psf_0 [optional]: initial estimate of PSF
    
    pupil [optional]: mask for the containment of the estimated object
    
    
    
    Outputs:
    
    psf_est: Estimated shack-hartmann point-spread-function
    
    o: Estimated SH object
    """
    
    numpy_or_cupy(setup.cupy_req)
    # free_gpu_memory()
    # If no inital estimate is provided, use circular function with ones
    if o_0 is None:
        object_est = np.array(circle(setup.res_SH/setup.N/5, 
                                     int(np.ceil(setup.res_SH/setup.N)), 
                                     circle_centre=(0, 0), origin='middle'))
    else:
        object_est = o_0
        
    if not psf_0 is None and not o_0 is None:
        warnings.warn("Initial estimates of both object and psf are provided."
                      +" Only psf estimate will be used")

    if not hasattr(setup, 'TIP_pad_o'):
        setup.TIP_pad_o = [int(np.floor((setup.image_sh.shape[0]+1)/2)), 
                           int(np.ceil((setup.image_sh.shape[0]+1)/2))]

    # Initialize function for deconvolution
    o_zero = fft.fftshift((np.pad(np.array(object_est), 
                                  setup.TIP_pad_o, mode='constant')))
    i_F = fft.fft2((np.pad(np.array(setup.image_sh), 
                           [0, int(object_est.shape[0]+1)], mode='constant')))
    o_F = fft.fft2(o_zero)
    # free_gpu_memory()
    
    # Generate tip_pupil if not yet loaded into setup. 
    # This function limits the extent of the estimated object
    if not hasattr(setup, 'tip_pupil'):
        if pupil is None:
            setup.tip_pupil  = fft.fftshift(np.array(circle(setup.res_subap/2, 
                                                            o_F.shape[0], 
                                                            circle_centre=(0, 
                                                                           0), 
                                                            origin='middle')))
            setup.tip_pupil = setup.tip_pupil.astype('float16')
            setup.tip_pupil  += .1*(setup.tip_pupil == 0)
        else:
            setup.tip_pupil = pupil
            
    # Run the TIP-algorithm
    for n in range(0, iterations): 
        if n != 0 or psf_0 is None: # If not initial psf_estimate provided
        
            # Estimate psf by deconvolution
            psf_est_F = i_F/(o_F+(1)*(np.abs(o_F)<(1)))
            psf_est = np.real(fft.ifft2(psf_est_F))     
            psf_est *= (psf_est>0)
            # In all iterations except last, increase the contrast of the image
            if n < iterations-1:
                psf_est -= np.min(psf_est)
                psf_est /= np.max(psf_est)
                psf_est **= (iterations-n)/1
                # psf_est[psf_est < (iterations-n)/10] = 0

            # Normalize the psf and convert to frequency domain
            psf_est -= np.min(psf_est)
            psf_est /= np.sum(psf_est)
            psf_est_F = fft.fft2(psf_est)
            
        else: # If psf_estimate is provided
            psf_0 /= np.sum(psf_0)
            psf_est_F = fft.fft2(np.pad(psf_0, 
                                        [0, int(i_F.shape[0]-psf_0.shape[0])], 
                                        mode='constant'))
        
        # In all iterations exeot last, find object by deconvolution
        if n < iterations-1:
            conj = np.conj(psf_est_F)
            o_F = (conj*i_F)/(conj*psf_est_F+1e-9)
            o = np.abs(fft.ifft2(o_F))
            o *= setup.tip_pupil
            o -= np.min(o)
            o /= np.max(o)
            o_F = fft.fft2(o)
    
    # Crop the estimated PSF and normalize it
    psf_est = np.real(psf_est)
    psf_est = convert_680_to_128(setup, psf_est, crop_index)
    psf_est -= np.min(psf_est)
    psf_est /= np.max(psf_est)
    setup.psf_sh_128 = psf_est 
    return psf_est, fft.fftshift(o)
   
def plot(image, figure = None):
    """
    Show grayscale image, but without the hassle of making sure that 
    the data type and memory is right
    
    Inputs:
    
    image: 2d array representing gray-scale image
    
    Outputs:
    
    None
    """
    if figure is None:
        plt.figure()
    else:
        plt.figure(figure)
    try:
        plt.imshow(image.astype('float32'), cmap='gray')
    except:
        plt.imshow(image.astype('float32').get(), cmap='gray')
        
def plot_phi(phi, figure = None, vmin = None, vmax = None):
    """
    Show wavefront image image, but without the hassle of making sure that 
    the data type and memory is right
    
    Inputs:
    
    image: 2d array representing the wavefront
    
    Outputs:
    
    None
    """
    
    cmap = copy(plt.cm.get_cmap('RdYlGn'))
    cmap.set_bad(alpha = 0)
    image = copy(phi)
    image[image == 0] = np.nan
    
    if figure is None:
        plt.figure()
    else:
        plt.figure(figure)
    try:
        plt.imshow(image.astype('float32'), cmap=cmap, vmin = vmin, 
                   vmax = vmax)
    except:
        plt.imshow(image.astype('float32').get(), cmap=cmap, vmin = vmin, 
                   vmax = vmax)
    plt.axis('off')

def deconvolve_CPU(start, end, iterations):
    # Function used for parallel deconvolution of image, do not call directly!
    global o_F, i_F, psf_F, psf_mirror_F
    tau = 1.5
    for n in range(iterations):
        o_F[start:end,] += tau*((i_F[start:end,]-psf_F[start:end,]
                                 *o_F[start:end,])*psf_mirror_F[start:end,])
        
def deconvolve(setup, mode = 'LR', iterations = 10, 
               regularization_term = 1e-1, force_cuda = False, min_iterations = 20, max_iterations = 80):
    """
    Deconvolves the image based on the loaded psf
    
    Inputs:
    
    mode: mode of deconvolution, options are: LR (lucy richardson), 
    LR2 (accelerated LR), CPU_LR (LR but parallel), Regularization, 
    Regularization_Filter, Regularization_Filter2
    TIP, LR_Steepest_Descent and LR_preconditioning
    
    Iterations: amount of iterations used for the iterative methods, set 
    iterations = 0 to stop when a stopping criterium is met
    
    regularization_term: regularization used for the methods that are use it
    
    force_cuda [bool]: use cuda eventhough cuda is not yet loaded
    
    Outputs:
    
    setup: DFWS class object with estimated object loaded
    
    o_est: Estimated object
     """
    global o_F, i_F, psf_F, psf_mirror_F
    
    # Load numpy or cupy depending on the mode and hardware requirements:
    if mode == 'CPU_LR':
        numpy_or_cupy(0)
    elif force_cuda:
        numpy_or_cupy(1)
    else:
        numpy_or_cupy(setup.cupy_req)
    # free_gpu_memory()
    # Check if the setup class has an image loaded
    if not hasattr(setup, 'image'):
        raise Exception("Setup does not have an image loaded,"
                        + " deconvolution cannot proceed")
    
    # Initialize the variables and make sure the psf is normalized
    setup.psf /= np.sum(setup.psf)
    
    # Define the Fourier transform counterparts of the variables
    pad_i = (setup.psf.shape[0]-1)/2
    pad_psf = setup.image.shape[0]-1
    i_F = fft.fft2((np.pad(np.array(setup.image), [int(np.floor(pad_i)), 
                                                   int(np.ceil(pad_i))], 
                           mode='reflect')))
    psf_F = fft.fft2((np.pad(np.array(setup.psf), [0, pad_psf], 
                             mode='constant')))
    o_F = copy(i_F)
    
    # If the deconvolution mode requires a mirrord psf, generate it
    if 'LR' in mode or mode == 'Steepest_Descent':
        psf_mirror_F = fft.fft2((np.pad(np.array(setup.psf[::-1, ::-1]), 
                                        [pad_psf, 0], mode='constant')))
    
    
    # Lucy richardson: the deconvolution used is actually the landweber method, 
    # which is nearly identical to the lucy richardson deconvolution except 
    # that it is fully in the frequency domain, making it significantly faster
    if mode == 'LR':
        if iterations:
            tau = 1.5
            a = tau*i_F*psf_mirror_F
            b = -1*tau*psf_F*psf_mirror_F
            for n in range(iterations):
                o_F += a + o_F*b
        else:
            tau = 1.5
            a = tau*i_F*psf_mirror_F
            b = -1*tau*psf_F*psf_mirror_F
            for n in range(min_iterations-1):
                o_F += a + o_F*b
            score1 = np.sum((a + o_F*b)**2)
            iterations = min_iterations
            increment = a + o_F*b
            o_F += increment
            score = np.sum((increment)**2)
            while score1*(1+1e-3) > score:
                iterations += 1
                if iterations >= max_iterations:
                    break
                score1 = copy(score)
                increment = a + o_F*b
                o_F += increment
                score = np.sum((increment)**2)
            print(iterations, ' iterations completed')

    # Lucy richardson V2: same as LR, except that the accelation parameter 
    # tau is sceduled
    elif mode == 'LR2':
        tau = 2.5
        a = i_F*psf_mirror_F
        b = -1*psf_F*psf_mirror_F
        for n in range(iterations):
            tau -= 1/iterations
            o_F += tau*(a + o_F*b)

    # Lucy richardson CPU: same as LR, but run parallel on the CPU. This 
    # may be nice if the graphics card does not support cuda
    elif mode == 'CPU_LR':
        threads = 4
        steps = np.around(np.arange(0, i_F.shape[0]+.1, i_F.shape[0]/threads))
        t = []
        for i in range(threads):
            t.append(threading.Thread(target=deconvolve_CPU, 
                                      args=(int(steps[i]),
                                            int(steps[i+1]),iterations)))
            t[i].start()
        
        for i in range(threads):
            t[i].join()
    
    # Regularization method
    elif mode == 'Regularization':
        if iterations != 10:
            print('Regularization mode does not work iteratively')
            
        o_F = i_F/(psf_F+regularization_term*(psf_F<regularization_term))
    
    # Regularization method but with added frequency domain filter
    elif mode == 'Regularization_Filter':
        if iterations != 10:
            print('Regularization mode does not work iteratively')
        
        psf_F = (psf_F+regularization_term*(psf_F<regularization_term))
        o_F = i_F/(psf_F)
        o_F_2 = np.log(1+np.abs(o_F))
        o_F_2 -= np.min(o_F_2)
        o_F_2 /= np.max(o_F_2)
        
        if not hasattr(setup, 'Regularization_Filter'):
            temp = -1*np.array(zernikeArray([4], o_F_2.shape[0]),
                               dtype='float32')[0,]
            temp -= np.min(temp)
            temp /= np.max(temp)
            temp[np.where(temp==temp[0,0])] = 0
            temp[np.where(np.fft.fftshift(np.array(circle(20, 
                                                          o_F_2.shape[0]))))] = 1
            temp **= 2
            setup.Regularization_Filter = temp*.2+.8
            
        o_F = o_F  + (np.mean(o_F) - o_F)*(o_F_2>setup.Regularization_Filter)

    # Regularization method but with a different added frequency domain filter
    elif mode == 'Regularization_Filter2':
        if iterations != 10:
            print('Regularization mode does not work iteratively')
        
        psf_F = (psf_F+regularization_term)#*(psf_F<regularization_term))
        o_F = i_F/(psf_F)
                
        if 1:#not hasattr(setup, 'G1'):
            setup.a1 = np.pad(np.ones([3,3]),[2,2])
            setup.a2 = np.ones([7,7])-setup.a1
            setup.a2 /= np.sum(setup.a2)
            setup.a2 = np.fft.fft2(np.pad(setup.a2, [0, 
                                                     o_F.shape[0]
                                                     -setup.a2.shape[0]]))
            setup.d = np.fft.fftshift(np.array(circle(20, o_F.shape[0])))
            setup.e = (setup.d==0)
            G1_shape = 5
            setup.G1 = np.ones([G1_shape, G1_shape])
            for i in range(G1_shape):
                for j in range(G1_shape):
                    i1 = np.abs(int(np.floor(G1_shape/2))-i)
                    i2 = np.abs(int(np.floor(G1_shape/2))-j)
                    setup.G1[i, j] = 1 * np.exp(-1/4*(i1**2+i2**2))
            setup.G1 = np.fft.fft2(np.pad(setup.G1, 
                                          [0, o_F.shape[0]-setup.G1.shape[0]]))
            setup.threshold = 1.25

        o_F_norm = np.abs(np.fft.fft2(o_F))
        c2 = np.abs(np.fft.ifft2(o_F_norm*setup.a2))
        b = ((c2/np.abs(o_F))>=setup.threshold)
        b = np.fft.ifft2(np.fft.fft2(b)*(setup.G1))
        b = ((1-b*setup.e))
        b *= (b>0)
        o_F *= b
    
    # Tangantial iterative propogations. This method assumes that the PSF 
    # is just an estimate and is able to change it. Since there are not 
    # multiple images available, the performance is not great.
    elif mode == 'TIP':
        pupil = fft.fftshift(np.array(circle(60, o_F.shape[0], 
                                             circle_centre=(0, 0), 
                                             origin='middle'))).astype('float16')
        for n in range(iterations):
        
            o_F = i_F/(psf_F+regularization_term*(psf_F<regularization_term))
            o = np.real(fft.ifft2(o_F))
            o -= np.min(o)
            o = o/np.max(o)
            o_F = fft.fft2(o)
            
            psf_F = i_F/(o_F+regularization_term*(o_F<regularization_term))
            psf = np.abs(fft.ifft2(psf_F))                
            psf /= np.sum(psf)
            psf *= pupil
            psf_F = fft.fft2(psf)
    
    # Steepest descent. Same as LR, but the acceleation parameter is sceduled 
    # based on the direction of steepest descent. Due to the calculation of 
    # norms not actually faster than the normal LR.
    elif mode == 'LR_Steepest_Descent':
        score = np.abs(np.sum((i_F-o_F*psf_F)**2))
        r = i_F - psf_F * o_F
        
        if iterations:
            for n in range(iterations):
                d = psf_mirror_F*r
                w = psf_F*d
                t = np.sqrt(np.sum(d**2))/np.sqrt(np.sum(w**2))
                o_F += t*d
                r -= t*w

        else:
            for n in range(min_iterations):
                d = psf_mirror_F*r
                w = psf_F*d
                t = np.sqrt(np.sum(d**2))/np.sqrt(np.sum(w**2))
                o_F += t*d
                if not n:
                    score = np.mean((t*d)**2)
                r -= t*w
            iterations = min_iterations
            score1 = np.mean((t*d)**2)
            while score1 < score:
                iterations += 1
                if iterations > max_iterations:
                    break
                score = copy(score1)
                d = psf_mirror_F*r
                w = psf_F*d
                t = np.sqrt(np.sum(d**2))/np.sqrt(np.sum(w**2))
                o_F += t*d
                r -= t*w
                score1 = np.mean((t*d)**2)
            print(iterations, ' iterations completed')
    elif mode == 'LR_Hybrid':
        score = np.sqrt(np.sum((i_F-o_F*psf_F)**2))
        r = i_F - psf_F * o_F
        
        for n in range(5):
            d = psf_mirror_F*r
            w = psf_F*d
            t = np.sqrt(np.sum(d**2))/np.sqrt(np.sum(w**2))
            # numpy.linalg.norm(d,2)/numpy.linalg.norm(w,2)
            o_F += t*d
            r -= t*w
        
        tau = 1.5
        a = tau*i_F*psf_mirror_F
        b = -1*tau*psf_F*psf_mirror_F
        if iterations:
            for n in range(iterations-5):
                o_F += a + o_F*b
        else:
            for n in range(10):
                o_F += a + o_F*b
            score1 = np.sqrt(np.sum((i_F-o_F*psf_F)**2)) 
            while score1 < score:
                score = copy(score1)
                o_F += a + o_F*b
                score1 = np.sqrt(np.sum((i_F-o_F*psf_F)**2)) 
            
    # LR with preconditioning. Uses less iterations than LR to converge, 
    # but the iterations take longer. This makes it in practice not actually 
    # faster
    elif mode == 'LR_preconditioning':
        
        tau = 1
        r = i_F - psf_F * o_F
        M = np.diag(np.dot(psf_F.T, psf_F)) + tau * np.tril(np.dot(psf_F.T, 
                                                                   psf_F)) 
        v = np.linalg.inv(M)*r
        d = psf_mirror_F*v
        
        for n in range(iterations):
            o_F += tau*d
            r = i_F - psf_F * o_F
            M = np.diag(np.dot(psf_F.T, psf_F)) + tau * np.tril(np.dot(psf_F.T, 
                                                                       psf_F))
            v = np.linalg.inv(M)*r
            d = psf_mirror_F*v
            
        
    else:
        raise Exception('Mode of deconvolution not recognized, options are: LR'
                        + '(lucy richardson), LR2 (accelerated LR), CPU_LR'
                        +' (LR but parallel), Regularization, Regularization_'
                        + 'Filter, Regularization_Filter2, TIP, LR_Steepest_'
                        +'Descent and LR_preconditioning')
    
    # Calculate the output and normalize it
    output = np.abs(fft.ifft2(o_F))
    output -= np.min(output)
    output /= np.max(output)
    # free_gpu_memory()
    return setup, tonumpy(output[0:setup.image.shape[0], 
                                 0:setup.image.shape[0]]).astype('float16')

def convert_680_to_256(setup, psf_est):
    """
    WILL BE CHANGED TO CONVERT_X_TO_Y IN FUTURE VERSIONS IN ORDER TO 
    COMBINE THESE SIMILAR FUNCTIONS
    
    Reduce black space within SH image to 256 pixels
    
    Input:
    
    psf_est: shack-hartmann patterns, estimated or true
    
    Output:
    
    psf_est: cropped input
    """
    begin = numpy.array(numpy.round((setup.res_SH/setup.N
                                     *numpy.arange(1,setup.N+1)
                                     -setup.res_SH/(setup.N*2))
                                    -256/(setup.N*2)), dtype='int16')
    end = numpy.array(numpy.round((setup.res_SH/setup.N
                                   *numpy.arange(1,setup.N+1)
                                   -setup.res_SH/(setup.N*2))
                                  +256/(setup.N*2)), dtype='int16')
    index = numpy.array(numpy.r_[int(begin[0]):int(end[0])], dtype='int16')
    
    for i in range(1, setup.N):
        index = numpy.r_[index, int(begin[i]):int(end[i])]
    
    index = numpy.r_[index, int(end[setup.N-1]+1)]
        
    index = np.array(index)
    return psf_est[index][:,index]

def convert_680_to_128(setup, psf_est, index = None):
    """
    WILL BE CHANGED TO CONVERT_X_TO_Y IN FUTURE VERSIONS IN ORDER TO 
    COMBINE THESE SIMILAR FUNCTIONS
    
    Reduce black space within SH image to 128 pixels
    
    Input:
    
    psf_est: shack-hartmann patterns, estimated or true
    
    index [optional]: cropping index that overrides the standard cropping. 
                      this can be usefull if the spacing of the SH-sensor is 
                      a little off
    
    Output:
    
    psf_est: cropped input
    """
    if index is None:
        if not hasattr(setup, 'setup.SH_crop_index'):
            if setup.N == 10:
                begin = numpy.array(numpy.round((setup.res_SH/setup.N
                                                 *numpy.arange(1,setup.N+1)
                                                 -setup.res_SH/(setup.N*2))-7),
                                    dtype='int16')
                end = numpy.array(numpy.round((setup.res_SH/setup.N
                                               *numpy.arange(1,setup.N+1)
                                               -setup.res_SH/(setup.N*2))+6),
                                  dtype='int16')
            else:
                begin = numpy.array(numpy.round((setup.res_SH/setup.N
                                                 *numpy.arange(1,setup.N+1)
                                                 -setup.res_SH/(setup.N*2))
                                                -128/(setup.N*2)), dtype='int16')
                end = numpy.array(numpy.round((setup.res_SH/setup.N
                                               *numpy.arange(1,setup.N+1)
                                               -setup.res_SH/(setup.N*2))
                                              +128/(setup.N*2)), dtype='int16')
            index = numpy.array(numpy.r_[int(begin[0]):int(end[0])],
                                dtype='int16')
            
            for i in range(1, setup.N):
                index = numpy.r_[index, int(begin[i]):int(end[i])]
                
            index = np.array(index)
                
            setup.SH_crop_index = index
        else:
            index = setup.SH_crop_index
        out = np.array(psf_est)[index][:,index]
    else:
        psf_est = psf_est[index,][:, index]
        out = convert_256_to_128(setup, psf_est)
    return out

def convert_680_to_128_test(setup, psf_est):
    """
    WILL BE CHANGED TO CONVERT_X_TO_Y IN FUTURE VERSIONS IN ORDER TO 
    COMBINE THESE SIMILAR FUNCTIONS
    
    Reduce black space within SH image to 128 pixels
    
    Rather than cropping, it overlaps the subaperture PSFs, which avoids 
    the PSFs
    
    being cut off. The noise is, however, also overlapped, which can cause 
    problems
    
    Input:
    
    psf_est: shack-hartmann patterns, estimated or true
    
    Output:
    
    psf_est: cropped input
    """
    begin = numpy.array(numpy.round((setup.res_SH/setup.N
                                     *numpy.arange(1,setup.N+1)
                                     -setup.res_SH/(setup.N*2))
                                    -128/(setup.N*2)), dtype='int16')
    end = numpy.array(numpy.round((setup.res_SH/setup.N
                                   *numpy.arange(1,setup.N+1)
                                   -setup.res_SH/(setup.N*2))
                                  +128/(setup.N*2)), dtype='int16')

    border = min(begin[0], 680-end[-1])
    out = numpy.zeros([128+2*border, 128+2*border])
    subap_size = end-begin
    for i in range(setup.N):
        for j in range(setup.N):
            out[np.sum(subap_size[0:i])
                :np.sum(subap_size[0:i+1])
                +2*border, np.sum(subap_size[0:j])
                :np.sum(subap_size[0:j+1])+2*border] += psf_est[begin[i]-border:
                                                                end[i]+border, 
                                                                begin[j]-border
                                                                :end[j]+border]

    return out[border:-border, border:-border]

def convert_256_to_128(setup, psf_est, offset = 0):
    """
    WILL BE CHANGED TO CONVERT_X_TO_Y IN FUTURE VERSIONS IN ORDER TO COMBINE
    THESE SIMILAR FUNCTIONS
    
    Reduce black space within SH image to 128 pixels from 256 pixels
    
    Input:
    
    psf_est: shack-hartmann patterns, estimated or true
    
    Output:
    
    psf_est: cropped input
    """
    res = 256+2*offset
    res_new = 128
    begin = -offset+numpy.array(numpy.floor((res/setup.N
                                             *numpy.arange(1,setup.N+1)
                                             -res/(setup.N*2))
                                            -res_new/(setup.N*2)),dtype='int32')
    end = -offset+numpy.array(numpy.floor((res/setup.N
                                           *numpy.arange(1,setup.N+1)
                                           -res/(setup.N*2))
                                          +res_new/(setup.N*2)), dtype='int32')     
    index = numpy.array(numpy.r_[int(begin[0]):int(end[0])], dtype='int16')
    for i in range(1, setup.N):
        index = numpy.r_[index, int(begin[i]):int(end[i])]
    index = numpy.r_[index, int(end[setup.N-1]+1)]        
    index = np.array(index)
    return psf_est[index][:,index]

def Allign_Shack_Hartmann(setup):
    """
    Allign SH image based on rotation and shift data in 'rotate_shift_data.npz'
    
    Inputs:
    
    setup: DFWS class object
    
    Outputs:
    
    setup: DFWS class object but with alligned shack-hartmann image
    """
    
    numpy_or_cupy(setup.cupy_req)
    # load allignment data from file
    if not hasattr(setup, 'alpha'):
        data = np.load('rotate_shift_data.npz')
        setup.angle = data['name1']
        setup.y0 = data['name2']
        setup.y1 = data['name3']
        setup.x0 = data['name4']
        setup.x1 = data['name5']
    
    # rotate and crop the shack-hartmann image
    setup.image_sh = rotate(setup.image_sh, setup.angle)[setup.y0:setup.y1,
                                                         setup.x0:setup.x1]
    
    return setup

def get_Wavefront_slopes(setup, obj_est = None):
    """
    Returns the slopes of the wavefront estimated from a correlation algorithm
    
    i.e. conventional way of extended scene wavefront sensing
    
    Input:
    
    setup: DFWS class object
    
    obj_est [optional]: estimated object (to be used as reference image)
    
    Output:
    
    Slopes: 3d array containing all the x and y slopes of the 
    """
    # setup the size of the reference image and retrieve the reference image
    border = 20
    if obj_est is None:
        ref = np.zeros([2*border, 2*border])
        # for i in range(2,4):
        #     for j in range(2,4):
        i = int(setup.N/2); j = int(setup.N/2)
        x = int(setup.res/(setup.N*2)+setup.res/setup.N*i)
        y = int(setup.res/(setup.N*2)+setup.res/setup.N*j)
        ref += setup.image_sh[x-border:x+border, y-border:y+border]
        ref /= np.max(ref)
    else:
        ref = obj_est[int((obj_est.shape[0]-2*border)/2)
                      :int((obj_est.shape[0]-2*border)/2)+2*border, 
                      int((obj_est.shape[0]-2*border)/2)
                      :int((obj_est.shape[0]-2*border)/2)+2*border]
    
    # run the cross correlation algorithm in order to find the shifts
    corr = np.zeros([setup.res_SH-2*border, setup.res_SH-2*border])
    for i in (range(setup.res_SH-2*border)):
        for j in range(setup.res_SH-2*border):
            corr[i,j] = np.sum(np.abs(setup.image_sh[i:i+2*border, 
                                                     j:j+2*border]-ref)**2)
     
    # convert the cross correlation matrix to wavefront slopes
    centers_int = (setup.res/(setup.N*2)+setup.res/setup.N
                   *np.arange(0, setup.N)-border).astype('uint64')
    slopes = np.zeros([setup.N, setup.N, 2])
    for i in range(setup.N):
        for j in range(setup.N):
            
            # skip some subapertures, for now done manually
            if min(setup.N-1-i, i)+min(setup.N-1-j, j) < 2 and setup.N == 6:
                continue
            
            # normalize image
            im = corr[int(centers_int[i]-border):int(centers_int[i]+border), 
                      int(centers_int[j]-border):int(centers_int[j]+border)]
            im -= np.min(im)
            im /= np.max(im)
            im += .1
            
            # find the minimum value of image
            x = int(np.median(np.where(im==np.min(im))[0]))
            y = int(np.median(np.where(im==np.min(im))[1]))
            
            # interpole the location of the minimum value for sub-pixel accuray
            if im[x+1,y] < im[x-1,y]:
                slopes[i, j, 1] = (x-.5*(im[x+1,y]-im[x-1, y])
                                   /(im[x+1,y]-2*im[x,y]+im[x-1,y]) 
                                   - border)
            elif im[x+1,y] > im[x-1,y]:
                slopes[i, j, 1] = (x+.5*(im[x-1,y]-im[x+1, y])
                                   /(im[x-1,y]-2*im[x,y]+im[x+1,y])
                                   - border)
            else:
                slopes[i, j, 1] = x
            
            if im[x,y+1] < im[x,y-1]:
                slopes[i, j, 0] = (y-.5*(im[x,y+1]-im[x,y-1])
                                   /(im[x,y+1]-2*im[x,y]+im[x,y-1]) 
                                   - border) 
            elif im[x,y+1] > im[x,y-1]:
                slopes[i, j, 0] = (y+.5*(im[x,y-1]-im[x,y+1])/
                                   (im[x,y-1]-2*im[x,y]+im[x,y+1]) 
                                   - border)
            else:
                slopes[i, j, 0] = y
    
    # Errors from slopes, for now done manually
    # to find this matrix, load a flat wavefront and retrieve the slopes,
    # these slopes are removed here
    if setup.N == 8:
        slopes -= np.array([[[ 0,  0], [ 0,  0], [ 0,  0], [ 2.49214765e-01,  6.06005467e-02], [-2.43867886e-03,  6.51673048e-02], [ 0,  0], [ 0,  0], [ 0,  0]], [[ 0,  0], [ 1.26021137e-01,  1.40449116e-01], [ 1.61539538e-01,  7.53880468e-02], [ 2.31480458e-01,  6.46776544e-02], [ 4.46428560e-04,  7.37487723e-02], [ 7.85162090e-02,  8.24028726e-02], [ 9.18516793e-02,  1.53844554e-01], [ 0,  0]], [[ 0,  0], [ 7.07881886e-02,  1.61335372e-01], [ 1.55958663e-01,  1.54818575e-01], [ 2.32523066e-01,  1.52663273e-01], [ 6.86735013e-04,  1.57221381e-01], [ 8.12891502e-02,  1.57301272e-01], [ 1.61515288e-01,  1.53466387e-01], [ 0,  0]], [[ 6.41848990e-02,  2.49900357e-01], [ 6.40621324e-02,  2.45348770e-01], [ 1.45220300e-01,  2.41369259e-01], [ 2.33756896e-01,  2.33699576e-01], [-7.12976917e-03,  2.36579861e-01], [ 7.54330625e-02,  2.40466783e-01], [ 1.58218462e-01,  2.48129523e-01], [ 1.58871532e-01,  2.56868945e-01]], [[ 6.31023923e-02,  4.24469389e-03], [ 7.05156546e-02, -4.91059888e-04], [ 1.52168511e-01, -2.50537007e-03], [ 2.43723214e-01, -1.59453223e-03], [-1.84971705e-04, -8.23751100e-04], [ 8.25983526e-02,  3.90210599e-03], [ 1.70600227e-01, -1.94522574e-03], [ 1.62476694e-01, -1.63321967e-02]], [[ 0,  0], [ 7.37600329e-02,  8.06012454e-02], [ 1.46580219e-01,  8.51030727e-02], [ 2.34301007e-01,  7.74258101e-02], [-4.54505208e-03,  8.85571013e-02], [ 8.34133964e-02,  8.46985159e-02], [ 1.69218255e-01,  7.71526022e-02], [ 0,  0]], [[ 0,  0], [ 1.66599356e-01,  1.14567227e-01], [ 1.48682772e-01,  1.62169471e-01], [ 2.36729042e-01,  1.65530106e-01], [-7.18202858e-03,  1.76395115e-01], [ 6.82716761e-02,  1.65037399e-01], [ 7.52139982e-02,  9.77784254e-02], [ 0,  0]], [[ 0,  0], [ 0,  0], [ 0,  0], [ 2.60798165e-01,  1.88154443e-01], [-1.56122997e-02,  1.75102584e-01], [ 0,  0], [ 0,  0], [ 0,  0]]])
    if setup.N == 6:
        slopes -= np.array([[[ 0,  0], [ 0,  0], [ 8.24572859e-03, -1.06432543e-01], [ 5.99529896e-03, -9.56992552e-03], [ 0,  0], [ 0,  0]], [[ 0,  0], [-5.83200145e-01, -5.82559357e-01], [-5.67643573e-01, -1.17170839e-01], [-5.71454886e-01, -8.44801997e-03], [-5.75496409e-01, -5.72065171e-01], [ 0,  0]], [[-1.08936067e-01,  1.25730858e-02], [-1.18221616e-01, -5.68024875e-01], [-1.06968880e-01, -1.07323550e-01], [-1.09466605e-01,  2.63584106e-03], [-1.16971681e-01, -5.54453151e-01], [-1.06919368e-01, -1.28305822e-01]], [[-5.03460532e-03,  7.68953472e-03], [-1.08713759e-02, -5.72399052e-01], [ 4.45833122e-03, -1.09633420e-01], [ 2.11680675e-05, -2.39525975e-06], [-5.06640915e-03, -5.56343821e-01], [-1.81281834e-03, -1.27903215e-01]], [[ 0,  0], [-5.67888652e-01, -5.75334994e-01], [-5.51749824e-01, -1.17642141e-01], [-5.58120205e-01, -1.00138714e-02], [-5.72177928e-01, -5.77601013e-01], [ 0,  0]], [[ 0,  0], [ 0,  0], [-1.25290046e-01, -1.13314551e-01], [-1.19016532e-01, -4.28573584e-03], [ 0,  0], [ 0,  0]]]) 
        
        slopes[:, :, 0]-=np.mean(slopes[:,:,0])
        slopes[:, :, 1]-=np.mean(slopes[:,:,1])
        slopes[0:2, 0, ] = 0
        slopes[4:, 0, ] = 0
        slopes[0, 0:2, ] = 0
        slopes[0, 4:, ] = 0
        slopes[0:2, 5, ] = 0
        slopes[4:, 5, ] = 0
        slopes[5, 0:2, ] = 0
        slopes[5, 4:, ] = 0
        
    slopes *= -1.7
    
    return slopes

def Get_Slopes_From_Wavefront(setup):
    """
    return slopes of the wavefront, based on the currently loaded wavefront, 
not on the SH image!
    TO BE REMOVED IN FUTURE VERSION
    """
    warnings.warn("This function returns slopes directly from wavefront," 
                  +"it accesses information normally not available")
    basis = np.zeros([2, setup.res_subap**2])
    basis[0,] = np.tile(np.arange(0, setup.res_subap), 
                        (setup.res_subap, 1)).reshape([setup.res_subap**2])
    basis[1,] = np.tile(np.arange(0, setup.res_subap), 
                        (setup.res_subap, 1)).T.reshape([setup.res_subap**2])
    basis /= np.max(basis)
    basis_inv = np.linalg.pinv(basis)
    
    slopes = np.zeros([setup.N, setup.N, 2])
    for i in range(setup.N):
        for ii in range(setup.N):
            if min(setup.N-1-i, i)+min(setup.N-1-ii, ii) < 2:
                continue
            curr = setup.wavefront[int(np.floor(setup.res_SH/setup.N*i))
                                   :int(np.floor(setup.res_SH/setup.N*i)
                                        +setup.res_subap), 
                                   int(np.floor(setup.res_SH/setup.N*ii))
                                   :int(np.floor(setup.res_SH/setup.N*ii)
                                        +setup.res_subap)]
            curr = curr.reshape([setup.res_subap**2])
            slopes[i, ii, ] = np.dot(curr, basis_inv)
      
    return slopes
    
def Get_Wavefront_From_Slopes(setup, slopes):   
    """
    Returns a fit of zernike polynomials from the given slopes
    
    Inputs:
    
    setup: DFWS class object
    
    slopes: retrieved wavefront slopes
    
    Outputs:
    
    z: array of Zernike coefficients
    """
    
    # calculate the wavefront slopes of the Zernike basis polynomials
    z = np.array(aotools.functions.zernike.zernikeArray(24, setup.N*2))        
    dz_dx = np.zeros([24, setup.N, setup.N], dtype='float32')
    dz_dy = np.zeros([24, setup.N, setup.N], dtype='float32')
    for i in range(setup.N):
        for j in range(setup.N):
            if min(setup.N-1-i, i)+min(setup.N-1-j, j) < 2:
                continue
            dz_dx[:,i,j] = ((z[:, 2*i, 2*j]-z[:, 2*i+1, 2*j])
                            +(z[:, 2*i, 2*j+1]-z[:, 2*i+1, 2*j+1]))/2
            
            dz_dy[:,i,j] = ((z[:, 2*i, 2*j]-z[:, 2*i, 2*j+1])
                            +(z[:, 2*i+1, 2*j]-z[:, 2*i+1, 2*j+1]))/2
    
    # calculate the inverse of the slopes
    dz_dx = np.reshape(dz_dx, [24, setup.N**2])
    dz_dy = np.reshape(dz_dy, [24, setup.N**2])
    dz_inv = np.linalg.pinv(np.append(dz_dy, dz_dx, axis=1))
    
    # calculate the Zernike coefficients from the inverse slopes 
    # and the given slopes
    z = np.dot(np.append(np.reshape(slopes[:, :, 0], [setup.N**2]), 
                         np.reshape(slopes[:, :, 1], [setup.N**2]), 
                         axis=0).reshape([1,72]), dz_inv).reshape([24])
    return z

def Make_A_V2(setup):
    """
    Make the transformation matrix for the calculation of wavefront from slopes
    
    slopes = A * wavefront
    
    Inputs:
        
    setup: DFWS class object
    
    Outputs:
        
    A: matrix for zonal wavefront reconstruction
    """
    A = np.zeros([(setup.N**2)*2+2, (setup.N+2)**2])
    Bx = np.zeros([setup.N*2, setup.N+2])
    Bx[::2, :-2] = np.eye(setup.N)
    Bx[::2, 2:] += -np.eye(setup.N)
    By = np.zeros([setup.N*2, setup.N+2])
    By[1::2, 1:-1] = np.eye(setup.N)
    for i in range(setup.N):
        A[(2*setup.N)*i:(2*setup.N)*(i+1), 
          (setup.N+2)*i:(setup.N+2)*(i+1)] = By*.5
        A[(2*setup.N)*i:(2*setup.N)*(i+1), 
          (setup.N+2)*(i+1):(setup.N+2)*(i+2)] = Bx*.5
        A[(2*setup.N)*i:(2*setup.N)*(i+1), 
          (setup.N+2)*(i+2):(setup.N+2)*(i+3)] = -By*.5
    
    A[-2, 0] = 1; A[-2, setup.N+1] = 1; A[-2, -1] = 1; A[-2, -(setup.N+2)] = 1;
    A[-1, :] = 1
    return A

def get_wavefront_zonal(setup, slopes = None):
    """
    Return wavefront from a zonal method
    
    Inputs:
        
    setup: DFWS class object
    
    slopes [optional]: slopes of the SH sensor
    
    Outputs:
    
    None
    """
    
    # Retrieve A and calculate inverse
    if not hasattr(setup, 'A_inv'):
        A = Make_A_V2(setup)
        setup.A_inv =  np.linalg.pinv(A)
    
    # Retrieve slopes and convert to wavefront
    if slopes is None:
        slopes = get_Wavefront_slopes(setup)
        
    wavefront = np.dot(setup.A_inv, np.append(slopes.reshape([setup.N**2*2]), 
                                        [0, 0]).reshape([setup.N**2*2+2]))
    wavefront = wavefront.reshape([setup.N+2,setup.N+2])
    
    # Load wavefront into setup
    setup.load_wavefront(wavefront)
    setup.make_psf()
    setup.wavefront_zonal = setup.wavefront

def get_wavefront_modal(setup, slopes = None):
    """
    Return wavefront from a modal method
    
    Inputs:
        
    setup: DFWS class object

    slopes [optional]: slopes of the SH sensor    

    Outputs:
    
    None
    """
    
    # Retrieve wavefront slopes
    if slopes is None:
        slopes = get_Wavefront_slopes(setup)
    
    # Get Zernike coefficients from slopes
    z = Get_Wavefront_From_Slopes(setup, slopes)
    
    # Load into wavefront
    setup.wavefront_from_zernike(z)
    setup.make_psf()
    setup.wavefront_modal = setup.wavefront

def deconvole_sh(setup):
    """
    Return deconvoluted sh image in order to retrieve strehl ratio
    
    Inputs:
        
    setup: DFWS class object
    
    Outputs:
    
    psf_est: estimated sh-pattern
    """
    # get images
    setup.image_temp = setup.image
    setup.psf_temp = setup.psf
    
    # downres setup
    downres = np.round(np.arange(0, 680, 369/39)).astype('int64')
    setup.image = setup.image_sh
    setup.psf = setup.image[downres,][:,downres]
    
    setup.psf_est = deconvolve(setup, mode = 'Regularization', 
                               iterations=50)[1]
    
    setup.image = setup.image_temp
    setup.psf = setup.psf_temp
    del setup.image_temp, setup.psf_temp
    return setup.psf_est

def get_wavefront_DLWFS(setup, model, tip_iterations = 3):
    """
    Return wavefront from a deep learning wavefront sensing method
    
    Inputs:
        
    setup: DFWS class object
    
    model: tensorflow neural network 
    
    test [bool] [optional]: use new, experimental version of the TIP algorithm
    
    tip_iterations [int] [optional]: Amount of iterations for the TIP algorithm
    
    Outputs:
    
    None
    """
    numpy_or_cupy(setup.cupy_req)
    
    downres = numpy.around(numpy.arange(0, 680, 
                                        (680+210*2)/(setup.res_SH/setup.N)))
    downres = downres.astype('uint16')
    obj_est = setup.image[downres,][:,downres]

    
    setup.obj_est = obj_est

    psf_est, o = Run_SH_TIP(setup, iterations = tip_iterations, o_0 = obj_est)

    setup.psf_est = psf_est
    setup.o_est = o

    psf_est = tonumpy(psf_est)
    
    with tf.device('/CPU:0'):
        predictions  = model.predict((psf_est.reshape([1]
                                                      +list(psf_est.shape)
                                                      +[1])))
        predictions = predictions.astype('float16')
            
    setup.load_wavefront(predictions[0, :, :, 0])
    setup.make_psf(no_SH = True)
    pupil = numpy.array(aotools.functions.pupil.circle(340, 680))
    setup.wavefront *= pupil
    setup.wavefront -= numpy.mean(setup.wavefront)
    setup.wavefront *= pupil
    setup.wavefront_DLWFS = setup.wavefront

def plot_wavefront(setup):
    """
    Make a nice plot of the calculated wavefronts
    """
    
    cmap = copy(plt.cm.get_cmap('RdYlGn'))
    cmap.set_bad(alpha = 0)

    if hasattr(setup, 'wavefront_zonal') and hasattr(setup, 'wavefront_modal'): 
        vmin = min(setup.wavefront_0.min(), setup.wavefront.min(), 
                   setup.wavefront_modal.min(), setup.wavefront_zonal.min())
        vmax = max(setup.wavefront_0.max(), setup.wavefront.max(), 
                   setup.wavefront_modal.max(), setup.wavefront_zonal.max())
        v = max(abs(vmin), abs(vmax))
    
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
        image1 = setup.wavefront_0.astype('float32')
        image1[image1 == 0] = np.nan
        im = axes[0, 0].imshow(tonumpy(image1), cmap=cmap, vmin=-v, vmax=v)
        axes[0, 0].set_title('True Wavefront')
        axes[0, 0].axis('off')
        
        image1 = setup.wavefront.astype('float32')
        image1[image1 == 0] = np.nan
        im = axes[0, 1].imshow(tonumpy(image1), cmap=cmap, vmin=-v, vmax=v)
        axes[0, 1].set_title('Reconstructed Wavefront DLWFS')
        axes[0, 1].axis('off')
        
        image1 = setup.wavefront_zonal.astype('float32')
        image1[image1 == 0] = np.nan
        im = axes[1, 0].imshow(tonumpy(image1), cmap=cmap, vmin=-v, vmax=v)
        axes[1, 0].set_title('Reconstructed Wavefront Zonal')
        axes[1, 0].axis('off')
        
        image1 = setup.wavefront_modal.astype('float32')
        image1[image1 == 0] = np.nan
        im = axes[1, 1].imshow(tonumpy(image1), cmap=cmap, vmin=-v, vmax=v)
        axes[1, 1].set_title('Reconstructed Wavefront Modal')
        axes[1, 1].axis('off')
    else:
        vmin = min(setup.wavefront_0.min(), setup.wavefront.min())
        vmax = max(setup.wavefront_0.max(), setup.wavefront.max())
        v = max(abs(vmin), abs(vmax))
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
        image1 = setup.wavefront_0.astype('float32')
        image1[image1 == 0] = np.nan
        im = axes[0].imshow(tonumpy(image1), cmap=cmap, vmin=-v, vmax=v)
        axes[0].set_title('True Wavefront')
        axes[0].axis('off')
        
        image1 = setup.wavefront.astype('float32')
        image1[image1 == 0] = np.nan
        im = axes[1].imshow(tonumpy(image1), cmap=cmap, vmin=-v, vmax=v)
        axes[1].set_title('Reconstructed Wavefront')
        axes[1].axis('off')
    
    cbar = fig.colorbar(im, ax=axes, location='right', shrink=0.7)
    cbar.set_label('Rad', labelpad=-0, rotation=90)
    plt.show()

def plot_object(setup):
    """
    Make a nice plot of the calculated wavefronts
    """
    if hasattr(setup, 'object'):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
        im = axes[0].imshow(tonumpy(setup.object).astype('float32'), 
                            cmap='gray')
        axes[0].set_title('True Object')
        axes[0].axis('off')
        im = axes[1].imshow(tonumpy(setup.image).astype('float32'), 
                            cmap='gray')
        axes[1].set_title('Received Image')
        axes[1].axis('off')
        im = axes[2].imshow(tonumpy(setup.object_estimate).astype('float32'), 
                            cmap='gray')
        axes[2].set_title('Estimated Object')
        axes[2].axis('off')
    else:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
        im = axes[0].imshow(tonumpy(setup.image).astype('float32'), 
                            cmap='gray')
        axes[0].set_title('Received Image')
        axes[0].axis('off')
        im = axes[1].imshow(tonumpy(setup.object_estimate).astype('float32'), 
                            cmap='gray')
        axes[1].set_title('Estimated Object')
        axes[1].axis('off')

    plt.show()
    
def plot_rms_bar(RMSE, names, remove_outlyers = False):   
    """
    Make a nice plot of the RMSE wavefront estimation performances
    """
    numpy_or_cupy(0)
    plt.figure(figsize=(12,5))
    spacing = .1
    for i in range(len(names)):
        means = np.mean(RMSE[:, i], axis=-1)
        std = np.std(RMSE[:, i], axis=-1)
        if remove_outlyers:
            outlyer  = (np.abs(RMSE[:,i]-means.repeat([RMSE.shape[2]])
                              .reshape([RMSE.shape[0],RMSE.shape[2]]))
                        >3*std.repeat([RMSE.shape[2]])
                                .reshape([RMSE.shape[0],RMSE.shape[2]]))
            RMSE_plot = (RMSE[:,i]*(1-outlyer)+means.repeat([RMSE.shape[2]])
                        .reshape([RMSE.shape[0],RMSE.shape[2]])*outlyer)
            std = np.std(RMSE_plot, axis=-1)
            means = np.mean(RMSE_plot, axis=-1)
        plt.errorbar(np.arange(0, RMSE.shape[0])-(spacing*2-spacing*i)/1.5, 
                     means, yerr= std, fmt='o', label=names[i])
        
        print(names[i], 'mean RMSE = ', np.mean(means[12:]))
    plt.xticks(np.arange(0, RMSE.shape[0]))
    plt.xlabel('D/r0')
    plt.ylabel('RMS Wavefront Error [rad]')
    plt.legend(loc="upper left")
    plt.title('Wavefront Estimation Error')
    plt.plot([-100, 100], [1, 1], 'k--')
    plt.axis([-.5, RMSE.shape[0]-.5, 0, 3])
    

def rmse(setup, b = None, unwrap = False):
    """
    Calculates the root mean square error of the circular wavefronts 
    setup.wavefront and setup.wavefront_0 (true wavefront)
    
    Inputs:
        
    setup: DFWS class object OR matrix 1
    
    b: matrix 2
    
    Outputs:
        
    rms: root mean square error between setup.wavefront and setup.wavefront_0
         OR root mean square error between setup and b
    
    
    """
    
    if (hasattr(setup, 'wavefront') and hasattr(setup, 'wavefront_0') 
        and b is None):
        setup.remove_ptt('wavefront_0')
        setup.remove_ptt('wavefront')
        
        residual = tonumpy(setup.wavefront)-tonumpy(setup.wavefront_0)
        
        if unwrap:
            while numpy.any(numpy.abs(residual)>numpy.pi):
                residual[residual>np.pi] -= numpy.pi
                residual[residual<-np.pi] += numpy.pi
            
        mse = numpy.mean((residual)**2)
        
        # Fill factor is the amount of pixels actually in the circle,
        # for which the rms wavefront error needs to be compensated, 
        # should be around pi/4, depending on the resolution
        size = setup.wavefront.shape[0]
        fill_factor = numpy.sum(circle(size/2, size))/size**2    
        rms = numpy.sqrt(mse/fill_factor)
        
    elif setup.shape == b.shape:
        rms = numpy.sqrt(numpy.mean((tonumpy(setup)-tonumpy(b))**2))
        
    return rms
