#   Deconvolution from Wavefront Sensing Solver!
#   Created by Bas de Bruijne, Sep 09, 2020
#   For questions, contact BasdBruijne@gmail.com
#
import DFWS_Simulator as sim
import aotools
import matplotlib.pyplot as plt
import numpy
#import cupy
import warnings
import sys
from tqdm import tqdm
import numpy as np
from numpy import fft
from scipy.ndimage import rotate
import cupy as cp
import threading
from scipy.signal import convolve2d

cupy = 0

def tonumpy(x):
    # Returns a variable from either cupy or numpy to numpy
    try:
        return cp.asnumpy(x)
    except:
        return x
    
def convolve2(a, b):
    # Fourier domain convolution between two matrices
    # Make sure matrices are square
    if a.shape[0] != a.shape[1] or b.shape[0] != b.shape[1]:
        raise Exception('Please enter square matrices')
    
    # Add padding to the matrices
    a_pad = np.pad(np.array(a), [0, b.shape[0]-1], mode='constant')
    b_pad = np.pad(np.array(b), [0, a.shape[0]-1], mode='constant')
    
    # Convolve the image and crop the edges
    edge = np.minimum(a.shape[0], b.shape[0])/2
    c = np.fft.ifft2(np.fft.fft2(a_pad)*np.fft.fft2(b_pad))[int(np.floor(edge)):-int(np.ceil(edge))+1, int(np.floor(edge)):-int(np.ceil(edge))+1]      
    return c

def numpy_or_cupy(cupy_req):
    # Check if numpy or cupy should be used and load right libraries
    
    global np, fft, rotate, cupy
    if cupy_req:
        import cupy as np  
        from cupy import fft
        from cupyx.scipy.ndimage import rotate
        cupy = 1
    else:
        import numpy as np 
        from numpy import fft
        from scipy.ndimage import rotate
        cupy = 0
        
def Downres_Image(setup):
    # Reduce the resolution of the image
    
    numpy_or_cupy(setup.cupy_req)
    # Downsizes the main image from sensor to the size of the subaperture images
    downres = np.around(np.arange(0, setup.res, setup.res/(setup.res_SH/setup.N))).astype('int')
    object_est = setup.image[downres, :]
    object_est = object_est[:, downres]
    return object_est

def Run_SH_TIP(setup, iterations = 20, o_0 = None, psf_0 = None, pupil = None):
    # Performs the TIP-alogirithm on the Shack-Hartmann image of setup in order to find the Shack-Hartmann PSF
    # o_0 = initial estimate of object
    # psf_0 = initial estimate of PSF
    # pupil = mask for the containment of the estimated object
    # 
    # output:
    # Estimated object & estimated PSF
    
    numpy_or_cupy(setup.cupy_req)
    if o_0 is None:
        # If no inital estimate is provided, use circular function with ones
        object_est = np.array(aotools.functions.pupil.circle(setup.res_SH/setup.N/5, int(np.ceil(setup.res_SH/setup.N)), circle_centre=(0, 0), origin='middle'))
    else:
        object_est = o_0
        
    if not psf_0 is None and not o_0 is None:
        warnings.warn("Initial estimates of both object and psf are provided. Only psf estimate will be used")

    o_zero = fft.fftshift(np.pad(object_est, [int(np.floor((setup.image_sh.shape[0]+1)/2)), int(np.ceil((setup.image_sh.shape[0]+1)/2))], mode='constant'))
    i_F = fft.fft2(np.pad(setup.image_sh, [0, int(object_est.shape[0]+1)], mode='constant'))
    o_F = fft.fft2(o_zero)
    
        
    if not hasattr(setup, 'tip_pupil'):
        if pupil is None:
            setup.tip_pupil  = fft.fftshift(np.array(aotools.functions.pupil.circle(18, o_F.shape[0], circle_centre=(0, 0), origin='middle'))).astype('float16')
            setup.tip_pupil  += .1*(setup.tip_pupil == 0)
        else:
            setup.tip_pupil = pupil
                
    # Run the TIP iterations
    for n in range(0, iterations): #tqdm(range(0, iterations)):
        if n != 0 or psf_0 is None:
            psf_est_F = i_F/(o_F+1)
            psf_est = np.abs(fft.ifft2(psf_est_F))                
            psf_est /= np.sum(psf_est)
            psf_est_F = fft.fft2(psf_est)

        else:
            psf_0 /= np.sum(psf_0)
            psf_est_F = fft.fft2(np.pad(psf_0, [0, int(i_F.shape[0]-psf_0.shape[0])], mode='constant'))
        
        conj = np.conj(psf_est_F)
        o_F = (conj*i_F)/(conj*psf_est_F+1e-9)
        o = np.real(fft.ifft2(o_F))
        o *= setup.tip_pupil
        o -= np.min(o)
        o = o/np.max(o)
        o_F = fft.fft2(o)
        
    psf_est = np.real(psf_est)
    psf_est = convert_680_to_128(psf_est)
    psf_est -= np.min(psf_est)
    psf_est /= np.max(psf_est)
    setup.psf_sh_128 = psf_est
        
    return psf_est, fft.fftshift(o)

def Denoise(setup):
    numpy_or_cupy(setup.cupy_req)
    # Removes noise by setting all image values to zero except for a region around each maximum value in the subaperture region
    # Kernel_1d provides the 1 dimensional function according to which the values are preserved around the maximum
    
    if not hasattr(setup, 'denoise_kernel'):
        kernel_size = 9
        setup.denoise_kernel = np.zeros([kernel_size, kernel_size])
        for i in range(0, kernel_size):
            for j in range(0, kernel_size):
                try:
                    setup.denoise_kernel[i, j] = 1/(((i-np.floor(kernel_size/2)))**2 + (j-np.floor(kernel_size/2))**2)**.5
                except:
                    setup.denoise_kernel[i, j] = 1
        setup.denoise_kernel[setup.denoise_kernel>=setup.denoise_kernel[np.floor(kernel_size/2).astype('int64'), 2]] = 1
    
    psf_tip_kern = np.zeros(setup.psf_SH_128.shape, dtype = 'float32')
    
    block_shape = setup.psf_SH_128.shape[0]
    for i in range(0, setup.N):
        for j in range(0, setup.N):
            try:
                if min(setup.N-1-i, i)+min(setup.N-1-j, j) < 2:
                    continue
                R = setup.denoise_kernel.shape[0]
                x0 = int(block_shape/setup.N*i)
                y0 = int(block_shape/setup.N*j)
                crop = setup.psf_SH_128[x0:int(block_shape/setup.N*(i+1)), y0:int(block_shape/setup.N*(j+1))]
                max_loc = np.where(crop==np.max(crop))
                kernel_big = np.zeros(setup.psf_SH_128.shape)
                kernel_big[int(x0+max_loc[0]-np.floor(R/2)):int(x0+max_loc[0]+R-np.floor(R/2)), int(y0+max_loc[1]-np.floor(R/2)):int(y0+max_loc[1]+R-np.floor(R/2))] = setup.denoise_kernel
                psf_tip_kern += setup.psf_SH_128*kernel_big
            except:
                pass
            
    psf_tip_kern -= np.min(psf_tip_kern)
    psf_tip_kern /= np.max(psf_tip_kern)
    return psf_tip_kern*(psf_tip_kern>.05)

def Denoise_Image(Noi_Img):
    # Remote noise from image
    # Discontinued will be removed in future.
    warnings.warn("Denoise_Image will be removed in future versions")
    
    t = 0.45
    Noi_Img = np.pad(Noi_Img, [30, 30])
    F_in = np.fft.fftshift(np.fft.fft2(Noi_Img))
    P = np.sqrt(np.real(F_in)**2+np.imag(F_in)**2)
    Th = 0.35; L_rgn = 3; F_out = F_in; Wmax = 21
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            d = np.sqrt((i-P.shape[0]/2)**2+(j-P.shape[1]/2)**2)
            fl = 0
            if d > L_rgn:
                W1 = 3; cond = 1
                while cond == 1:
                    W2 = W1 + 2
                    # S1_int = P[i-W1:i+W1, j-W1:j+W1]
                    # S2_int = P[i-W2:i+W2, j-W2:j+W2]
                    S1_int = P[max(i-W1,0):min(i+W1, P.shape[0]), max(j-W1,0):min(j+W1, P.shape[1])]
                    S2_int = P[max(i-W2,0):min(i+W2, P.shape[0]), max(j-W2,0):min(j+W2, P.shape[1])]
                    S1 = np.mean(S1_int)
                    S2 = (np.sum(S2_int)-np.sum(S1_int))/(np.prod(S2_int.shape)-np.prod(S1_int.shape))
                    if S2/S1 <= t:
                        fl = 1
                        if W2 >= Wmax:
                            cond = 0
                        else:
                            W1 += 2
                    else:
                        cond = 0
            if fl == 1:
                print('Notch at i = ', i, ' j = ', j)
                G = Gaussian_Filter(P.shape[0], i, j, 1, 1/W1**2)
                F_out = np.minimum(F_out, F_out*G)
    Image_out = np.abs(np.fft.ifft2(np.fft.fftshift(F_out)))
    return Image_out
    
def plot(image, figure = None):
    # Show grayscale image, but without the hassle of making sure that the data type and memory is right
    
    if figure is None:
        plt.figure()
    else:
        plt.figure(figure)
    try:
        plt.imshow(image.astype('float32'), cmap='gray')
    except:
        plt.imshow(image.astype('float32').get(), cmap='gray')

def deconvolve_CPU(start, end, iterations):
    # Function used for parallel deconvolution of image, do not call directly!
    
    global o_F, i_F, psf_F, psf_mirror_F
    tau = 1.5
    for n in range(iterations):
        o_F[start:end,] += tau*((i_F[start:end,]-psf_F[start:end,]*o_F[start:end,])*psf_mirror_F[start:end,])
        
def deconvolve(setup, mode = 'LR', iterations = 10, regularization_term = 1e-1, force_cuda = False):
    # Deconvolves the image based on the loaded psf
    # Inputs:
    # mode = mode of deconvolution, scroll down to see them all
    # Iterations = amount of iterations used for the iterative methods
    # regularization_term = regularization used for the methods that are use it
    # force_cuda (bool) = use cuda eventhough cuda is not yet loaded
    
    global o_F, i_F, psf_F, psf_mirror_F
    
    if mode == 'CPU_LR':
        numpy_or_cupy(0)
    elif force_cuda:
        numpy_or_cupy(1)
    else:
        numpy_or_cupy(setup.cupy_req)
    
    if not hasattr(setup, 'image'):
        raise Exception("Setup does not have an image loaded, deconvolution cannot proceed")
    
    # Initialize the variables and make sure the psf is normalized
    setup.psf /= np.sum(setup.psf)
    
    # Define the Fourier transform counterparts of the variables
    i_F = fft.fft2(np.pad(np.array(setup.image), [int(np.floor((setup.psf.shape[0]-1)/2)), int(np.ceil((setup.psf.shape[0]-1)/2))], mode='reflect'))
    psf_F = fft.fft2(np.pad(np.array(setup.psf), [0, setup.image.shape[0]-1], mode='constant'))
    o_F = i_F

    if 'LR' in mode or mode == 'Steepest_Descent':
        psf_mirror_F = fft.fft2(np.pad(np.array(setup.psf[::-1, ::-1]), [setup.image.shape[0]-1, 0], mode='constant'))
        # psf_mirror_F = fft.fft2(np.pad(np.array(setup.psf[::-1, ::-1]), [setup.image.shape[0]-setup.psf.shape[0], 0], mode='constant'))
        
    if mode == 'LR':
        tau = 1.5
        a = tau*i_F*psf_mirror_F
        b = -1*tau*psf_F*psf_mirror_F
        for n in range(iterations):
            o_F += a + o_F*b

    elif mode == 'LR2':
        tau = 2
        a = i_F*psf_mirror_F
        b = -1*psf_F*psf_mirror_F
        for n in range(iterations):
            tau -= .5/iterations
            o_F += tau*(a + o_F*b)
  
    elif mode == 'CPU_LR':
        # 
        threads = 4
        steps = np.around(np.arange(0, i_F.shape[0]+.1, i_F.shape[0]/threads))
        t = []
        for i in range(threads):
            t.append(threading.Thread(target=deconvolve_CPU, args=(int(steps[i]),int(steps[i+1]),iterations)))
            t[i].start()
        
        for i in range(threads):
            t[i].join()
            
    elif mode == 'Regularization':
        if iterations != 10:
            print('Regularization mode does not work iteratively')
            
        o_F = i_F/(psf_F+regularization_term*(psf_F<regularization_term))
        
    elif mode == 'Regularization_Filter':
        if iterations != 10:
            print('Regularization mode does not work iteratively')
        
        psf_F = (psf_F+regularization_term*(psf_F<regularization_term))
        o_F = i_F/(psf_F)
        o_F_2 = np.log(1+np.abs(o_F))
        o_F_2 -= np.min(o_F_2)
        o_F_2 /= np.max(o_F_2)
        
        if not hasattr(setup, 'Regularization_Filter'):
            setup.Regularization_Filter = -1*np.array(aotools.functions.zernike.zernikeArray([4], o_F_2.shape[0]), dtype='float32')[0,]
            setup.Regularization_Filter -= np.min(setup.Regularization_Filter)
            setup.Regularization_Filter /= np.max(setup.Regularization_Filter)
            setup.Regularization_Filter[np.where(setup.Regularization_Filter==setup.Regularization_Filter[0,0])] = 0
            setup.Regularization_Filter[np.where(np.fft.fftshift(np.array(aotools.functions.pupil.circle(20, o_F_2.shape[0]))))] = 1
            setup.Regularization_Filter **= 2
            setup.Regularization_Filter = setup.Regularization_Filter*.2+.8
            
        o_F = o_F  + (np.mean(o_F) - o_F)*(o_F_2>setup.Regularization_Filter)
        
    elif mode == 'Regularization_Filter2':
        if iterations != 10:
            print('Regularization mode does not work iteratively')
        
        psf_F = (psf_F+regularization_term)#*(psf_F<regularization_term))
        o_F = i_F/(psf_F)
                
        if not hasattr(setup, 'G1'):
            setup.a1 = np.pad(np.ones([3,3]),[2,2])
            setup.a2 = np.ones([7,7])-setup.a1
            setup.a1 /= np.sum(setup.a1)
            setup.a2 /= np.sum(setup.a2)
            setup.a2 = np.fft.fft2(np.pad(setup.a2, [0, o_F.shape[0]-setup.a2.shape[0]]))
            setup.d = np.fft.fftshift(np.array(aotools.functions.pupil.circle(6, o_F.shape[0])))
            setup.e = (setup.d==0)
            G1_shape = 5
            setup.G1 = np.ones([G1_shape, G1_shape])
            for i in range(G1_shape):
                for j in range(G1_shape):
                    i1 = np.abs(int(np.floor(G1_shape/2))-i)
                    i2 = np.abs(int(np.floor(G1_shape/2))-j)
                    setup.G1[i, j] = 1 * np.exp(-1/4*(i1**2+i2**2))
            setup.G1 = np.fft.fft2(np.pad(setup.G1, [0, o_F.shape[0]-setup.G1.shape[0]]))
            setup.threshold = .25

        o_F_norm = np.abs(o_F)
        c2 = np.fft.ifft2(np.fft.fft2(o_F_norm)*(setup.a2))
        b = (c2/o_F_norm<=setup.threshold)
        b = np.fft.ifft2(np.fft.fft2(b)*(setup.G1))
        b = ((1-b*setup.e))
        b *= (b>0)
        o_F *= b

    elif mode == 'TIP':
        pupil = fft.fftshift(np.array(aotools.functions.pupil.circle(60, o_F.shape[0], circle_centre=(0, 0), origin='middle'))).astype('float16')
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
    
    elif mode == 'LR_Steepest_Descent':
        
        r = i_F - psf_F * o_F
        
        for n in range(iterations):
            d = psf_mirror_F*r
            w = psf_F*d
            t = numpy.linalg.norm(d,2)/numpy.linalg.norm(w,2)
            o_F += t*d
            r -= t*w
            
    elif mode == 'LR_preconditioning':
        
        tau = 1
        r = i_F - psf_F * o_F
        M = np.diag(np.dot(psf_F.T, psf_F)) + tau * np.tril(np.dot(psf_F.T, psf_F)) 
        v = np.linalg.inv(M)*r
        d = psf_mirror_F*v
        
        for n in range(iterations):
            o_F += tau*d
            r = i_F - psf_F * o_F
            M = np.diag(np.dot(psf_F.T, psf_F)) + tau * np.tril(np.dot(psf_F.T, psf_F))
            v = np.linalg.inv(M)*r
            d = psf_mirror_F*v
            
    else:
        raise Exception('Deconvolution method not recognized')
    
    output = np.abs(fft.ifft2(o_F))
    output -= np.min(output)
    output /= np.max(output)

    return setup, tonumpy(output[0:setup.image.shape[0], 0:setup.image.shape[0]]).astype('float16')

def convert_680_to_256(psf_est):
    # Reduce black space within SH image to 256 pixels
    
    begin = numpy.array(numpy.round((680/6*numpy.arange(1,7)-680/12)-256/12), dtype='int16')
    end = numpy.array(numpy.round((680/6*numpy.arange(1,7)-680/12)+256/12), dtype='int16')
    index = numpy.array(numpy.r_[int(begin[0]):int(end[0])], dtype='int16')
    
    for i in range(1, 6):
        index = numpy.r_[index, int(begin[i]):int(end[i])]
    
    index = numpy.r_[index, int(end[5]+1)]
        
    index = np.array(index)
    return psf_est[index][:,index]

def convert_680_to_128(psf_est):
    # Reduce black space within SH image to 128 pixels
    begin = numpy.array(numpy.round((680/6*numpy.arange(1,7)-680/12)-128/12), dtype='int16')
    end = numpy.array(numpy.round((680/6*numpy.arange(1,7)-680/12)+128/12), dtype='int16')
    index = numpy.array(numpy.r_[int(begin[0]):int(end[0])], dtype='int16')
    
    for i in range(1, 6):
        index = numpy.r_[index, int(begin[i]):int(end[i])]
        
    index = np.array(index)
    return psf_est[index][:,index]

def convert_256_to_128(psf_est, offset = 0):
    # Reduce black space within SH image to 128 pixels from 256 pixels
    res = 256+2*offset
    res_new = 128
    begin = -offset+numpy.array(numpy.floor((res/6*numpy.arange(1,7)-res/12)-res_new/12), dtype='int32')
    end = -offset+numpy.array(numpy.floor((res/6*numpy.arange(1,7)-res/12)+res_new/12), dtype='int32')     
    index = numpy.array(numpy.r_[int(begin[0]):int(end[0])], dtype='int16')
    for i in range(1, 6):
        index = numpy.r_[index, int(begin[i]):int(end[i])]
    index = numpy.r_[index, int(end[5]+1)]        
    index = np.array(index)
    return psf_est[index][:,index]

def Allign_Shack_Hartmann(setup):
    # Allign SH image based on rotation and shift data in 'rotate_shift_data.npz'
    
    numpy_or_cupy(setup.cupy_req)
    if not hasattr(setup, 'alpha'):
        data = np.load('rotate_shift_data.npz')
        setup.angle = data['name1']
        setup.y0 = data['name2']
        setup.y1 = data['name3']
        setup.x0 = data['name4']
        setup.x1 = data['name5']

    setup.image_sh = rotate(setup.image_sh, setup.angle)[setup.y0:setup.y1, setup.x0:setup.x1]
    
    return setup

def get_Wavefront_slopes(setup, obj_est = None):
    # Returns the slopes of the wavefront estimated from a correlation algorithm
    # i.e. conventional way of extended scene wavefront sensing
    border = 25
    if obj_est is None:
        ref = np.zeros([2*border, 2*border])
        for i in range(2,4):
            for j in range(2,4):
                i = 3; j = 3
                x = int(setup.res/(setup.N*2)+setup.res/setup.N*i)
                y = int(setup.res/(setup.N*2)+setup.res/setup.N*j)
                ref += setup.image_sh[x-border:x+border, y-border:y+border]
        ref /= np.max(ref)
    else:
        ref = obj_est[int((obj_est.shape[0]-2*border)/2):int((obj_est.shape[0]-2*border)/2)+2*border, int((obj_est.shape[0]-2*border)/2):int((obj_est.shape[0]-2*border)/2)+2*border]
        
    corr = np.zeros([setup.res_SH-2*border, setup.res_SH-2*border])
    for i in tqdm(range(setup.res_SH-2*border)):
        for j in range(setup.res_SH-2*border):
            corr[i,j] = np.sum(np.abs(setup.image_sh[i:i+2*border, j:j+2*border]-ref)**2)
     
    centers = setup.res/(setup.N*2)+setup.res/setup.N*np.arange(0, setup.N)-border       
    centers_int = (setup.res/(setup.N*2)+setup.res/setup.N*np.arange(0, setup.N)-border).astype('uint64')
    slopes = np.zeros([setup.N, setup.N, 2])
    for i in range(setup.N):
        for j in range(setup.N):
            if min(setup.N-1-i, i)+min(setup.N-1-j, j) < 2:
                continue
            im = corr[int(centers_int[i]-border):int(centers_int[i]+border), int(centers_int[j]-border):int(centers_int[j]+border)]
            im -= np.min(im)
            im /= np.max(im)
            im += .1
            # im = np.max(im)-im
            x = int(np.median(np.where(im==np.min(im))[0]))
            y = int(np.median(np.where(im==np.min(im))[1]))
            
            if im[x+1,y] < im[x-1,y]:
                slopes[i, j, 1] = x-.5*(im[x+1,y]-im[x-1, y])/(im[x+1,y]-2*im[x,y]+im[x-1,y]) - border    
            elif im[x+1,y] > im[x-1,y]:
                slopes[i, j, 1] = x+.5*(im[x-1,y]-im[x+1, y])/(im[x-1,y]-2*im[x,y]+im[x+1,y]) - border 
            else:
                slopes[i, j, 1] = x
            
            # slopes[i,j,0] += (centers-centers_int)[3] - (centers-centers_int)[i]
            
            if im[x,y+1] < im[x,y-1]:
                slopes[i, j, 0] = y-.5*(im[x,y+1]-im[x,y-1])/(im[x,y+1]-2*im[x,y]+im[x,y-1]) - border    
            elif im[x,y+1] > im[x,y-1]:
                slopes[i, j, 0] = y+.5*(im[x,y-1]-im[x,y+1])/(im[x,y-1]-2*im[x,y]+im[x,y+1]) - border 
            else:
                slopes[i, j, 0] = y
    
    # Sloppy, I know
    slopes -= np.array([[[ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 8.24572859e-03, -1.06432543e-01],
        [ 5.99529896e-03, -9.56992552e-03],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00]],

       [[ 0.00000000e+00,  0.00000000e+00],
        [-5.83200145e-01, -5.82559357e-01],
        [-5.67643573e-01, -1.17170839e-01],
        [-5.71454886e-01, -8.44801997e-03],
        [-5.75496409e-01, -5.72065171e-01],
        [ 0.00000000e+00,  0.00000000e+00]],

       [[-1.08936067e-01,  1.25730858e-02],
        [-1.18221616e-01, -5.68024875e-01],
        [-1.06968880e-01, -1.07323550e-01],
        [-1.09466605e-01,  2.63584106e-03],
        [-1.16971681e-01, -5.54453151e-01],
        [-1.06919368e-01, -1.28305822e-01]],

       [[-5.03460532e-03,  7.68953472e-03],
        [-1.08713759e-02, -5.72399052e-01],
        [ 4.45833122e-03, -1.09633420e-01],
        [ 2.11680675e-05, -2.39525975e-06],
        [-5.06640915e-03, -5.56343821e-01],
        [-1.81281834e-03, -1.27903215e-01]],

       [[ 0.00000000e+00,  0.00000000e+00],
        [-5.67888652e-01, -5.75334994e-01],
        [-5.51749824e-01, -1.17642141e-01],
        [-5.58120205e-01, -1.00138714e-02],
        [-5.72177928e-01, -5.77601013e-01],
        [ 0.00000000e+00,  0.00000000e+00]],

       [[ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00],
        [-1.25290046e-01, -1.13314551e-01],
        [-1.19016532e-01, -4.28573584e-03],
        [ 0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00]]]) 
    
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
    # return slopes of the wavefront, based on the currently loaded wavefront, not on the SH image!
    warnings.warn("This function returns slopes directly from wavefront, it accesses information normally not available")
    basis = np.zeros([2, setup.res_subap**2])
    basis[0,] = np.tile(np.arange(0, setup.res_subap), (setup.res_subap, 1)).reshape([setup.res_subap**2])
    basis[1,] = np.tile(np.arange(0, setup.res_subap), (setup.res_subap, 1)).T.reshape([setup.res_subap**2])
    basis /= np.max(basis)
    basis_inv = np.linalg.pinv(basis)
    
    slopes = np.zeros([setup.N, setup.N, 2])
    for i in range(setup.N):
        for ii in range(setup.N):
            if min(setup.N-1-i, i)+min(setup.N-1-ii, ii) < 2:
                continue
            curr = setup.wavefront[int(np.floor(setup.res_SH/setup.N*i)):int(np.floor(setup.res_SH/setup.N*i)+setup.res_subap), int(np.floor(setup.res_SH/setup.N*ii)):int(np.floor(setup.res_SH/setup.N*ii)+setup.res_subap)]
            curr = curr.reshape([setup.res_subap**2])
            slopes[i, ii, ] = np.dot(curr, basis_inv)
      
    return slopes
    
def Get_Wavefront_From_Slopes(setup, slopes):   
    # Returns a fit of zernike polynomials from the given slopes
    
    z = np.array(aotools.functions.zernike.zernikeArray(24, setup.N*2))        
    dz_dx = np.zeros([24, setup.N, setup.N], dtype='float32')
    dz_dy = np.zeros([24, setup.N, setup.N], dtype='float32')
    for i in range(setup.N):
        for j in range(setup.N):
            if min(setup.N-1-i, i)+min(setup.N-1-j, j) < 2:
                continue
            dz_dx[:,i,j] = ((z[:, 2*i, 2*j]-z[:, 2*i+1, 2*j])+(z[:, 2*i, 2*j+1]-z[:, 2*i+1, 2*j+1]))/2
            
            dz_dy[:,i,j] = ((z[:, 2*i, 2*j]-z[:, 2*i, 2*j+1])+(z[:, 2*i+1, 2*j]-z[:, 2*i+1, 2*j+1]))/2
    
    # dz_dx_inv = np.linalg.pinv(np.reshape(dz_dx, [24, setup.N**2]))
    # dz_dy_inv = np.linalg.pinv(np.reshape(dz_dy, [24, setup.N**2]))
    dz_dx = np.reshape(dz_dx, [24, setup.N**2])
    dz_dy = np.reshape(dz_dy, [24, setup.N**2])
    
    dz_inv = np.linalg.pinv(np.append(dz_dy, dz_dx, axis=1))
    
    z = np.dot(np.append(np.reshape(slopes[:, :, 0], [setup.N**2]), np.reshape(slopes[:, :, 1], [setup.N**2]), axis=0).reshape([1,72]), dz_inv).reshape([24])
    return z

def Make_A_V2(setup):
    # Make the transformation matrix for the calculation of wavefront from slopes
    # slopes = A * wavefront
    
    A = np.zeros([(setup.N**2)*2+2, (setup.N+2)**2])
    Bx = np.zeros([setup.N*2, setup.N+2])
    Bx[::2, :-2] = np.eye(setup.N)
    Bx[::2, 2:] += -np.eye(setup.N)
    By = np.zeros([setup.N*2, setup.N+2])
    By[1::2, 1:-1] = np.eye(setup.N)
    for i in range(setup.N):
        A[(2*setup.N)*i:(2*setup.N)*(i+1), (setup.N+2)*i:(setup.N+2)*(i+1)] = By*.5
        A[(2*setup.N)*i:(2*setup.N)*(i+1), (setup.N+2)*(i+1):(setup.N+2)*(i+2)] = Bx*.5
        A[(2*setup.N)*i:(2*setup.N)*(i+1), (setup.N+2)*(i+2):(setup.N+2)*(i+3)] = -By*.5
    
    A[-2, 0] = 1; A[-2, setup.N+1] = 1; A[-2, -1] = 1; A[-2, -(setup.N+2)] = 1;
    A[-1, :] = 1
    return A

def get_wavefront_zonal(setup):
    # Return wavefront from a zonal method
    
    A = Make_A_V2(setup)
    A_inv =  np.linalg.pinv(A) #np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)
    slopes = get_Wavefront_slopes(setup)
    wavefront = np.dot(A_inv, np.append(slopes.reshape([72]), [0, 0]).reshape([74])).reshape([8,8])
    setup.load_wavefront(wavefront)
    setup.make_psf()
    setup.wavefront_zonal = setup.wavefront

def get_wavefront_modal(setup):
    # Return wavefront from a modal method
    
    slopes = get_Wavefront_slopes(setup)
    z = Get_Wavefront_From_Slopes(setup, slopes)
    setup.wavefront_from_zernike(z)
    setup.make_psf()
    setup.wavefront_modal = setup.wavefront

def get_wavefront_DLWFS(setup, model):
    # Return wavefront from a deep learning method
    
    downres = np.round(np.arange(0, 680, 369/39)).astype('int64')
    obj_est = setup.image[downres,][:,downres]
    psf_est, o = Run_SH_TIP(setup, iterations = 3, o_0 = obj_est, psf_0 = None, pupil = None, temp = None)
    plot(psf_est)

    predictions  = model.predict((psf_est.reshape([1]+list(psf_est.shape)+[1])).astype('float16'))
    setup.load_wavefront(predictions[0, :, :, 0])
    setup.make_psf()
    pupil = np.array(aotools.functions.pupil.circle(340, 680))
    setup.wavefront *= pupil
    setup.wavefront -= np.mean(setup.wavefront)
    setup.wavefront *= pupil
    setup.wavefront_DLWFS = setup.wavefront

def plot_wavefront(setup):
    # Make a nice plot of the calculated wavefronts
    
    cmap = plt.cm.get_cmap('RdYlGn')
    cmap.set_bad(alpha = 0)

    if hasattr(setup, 'wavefront_zonal') and hasattr(setup, 'wavefront_modal'): 
        vmin = min(setup.wavefront_0.min(), setup.wavefront.min(), setup.wavefront_modal.min(), setup.wavefront_zonal.min())
        vmax = max(setup.wavefront_0.max(), setup.wavefront.max(), setup.wavefront_modal.max(), setup.wavefront_zonal.max())
        v = max(abs(vmin), abs(vmax))
    
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
        image1 = setup.wavefront_0.astype('float32')
        image1[image1 == 0] = np.nan
        im = axes[0, 0].imshow(image1, cmap=cmap, vmin=-v, vmax=v)
        axes[0, 0].set_title('True Wavefront')
        axes[0, 0].axis('off')
        
        image1 = setup.wavefront.astype('float32')
        image1[image1 == 0] = np.nan
        im = axes[0, 1].imshow(image1, cmap=cmap, vmin=-v, vmax=v)
        axes[0, 1].set_title('Predicted Wavefront DLWFS')
        axes[0, 1].axis('off')
        
        image1 = setup.wavefront_zonal.astype('float32')
        image1[image1 == 0] = np.nan
        im = axes[1, 0].imshow(image1, cmap=cmap, vmin=-v, vmax=v)
        axes[1, 0].set_title('Predicted Wavefront Zonal')
        axes[1, 0].axis('off')
        
        image1 = setup.wavefront_modal.astype('float32')
        image1[image1 == 0] = np.nan
        im = axes[1, 1].imshow(image1, cmap=cmap, vmin=-v, vmax=v)
        axes[1, 1].set_title('Predicted Wavefront Modal')
        axes[1, 1].axis('off')
    else:
        vmin = min(setup.wavefront_0.min(), setup.wavefront.min())
        vmax = max(setup.wavefront_0.max(), setup.wavefront.max())
        v = max(abs(vmin), abs(vmax))
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
        image1 = setup.wavefront_0.astype('float32')
        image1[image1 == 0] = np.nan
        im = axes[0].imshow(image1, cmap=cmap, vmin=-v, vmax=v)
        axes[0].set_title('True Wavefront')
        axes[0].axis('off')
        
        image1 = setup.wavefront.astype('float32')
        image1[image1 == 0] = np.nan
        im = axes[1].imshow(image1, cmap=cmap, vmin=-v, vmax=v)
        axes[1].set_title('Predicted Wavefront')
        axes[1].axis('off')
    
    cbar = fig.colorbar(im, ax=axes, location='right', shrink=0.7)
    cbar.set_label('Rad', labelpad=-0, rotation=90)
    plt.show()
    
def plot_rms_bar(RMSE, names):   
    # Make a nice plot of the RMSE wavefront estimation performances
    
    plt.figure(figsize=(8,5))
    spacing = .1
    for i in range(len(names)):
        plt.errorbar(np.arange(0, RMSE.shape[0]*2, 2)-spacing*2+spacing*i, np.mean(RMSE[:, i], axis=-1), yerr= np.std(RMSE[:, i], axis=-1), fmt='o', label=names[i])
    plt.xticks(np.arange(0, RMSE.shape[0]*2, 2))
    plt.xlabel('D/r0')
    plt.ylabel('RMS Wavefront Error')
    plt.legend(loc="upper left")
    plt.title('Wavefront Estimation Error')