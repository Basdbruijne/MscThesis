"""
Deconvolution from wavefront Sensing Simulator!!

Created by Bas de Bruijne, aug 26, 2020

For questions, contact BasdBruijne@gmail.com

References:

1 - Jason D. Schmidt, "Numerical Simulation of Optical Wave Propagation with 
Examples in MATLAB"

------------------------------------------------------------------------------
EXAMPLE OF HOW TO USE THIS FILE

import DFWS_Simulator as sim

dfws = sim.DFWS(1, 6, 680, 680, 0, 0, .6)   # Initialise system with diameter 
    of 1, 6x6 shack-hartmann sensor and 680x680 pixels of imaging resolution

dfws.random_object()                        # Load Object

dfws.wavefront_kolmogorov(0.2)              # Make D/r0 = 5 turbulent phase 
                                                screen

dfws.make_psf()                             # Adjust the wavefront to the 
                                              right size and generate the
                                              point spread functions

dfws.make_image()                           # Generate the output images


Variables contained within dfws:

    - dfws.psf                              # point spread function of main
    sensor

    - dfws.psf_sh                           # point spread function of shack-
                                                hartmann wavefront sensor

    - dfws.wavefront                        # loaded wavefront

    - dfws.image                            # image that would be seen by the 
    main sensor

    - dfws.image_sh                         # image that would be seen by the 
                                              shack-hartmann wavefront sensor
    
use dir(dfws) to get a complete overview of all the available functions
 and variables
    
------------------------------------------------------------------------------
"""

import aotools
import matplotlib.pyplot as plt
import warnings
import os
import numpy
import numpy as np
from numpy import fft
from numpy.random import normal as randn
import cupy as cp
import scipy.io
from scipy.interpolate import interp2d
from scipy.ndimage import rotate
from scipy.ndimage import rotate as cpu_rotate
import imageio
import glob
from random import choice
from  aotools.functions.pupil import circle
from aotools.functions.zernike import zernikeArray
from copy import copy
from scipy import sparse
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

def to_numpy(x):
    """
    Returns a variable from either cupy or numpy to numpy.
    Cupy is the CUDA accelerated version of numpy and will
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
    c = np.real(np.fft.ifft2(np.fft.fft2(a_pad)*np.fft.fft2(b_pad)))
    c = c[int(np.floor(edge)):-int(np.ceil(edge))+1, :]
    c = c[:, int(np.floor(edge)):-int(np.ceil(edge))+1]      
    return c

def random(randomness, percentage):
    """
    Returns a number between 1-percentage and 1+percentage
    
    Input: Randomness [0 or 1]
    
    Output: Percentage [0 - 100]
    """
    if randomness:
        out = np.random.rand()*percentage/50+(1-percentage/100)
    else:
        out = 1
    return out

class DFWS:
    """
    This class (Deconvolution From wavefront Sensing) is design to simulate 
    a DFWS adaptive optics system
    
    The different attributes represent the interaction of the two components 
    with the wavefront
    """
    def __init__(self, D, N, Res, Res_SH, cupy_req = False, 
                 randomness = False, temp = .48):
        """    
        Initialization
            
        Inputs:
            
        D: Diameter of simulated telescope
            
        N: Number of subapertures in the shack-hartmann sensor. Only square 
        arrays are supported
            
        Res: Resolution of the main imaging sensor
            
        Res_SH: Resolution of the SH-sensor
            
        cupy_req [bool]: define if the CUDA accelerated numpy library (cupy) 
        should be used
            
        randomness [bool]: define if a random error should be added to all 
                           variables, this could be usefull for checking the 
                           robustness of the controllers
            
        temp: Temporary variable used for system tweaking, to be removed in 
              future version
        """        
        global np, fft, cupy, rotate, mempool, pinned_mempool, sparse
        if cupy_req:
            import cupy as np
            from cupy import fft
            from cupyx.scipy.ndimage import rotate
            from cupyx.scipy import sparse
            mempool = np.get_default_memory_pool()
            pinned_mempool = np.get_default_pinned_memory_pool()
            cupy = 1
        else:
            pass
        
        # Setup variables
        self.D = D
        self.D_SH = D/N
        self.N = int(np.ceil(N))
        self.res = int(Res)
        self.res_SH = int(Res_SH)
        self.p = 5.20e-6 #pixel spacing
        self.p_SH = 5.20e-6
        self.res_subap = int(self.res_SH/self.N)
        self.NA_SH = 1.46*.6/(2*4.2)
        self.randomness = randomness
        self.phase_screens = [None, None, None, None]
        self.cupy_req = cupy_req
        
        # Generate pupil functions
        self.pad_main = int(self.res*temp) #int(self.res*.48)
        self.pupil_func = numpy.array(circle(Res/2, Res+self.pad_main*2, 
                                             circle_centre=(0, 0), 
                                             origin='middle'), dtype = 'int8')
        
        # Inits for SH psf function   
        # a piece-wise linear defocus needs to be generated for the simulation 
        # of the shack-hartmann sensor
        
        factor = 3 # the amount of padding the wavefront needs
        res = int(int(self.res_SH/2*factor)*2+2)
        pupil = int(self.res_SH/2*factor)
        self.phase_screen_pad = int((res-2*pupil)/2)
        mask = numpy.array(circle(pupil*.9, res), dtype='int8')
        
        if res%2:
            res_phi_l = int((numpy.ceil(res*numpy.sqrt(2)/2)*2))-1
            crop = int((res_phi_l-res)/2)
        else:
            res_phi_l = int((numpy.ceil(res*numpy.sqrt(2)/2)*2))
            crop = int((res_phi_l-res)/2)
        phi_l = aotools.functions.zernike.zernikeArray([4], res_phi_l)
        phi_l = numpy.array(phi_l, dtype='float32')[0,crop:crop+res, 
                                                     crop:crop+res]
        copy_index = 2*pupil/self.N*numpy.arange(0, self.N+1)
        copy_index = numpy.floor(copy_index).astype('uint16')
      
        x2 = numpy.linspace(0, 1, 2*pupil+1)
        y2 = numpy.linspace(0, 1, 2*pupil+1)
        f = interp2d(numpy.linspace(0, 1, self.N+1), 
                     numpy.linspace(0, 1, self.N+1), 
                     phi_l[copy_index, :][:, copy_index], kind='linear')
        phi_SH = numpy.pad(f(x2, y2), ([int((res-pupil*2)/2), 
                                        int((res-pupil*2)/2)-1], 
                                       [int((res-pupil*2)/2), 
                                        int((res-pupil*2)/2)-1]))
        phi_SH = np.array(phi_SH)
        phi_SH -= np.mean(phi_SH)
        #Tune this parameter for the spacing of the spots of the SH-sensor
        phi_SH *= 927 
        
        self.pupil_func_SH_v4 = to_numpy(mask)
        self.phi_SH = to_numpy(phi_SH)

        
    def wavefront_from_zernike(self, zCoeffs = None):
        """
        Generate wavefront from Zernike Coefficients
        
        Input: zCoeffs, list of Zernike coefficients
        
        Outputs:
        
        None (variables loaded into class)
        
        Also makes psf
         
        """
        # remove previous phase screen
        self.phase_screen_Original = None
        
        # Check if valid inputs are given
        if zCoeffs is None:
            warnings.warn("No coefficiens provided, using flat wavefront.")
            self.zCoeffs =to_numpy( np.zeros([2], dtype='float16'))
            del zCoeffs
        else:
            # Convert the array to numpy/cupy
            self.zCoeffs = to_numpy(np.array(zCoeffs, dtype='float16'))  
            del zCoeffs # Delete the original to free up memory
            
        # Check if wavefront modes are already loaded or present in file
        Res = self.res#max(self.res, self.res_SH)
        if not hasattr(self, 'Zs'):
            file = ('wavefronts/' + str(self.zCoeffs.shape[0]) 
            + '_' + str(Res) + '.npy')
            try:
                self.Zs = to_numpy(np.load(file))
            except:
                if not os.path.exists('wavefronts'):
                    try:
                        os.makedirs('wavefronts')
                    except:
                        pass
                self.Zs = zernikeArray(self.zCoeffs.shape[0], Res)
                self.Zs = to_numpy(np.array(self.Zs, dtype = 'float16'))
                
                numpy.save(file, self.Zs.astype('float16'))

        # Check if loaded wavefront modes are of the right size
        if self.Zs.shape[0] != self.zCoeffs.shape[0]:
            file = 'wavefronts/' + str(self.zCoeffs.shape[0]) 
            + '_' + str(Res) + '.npy'
            try:
                self.Zs = to_numpy(np.load(file))
            except:
                pass
            if self.Zs.shape[0] != self.zCoeffs.shape[0]:
                if not os.path.exists('wavefronts'):
                    os.makedirs('wavefronts')
                self.Zs = zernikeArray(self.zCoeffs.shape[0], Res)
                self.Zs = to_numpy(np.array(self.Zs, dtype = 'float16'))
                numpy.save(file, self.Zs.astype('float16'))
            
        # Make wavefront from wavefrond modes and coefficients
        phase_screen = np.tensordot(np.array(self.zCoeffs), 
                                    np.array(self.Zs), 
                                    axes=1)#.astype('float16')
        
        # Make the wavefront zero-mean 
        pupil = np.array(circle(phase_screen.shape[0]/2, 
                                phase_screen.shape[0]), dtype='int8')
        phase_screen *= pupil
        phase_screen -= np.mean(phase_screen)
        phase_screen *= pupil

        self.phase_screen = to_numpy(phase_screen)
    
    def load_wavefront(self, Phi):
        """
         Loads a wavefront from array into the class
        
         Inputs:
        
         Phi: The provided wavefront
        
         Outputs:
        
         None (variables loaded into class)
        """
        self.phase_screen_Original = None
        self.phase_screen = to_numpy(np.array(Phi))
    
    def wavefront_from_disk(self, alpha, factor = 1, screen = 1):
        """
        Imports the wavefront given a turbulence disk from file
        
        Inputs:
        
        alpha: The angle at which the turbulence is places
        
        factor: Parameter by which the phase is multiplied for stronger 
                or weaker turbulence
        
        screen [int]: ID of turbulence simulator to be loaded. 1 are 2 are 
                      currently available
        
        Outputs:
        
        None (variables loaded into class)
        """
        self.phase_screen_Original = None
        
        # Load the phase screen from file and calculate the amount of pixels 
        # fitting in the aperture
        if self.phase_screens[screen] is None:
            temp = factor*np.array(scipy.io.loadmat('PhaseScreen'
                                                    + str(screen) 
                                                    + '.mat')['phase1'])
            self.phase_screens[screen] = temp.astype('float32')

        # Load the selected phasescreen and determine the cropped size based 
        # on the telescopes diameter
        self.phase_screen = self.phase_screens[screen]
        phase_screen_Spacing = 0.02032e-3 #[m]
        Cropped_size = int(np.floor(self.D/phase_screen_Spacing))
        
        # Given rotational angle alpha, find out where to crop the screen
        y = np.sin(alpha)*(self.phase_screen.shape[0]/2
                           -Cropped_size/2)+self.phase_screen.shape[0]/2
        x = np.cos(alpha)*(self.phase_screen.shape[0]/2
                           -Cropped_size/2)+self.phase_screen.shape[0]/2
        phase_screen_Crop = self.phase_screen[int(y-Cropped_size/2)
                                              :int(y+Cropped_size/2), 
                                              int(x-Cropped_size/2)
                                              :int(x+Cropped_size/2)]
        pupil = np.array(circle(phase_screen_Crop.shape[0]/2, 
                                phase_screen_Crop.shape[0]), dtype='int8')
        
        # Optional: Rotate and flip/mirror the wavefront randomly to 
        # increase variaty
        if True:
            temp = rotate(phase_screen_Crop, np.random.rand()*360)
            phase_screen_Crop_Rotate = temp.astype('float32')
            margin = int((phase_screen_Crop_Rotate.shape[0]
                          -phase_screen_Crop.shape[0])/2)
            crop_shape = margin+phase_screen_Crop.shape[0]
            phase_screen_Crop = phase_screen_Crop_Rotate[margin:crop_shape, 
                                                         margin:crop_shape]

            if np.random.rand() < .3:
                phase_screen_Crop = phase_screen_Crop[::-1, :]
            if np.random.rand() < .3:
                phase_screen_Crop = phase_screen_Crop[:, ::-1]
            if np.random.rand() < .3:
                phase_screen_Crop = phase_screen_Crop.T
        
        # Make the wavefront zero-mean 
        phase_screen_Crop *= pupil
        phase_screen_Crop -= np.mean(phase_screen_Crop)
        phase_screen_Crop *= pupil
        
        self.phase_screen = to_numpy(phase_screen_Crop.astype('float32'))
        
   
    def wavefront_kolmogorov_Vdovin(self, r0):
        """
        Make a phase screen based on Kolmogorov statistics, using Gleb Vdovins
        code.
        
        Inputs:
        
        r0: fried parameter. Only the ratio between r0 and D (see init) 
            is used
        
        Outputs:
        
        None (variables loaded into class)
        
       
        
        Code made by Gleb Vdovin, converted from matlab to python be me
        
        with reference to : R.G. Lane A Glindemann, C. Dainty
        
        "Simulation of a Kolmogorov phase screen"
        
        Waves in Random Media 2. 1992, pp 209-224.
        
        This code is very flow when using cupy (due to the many loops), 
        so only numpy is used here
        
       
        
        The phase screens generated are of size 2^n+1 with positive integer n. 
        It is later cropped to the required resolution. Use milk_phaseScreen 
        to reuse the original size phase screen
        """
        
        warnings.warn("It is recommended to use wavefront_kolmogorov instead")
        
        def step_down(rr,f11, f12, f21, f22):
            """
            Function used for function wavefront_Kolmogorov
            The input is values of a square section of wavefront, the output 
            generated new point in between the square section by interpolating
            and adding a random value depending on the turbulence strength
            
            Input:
                
            rr: Relative turbulence strength
            
            f11 to f22: values of pre-defined square or wavefront values
            
            Output:
                
            o_c: value of center pixel
            
            o_xx_yy: value of pixel between inputs fxx and fyy
            
            """
            rr56=rr**(5/6);
            sq47=0.6687*rr56;
            o_c = (f11+ f12 +f21 + f22)*0.25 + (0.7804*rr56)*randn();
            o_11_21=(f11+ f21)*0.5 + sq47*randn();
            o_11_12=(f11+ f12)*0.5 + sq47*randn();
            o_12_22=(f12+ f22)*0.5 + sq47*randn();
            o_21_22=(f21+ f22)*0.5 + sq47*randn();
        
            return o_c,o_11_12,o_11_21,o_12_22,o_21_22

        self.phase_screen_Original = None
        n_cycles = numpy.ceil(numpy.log(self.res)
                              /numpy.log(2)).astype('uint16')
        dr0=self.D/r0
        
        nmax=int(2**(n_cycles)+1)
        ph= numpy.zeros([nmax,nmax])
            
        a1d=numpy.sqrt(10.757)*randn()/2
        ad1=numpy.sqrt(10.757)*randn()/2
        
        ph[0, 0]=numpy.sqrt(0.7506)*randn()+a1d
        ph[nmax-1,nmax-1]=numpy.sqrt(0.7506)*randn()-a1d
        ph[nmax-1,0]=numpy.sqrt(0.7506)*randn()+ad1
        ph[0, nmax-1]=numpy.sqrt(0.7506)*randn()-ad1
        
        step=int(nmax-1)*2
    
        for it in range(0,int(n_cycles+1)):
    
            for ii1 in range(0,int(2**(it-1))):
                i1=int(1+(ii1-1)*step)
         
                for jj1 in range(0,int(2**(it-1))):
                    j1=int(1+(jj1-1)*step)
                
                    (ph[int((i1+i1+step)/2),int((j1+j1+step)/2)],
                    ph[i1,int((j1+j1+step)/2)], 
                    ph[int((i1+i1+step)/2),j1], 
                    ph[int((i1+i1+step)/2),j1+step], 
                    ph[i1+step,int((j1+j1+step)/2)])= step_down(1./(2**(it-1)),
                                                                ph[i1,j1], 
                                                                ph[i1,j1+step], 
                                                                ph[i1+step,j1], 
                                                                ph[i1+step,
                                                                   j1+step])
    
            step=int(step/2)
        ph = np.array(ph)
        self.Kolmogorov_Screen_Big = to_numpy((ph*(nmax/self.res)**(5/6)
                                               *dr0**(5/6)).astype('float32'))
        self.phase_screen = np.array((ph[0:self.res,0:self.res]*(nmax/self.res)
                                      **(5/6)*dr0**(5/6)).astype('float32'))
        self.phase_screen *= np.array(circle(self.res/2, self.res))
        self.phase_screen -= np.mean(self.phase_screen)
        self.phase_screen *= np.array(circle(self.res/2, self.res))
        self.phase_screen = to_numpy(self.phase_screen)
        
    def wavefront_kolmogorov(self, r0, switch = 32, force_cpu = True):
        """
        Make a phase screen based on Kolmogorov statistics
        
        Inputs:
        
        r0: fried parameter. Only the ratio between r0 and D (see init) 
            is used
            
        switch: The resolution at which the methods switches to upsampling
            the wavefront in blocks. Must be power of 2, decrease if the 
            pogram gives memory errors.
                        
        Outputs:
        
        None (variables loaded into class)
        
        The difference between this function and wavefront_kolmogorov, is that
        this function works with matrices and dot products rather than loops.
        This can signiciantly speed up the process, however, it requires a lot 
        more memory too so it may not work for larger phase screens.
        
        with reference to : R.G. Lane A Glindemann, C. Dainty
        
        "Simulation of a Kolmogorov phase screen"
        
        Waves in Random Media 2. 1992, pp 209-224.
        
        This code is very flow when using cupy (due to the many loops), 
        so only numpy is used here
        
       
        
        The phase screens generated are of size 2^n+1 with positive integer n. 
        It is later cropped to the required resolution. Use milk_phaseScreen 
        to reuse the original size phase screen
        """

        if not force_cpu and self.cupy_req:
            import cupy as np
            from cupyx.scipy import sparse
        else:
            import numpy as np
            from scipy import sparse
            
        if np.log(switch)/np.log(2)%1:
            raise Exception('Switch must be a power of 2')
        
        def make_stepdown_matrix(size_in):
            """
            Function used for the generation of interpolation matrices to 
            increase the size of the wavefront.
            
            Input:
                
            size_in: Size of current wavefront

            Output:
                
            A: matrix for: wavefront_new = A * wavefront_out + random
                                    
            """
                
            try:
                sA = np.load('wavefronts/KolmogorovA'+str(size_in)+'.npy')
            except:
                size_out = 2*(size_in-1)+1
                A = np.zeros([size_out**2, size_in**2], dtype='float16')
                
                B_even = np.zeros([size_out, size_in])
                for i in range(B_even.shape[1]):
                    if i < size_in-1:
                        B_even[i*2+1, i+1] = 1/2
                        B_even[i*2+1, i] = 1/2
                    B_even[i*2, i] = 1
                
                B_odd = np.zeros([size_out, size_in*2])
                for i in range(int(B_odd.shape[1]/2)):
                    if i < size_in-1:
                        B_odd[i*2+1, i+1] = 1/4
                        B_odd[i*2+1, i] = 1/4
                        B_odd[i*2+1, i+1+size_in] = 1/4
                        B_odd[i*2+1, i+size_in] = 1/4
                    B_odd[i*2, i] = 1/2
                    B_odd[i*2, i+size_in] = 1/2
                    
                
                for i in range(size_in):
                    A[i*2*size_out:(i*2+1)*size_out, 
                      i*size_in:(i+1)*size_in] = B_even
                    if i < size_in-1:
                        A[(i*2+1)*size_out:(i*2+2)*size_out, 
                          i*size_in:(i+2)*size_in] = B_odd
                        
                sA = A
                
                np.save('wavefronts/KolmogorovA'+str(size_in), sA)
            return sA
        
        def make_random_mask(size_out):
            """
            Function used for the generation of random masks to 
            increase the size of the wavefront.
            
            Input:
                
            size_in: Size of current wavefront
                        
            Output:
                
            A: matrix for: wavefront_new = B * wavefront_out + random * A
                        
            """
            try:
                sA = np.load('wavefronts/KolmogorovA2'+str(size_out)+'.npy')
            except:
                A = np.zeros([size_out, size_out], dtype='uint8')
                
                for i in range(size_out):
                    for j in range(size_out):
                        if j%2 or i%2:
                            A[i,j] = 1
                sA = A
                np.save('wavefronts/KolmogorovA2'+str(size_out), sA)
                
            return sA.astype('float32')
        
        self.phase_screen_Original = None
        n_cycles = numpy.ceil(numpy.log(self.res)
                              /numpy.log(2)).astype('uint16')
        dr0=self.D/r0
        nmax=int(2**(n_cycles)+1)
        
        # Initialize the first 4 points of the phase screen
        ph= np.zeros([2,2], dtype='float32')

        a1d=np.sqrt(10.757)*randn()/2
        ad1=np.sqrt(10.757)*randn()/2
        
        ph[0, 0]=np.sqrt(0.7506)*randn()+a1d
        ph[1,1]=np.sqrt(0.7506)*randn()-a1d
        ph[1,0]=np.sqrt(0.7506)*randn()+ad1
        ph[0,1]=np.sqrt(0.7506)*randn()-ad1
        
        # Start the upsampling
        cycle = 1
        while ph.shape[0] < self.res:
            size_in = ph.shape[0]
            size_out = 2*(size_in-1)+1

            # If the resolution of larger than switch, calculate in block 
            if size_in > switch*1.5:
                # Setup step_down and make it space
                step_down = sparse.csr_matrix(make_stepdown_matrix(switch+1)
                                              .astype('float32'))
                mask = make_random_mask(2*switch+1)
                ph_out = np.zeros([size_out, size_out])
                
                # For each block in nescecary
                for i in range(int((size_out-1)/(switch*2))):
                    for j in range(int((size_out-1)/(switch*2))):
                        # Crop the block out of the wavefront and make 
                        # it sparse
                        ph_int = sparse.csr_matrix(ph[i*switch:(i+1)*switch+1, 
                                                      j*switch:(j+1)*switch+1]
                                                   .reshape([(switch+1)**2,
                                                             1]))
                        # Perform the block product and unsparse
                        ph_out_int = step_down.dot(ph_int)
                        ph_out_int = ph_out_int.toarray().reshape([2*switch+1,
                                                                   2*switch+1])
                        ph_out_int += (0.6687*(1./(2**(cycle-1)))**(5/6) 
                              * np.random.randn(2*switch+1, 
                                                2*switch+1).astype('float16')
                              * mask)
                        # Place the calculated block in the output matrix
                        ph_out[i*2*switch:(i+1)*2*switch+1, 
                               j*2*switch:(j+1)*2*switch+1] = ph_out_int
                ph = copy(ph_out)
                
            else:
                ph = np.dot(make_stepdown_matrix(size_in), 
                            ph.reshape([ph.shape[0]*ph.shape[1], 
                                        1])).reshape([size_out,size_out])
                    
                ph += (0.6687*(1./(2**(cycle-1)))**(5/6) 
                      * np.random.randn(size_out, size_out).astype('float16')
                      * make_random_mask(size_out))
            cycle += 1
        
        # Store the big wavefront and crop it to the right size
        self.Kolmogorov_Screen_Big = to_numpy((ph*(nmax/self.res)**(5/6)
                                               *dr0**(5/6)).astype('float32'))
        self.phase_screen = np.array((ph[0:self.res,0:self.res]*(nmax/self.res)
                                      **(5/6)*dr0**(5/6)).astype('float32'))
        self.phase_screen *= np.array(circle(self.res/2, self.res))
        self.phase_screen -= np.mean(self.phase_screen)
        self.phase_screen *= np.array(circle(self.res/2, self.res))
        self.phase_screen = to_numpy(self.phase_screen)
        
    def milk_phaseScreen(self):
        """
        This function rotates and mirrors the existing phase screen randomly, 
        this way you can have a new phase screen quicker
        
        Inputs:
        
        None (loads variables from class)
        
        Outputs:
        
        None (variables loaded into class)
        """
        # Check if a phase screen is loaded into the class
        if self.phase_screen_Original is None:
            if not hasattr(self, 'phase_screen'):
                print('Please generate a phase screen first')
                raise
            self.phase_screen_Original = self.phase_screen
        
        # Generate a pupil function if nescecary
        if not hasattr(self, 'Milk_PhaseScreen_pupil'):
            size = self.phase_screen.shape[0]
            self.Milk_PhaseScreen_pupil = np.array(circle(size/2, 
                                                          size), dtype='int8')
        
        # Determine the amount by which the phasescreen needs to be cropped
        # and crop it
        marginx = np.random.randint(0, self.Kolmogorov_Screen_Big.shape[0]
                                    -self.res)
        marginy = np.random.randint(0, self.Kolmogorov_Screen_Big.shape[1]
                                    -self.res)
        temp = self.Kolmogorov_Screen_Big[int(marginx):int(marginx+self.res), 
                                          int(marginy):int(marginy+self.res)]
        self.phase_screen_Original = temp
        
        # Randomly rotate the phase screen and crop it again to size
        phase_screen_Crop_Rotate = rotate(np.array(self.phase_screen_Original), 
                                          np.random.rand()
                                          *360).astype('float32')
        margin = int((phase_screen_Crop_Rotate.shape[0]
                      -self.phase_screen_Original.shape[0])/2)
        size = margin+self.phase_screen_Original.shape[0]
        self.phase_screen = phase_screen_Crop_Rotate[margin:size, 
                                                     margin:size]
        self.phase_screen = self.phase_screen.astype('float32')
    
        # Randomly flip and mirror the phase screen
        if np.random.rand() < .3:
            self.phase_screen = self.phase_screen[::-1, :]
        if np.random.rand() < .3:
            self.phase_screen = self.phase_screen[:, ::-1]
        if np.random.rand() < .3:
            self.phase_screen = self.phase_screen.T
        
        # Multiply by the pupil and convert to numpy array
        try:
            self.phase_screen *= self.Milk_PhaseScreen_pupil
        except:
            size = self.phase_screen.shape[0]
            self.Milk_PhaseScreen_pupil = np.array(circle(size/2, 
                                                          size), dtype='int8')
            self.phase_screen *= self.Milk_PhaseScreen_pupil
        
        self.Milk_PhaseScreen_pupil = to_numpy(self.Milk_PhaseScreen_pupil)
        self.phase_screen = to_numpy(self.phase_screen)
        
    def make_psf(self, no_SH = False, no_main = False, 
                 no_main_wavefront = False):
        """
        Make the psf for the two images given the loaded phase-screen
        
        The wavefront is also generated here, which is equal to the 
        phase-screen 
        
        but converted to the correct resolution.
        
       
        
        Inputs:
        
        no_SH [bool] [optional]: skip the generation of the shack-hartmann PSF
        
        no_main [bool] [optional]: skip the generation of the main PSF
        
        no_main_wavefront [bool] [optional]: skip the generation of 
                                             the main wavefront
        
        Outputs:
        
        None (variables loaded into class)
        """
        # Make sure a phase sceen is loaded        
        if not hasattr(self, 'phase_screen'):
            warnings.warn("No phase screen found, making default wavefront")
            self.phase_screen = np.zeros([2050, 2050], dtype = 'float32')
            return
        
        # The SH sensor and main sensor require different shapes of phase-
        # screen these lines of code make sure that they are the right shape.
        if not no_SH:
            # Make the SH psf:
            self.psf_sh = numpy.zeros([self.res_SH, self.res_SH]
                                      , dtype = 'float16')
            
            # Adjust the phase screen resolution
            if self.phase_screen.shape[0] < self.phi_SH.shape[0]:
                self.phase_screen = to_numpy(self.phase_screen)
                x2 = numpy.linspace(0, 1,  self.phi_SH.shape[0])
                y2 = numpy.linspace(0, 1,  self.phi_SH.shape[0])
                f = interp2d(numpy.linspace(0, 1, self.phase_screen.shape[0])
                             , numpy.linspace(0, 1, 
                                              self.phase_screen.shape[0]),  
                             self.phase_screen, kind='cubic')
                phase_screen = numpy.array(f(x2, y2))
                self.phase_screen = np.array(self.phase_screen, 
                                             dtype = 'float32')
            elif self.phase_screen.shape[0] == self.phi_SH.shape[0]:
                phase_screen = numpy.pad(self.phase_screen.shape,
                                         [self.phase_screen_pad])
            else:
                size = self.phase_screen.shape[0]
                downsample =  np.floor(np.arange(0, size, size
                                                 /self.phi_SH.shape[0]))
                downsample = downsample.astype('uint16')
                phase_screen = self.phase_screen[downsample, ][:, downsample]
            self.phase_screen = to_numpy(self.phase_screen)
            # Make the psf and adjust the resolution again
            psf_sh = numpy.fft.fftshift(numpy.abs(numpy.fft.fft2(
                (self.pupil_func_SH_v4
                 *numpy.exp(complex(1j)*self.phi_SH+complex(1j)
                            *to_numpy(phase_screen)))))**2)
            downsample =  numpy.floor(numpy.arange(0, 
                                                   psf_sh.shape[0], 
                                                   psf_sh.shape[0]
                                                   /self.res_SH)).astype('int')
            self.psf_sh = (psf_sh[downsample, ][:, downsample]
                           +psf_sh[downsample+1, ][:, downsample+1]
                           +psf_sh[downsample+2, ][:, downsample+2])
            self.psf_sh -= numpy.min(self.psf_sh)
            self.psf_sh /= numpy.max(self.psf_sh)
            self.psf_sh = self.psf_sh.astype('float16')
            self.phase_screen = to_numpy(self.phase_screen)
            self.phase_screen_SH = to_numpy(phase_screen)
            free_gpu_memory()
        # Make the Main sensor psf
        # Adjust the phase screen resolution
        if not no_main_wavefront:
            free_gpu_memory()
            self.wavefront =numpy.zeros([self.res, self.res],dtype = 'float32')
            if self.phase_screen.shape[0] < self.wavefront.shape[0]:
               
                self.phase_screen = to_numpy(self.phase_screen)
                x2 = numpy.linspace(0, 1,  self.wavefront.shape[0])
                y2 = numpy.linspace(0, 1,  self.wavefront.shape[0])
                f = interp2d(numpy.linspace(0, 1, self.phase_screen.shape[0]), 
                             numpy.linspace(0, 1, self.phase_screen.shape[0]),  
                             self.phase_screen, kind='cubic')
                self.wavefront = numpy.array(f(x2, y2), dtype = 'float32')
                self.wavefront *= numpy.array(circle(self.wavefront.shape[0]/2, 
                                                     self.wavefront.shape[0]))
            elif self.phase_screen.shape[0] == self.wavefront.shape[0]:
                self.wavefront = self.phase_screen
            else:
                size = self.phase_screen.shape[0]
                downsample = (numpy.floor(numpy.arange(0, size, 
                                                      size/self.wavefront
                                                      .shape[0]))
                              .astype('uint16'))
                self.wavefront = to_numpy(self.phase_screen[downsample, ]
                                          [:, downsample])
            self.wavefront = to_numpy(self.wavefront)
            free_gpu_memory()
        if not no_main:
            free_gpu_memory()
            # Make the psf and adjust the resolution again
            wavefront_Pad = numpy.pad(self.wavefront, 
                                      self.pad_main)[0:self.pupil_func.shape[0], 
                                                     0:self.pupil_func.shape[0]]
            psf = fft.fftshift(np.abs(fft.fft2(
                np.array(self.pupil_func)
                *np.exp(complex(1j)*np.array(wavefront_Pad))))**2)
            psf -= np.min(psf)
            psf /= np.max(psf)
            self.psf = to_numpy(psf[self.pad_main:-self.pad_main, 
                                    self.pad_main:-self.pad_main]
                                .astype('float16'))
            free_gpu_memory()
            
    def load_object(self, Object):
        """
        Load an object and calculate resulting image
        
        Inputs:
        
        Object: square array representing a gray-scale image to be 
        used as object (source)
        
        Outputs:
        
        None (variables loaded into class)
        """
        # Convert object to grey scale if needed and normalize the image
        # if not Object is None:
        #     Object = np.array(Object)
        #     if len(Object.shape) == 3:
        #         Object = np.sum(Object, axis=-1).astype('float32')
        #     Object -= np.min(Object)
        #     Object /= np.max(Object)
        #     Object[Object<np.mean(Object[-50:, -50:])*2] = 0
        
        Object = np.array(Object)
        # Check if object is given and of right dimensions, otherwise 
        #generate a template
        if Object.shape[0] < self.res:
            pad_size = (self.res-Object.shape[0])/2
            Object = np.pad(Object, [int(np.floor(pad_size)), 
                                     int(np.ceil(pad_size))]).astype('float32')
        if Object is None or Object.shape[0] > self.res:
            warnings.warn("No object provided or object" +
                          " is of wrong dimension, using the default object.")
            Object1 = np.array(circle(int(self.res/5), self.res))
            Object2 = np.zeros([self.res, self.res])
            Object2[0:int(self.res/2), 0:int(self.res/2)] = 1; 
            Object2[int(self.res/2):, int(self.res/2):] = 1
            Object = Object1*Object2
        
        # Save the object and match resolution for the subapertures
        self.Object = to_numpy(Object.astype('float32'))
        # This extra padding is just to make the magnifications of
        # the different images match my particular setup
        Object = np.pad(Object, 210) 
        downres = np.around(np.arange(0, 
                                      Object.shape[0], 
                                      Object.shape[0]
                                      /(self.res_SH/self.N))).astype('uint16')
        self.Object_SH = Object[downres, :]
        self.Object_SH = to_numpy(self.Object_SH[:, downres].astype('float32'))
        self.object = to_numpy(self.Object)
        self.object_sh = to_numpy(self.Object_SH)
    
    def random_object(self, pref = None):
        """
        Retrieves a random object from the Objects folder
        
        Inputs:
        
        None (loads variables from class or file)
        
        Outputs:
        
        None (variables loaded into class)
        """
        # Select object randomly
        objectt_list = (glob.glob("Objects/*.png") 
                        + glob.glob("Objects/*.jpg") 
        + glob.glob("Objects/*.PNG") + glob.glob("Objects/*.JPG") 
        + glob.glob("Objects/*.JPEG") + glob.glob("Objects/*.jpeg"))
        if not pref is None:
            objectt_list = glob.glob("Objects/"+str(pref)+'*')
        objectt = choice(objectt_list)
        self.object_file = objectt
        objectt = numpy.array(imageio.imread(objectt)).astype('float32')
        if len(objectt.shape) > 2:
            objectt = numpy.sum(objectt, axis=-1) # Reduce image to grayscale
        objectt -= numpy.min(objectt)
        objectt /= numpy.max(objectt)
        # Make the background of the object true black
        objectt[objectt<numpy.mean(objectt[-50:, -50:])*2] = 0 

        # Crop object to make sure that it is square
        if objectt.shape[0] != objectt.shape[1]:
            size = min(objectt.shape)
            objectt = objectt[int((objectt.shape[0]-size)/2)
                              :int((objectt.shape[0]-size)/2)+size, 
                              int((objectt.shape[1]-size)/2)
                              :int((objectt.shape[1]-size)/2)+size]
        
        # Reduce size of object if nescecary
        if objectt.shape[0] > self.res:
            crop_index = numpy.arange(0, objectt.shape[0], 
                                      objectt.shape[0]
                                      /self.res).astype('uint64')
            objectt = objectt[crop_index,][:, crop_index]
        elif objectt.shape[0] < self.res:
            pad_size = (self.res-objectt.shape[0])/2
            temp = [int(numpy.floor(pad_size)), int(numpy.ceil(pad_size))]
            objectt = numpy.pad(objectt, temp).astype('float32')
            
        # Randomly rotate the object
        objectt_Rotate = cpu_rotate(numpy.array(objectt), 
                                    numpy.random.rand()*360).astype('float32')
        margin = int((objectt_Rotate.shape[0]-objectt.shape[0])/2)
        objectt = objectt_Rotate[margin:margin+objectt.shape[0],
                                 margin:margin
                                 +objectt.shape[0]].astype('float32')
        
        # Randomly flip and mirror the object
        if numpy.random.rand() < .3:
            objectt = objectt[::-1, :]
        if numpy.random.rand() < .3:
            objectt = objectt[:, ::-1]
        if numpy.random.rand() < .3:
            objectt = objectt.T
            
        # Load object into setup
        self.load_object(objectt)   
        return
    
    def add_noise(self, x):
        """
        This function returns the noise distribution to x as found in 
        the saved noise files.
        
        This function is used internally but can also be called saperately. 
        make_image automatically
        
        adds noise to the image.
        
       
        
        Inputs:
        
        x: Variable to be added noise to
        
        Outputs:
        
        x: Variable with added noise
        """
        # Check if a noise pattern is already loaded
        if not hasattr(self, 'noise_pattern'):
            res = max(self.res, self.res_SH)
            try:
                # Load noise data from files
                self.noise_hight = numpy.load('CCD_Noise.npy')
                self.noise_edges = numpy.load('CCD_Noise_edges.npy')
                
                # Extract a noise disctribution from the data
                noise = numpy.zeros([res*res])
                for i in range(len(noise)):
                    temp = self.noise_hight<numpy.random.rand()
                    noise[i] = self.noise_edges[numpy.max(numpy.where(temp))]
                    
                # Add some uniform noise to convert the discete disctribution 
                # into a continuous one
                self.noise_pattern = (noise.reshape([res, res]) 
                + numpy.random.rand(x.shape[0], x.shape[1])
                *numpy.abs(numpy.mean(self.noise_edges[0:-1]
                                      -self.noise_edges[1:])))
            
            except:
                warnings.warn("Noise distribution file not found," +
                              "returning poisson distribution instead")
                self.noise_pattern = numpy.random.poisson(5, [res, res])/800
        
        # Shuffle the noise sheet, if the resolution is high enough, this is 
        # equavalent to generating a new noise pattern
        # this method is better because generating the noise from the file
        # is takes longer
        numpy.random.shuffle(self.noise_pattern)
        
        # Normalize x to make sure the noise is in the right scale
        x -= np.min(x)
        x /= np.max(x)
        
        # Crop the noise sheet to the right rize and add it to the variable
        x += np.array(self.noise_pattern[0:x.shape[0], 0:x.shape[1]])
        return x
    
    def make_image(self, remove_nonfull_subapertures = True):
        """
        Generated the images for the two sensors by convolution the object
        
        and PSF.
        
        Inputs:
        
        None (loads variables from class or file)
        
        Outputs:
        
        None (variables loaded into class)
        """
        # Check if PSFs are loaded correctly
        if not (hasattr(self, 'psf_sh') and hasattr(self, 'psf')):
            warnings.warn("No psf found, making psf from default wavefront")
            self.wavefront_from_zernike()
        
        # Check if the SH mask is loaded correctly. The SH mask is a mask that 
        # dims certain areas of the shack-hartmann sensor in order to make it 
        # better represent a real shack-hartmann sensor
        if not hasattr(self, 'image_sh_Mask'):
            self.image_sh_Mask = np.zeros([self.res_SH, self.res_SH], 
                                          dtype = 'float16')
            for i in range(int(self.res_SH/2*(.85-.0)), 
                           int(self.res_SH/2*(.85+.05))):
                self.image_sh_Mask += np.array(circle(i, self.res_SH))
            self.image_sh_Mask = self.image_sh_Mask/np.max(self.image_sh_Mask)
        
        # Check if the diffraction psf is loaded correctly. The diffraction psf 
        # an psf that adds diffraction patterns to the SH-sensor in order to 
        # make it more realistic. Note that this is not a physically correct  
        # way to model this effect.
        if not hasattr(self, 'Diffraction_psf'):
            self.Diffraction_psf = np.zeros([113, 113], dtype = 'float32')
            self.Diffraction_psf[56, :] = .002
            self.Diffraction_psf[:, 56] = .002
            self.Diffraction_psf[56, 56] = 1
            
        # Make the image using the convolve2 function
        self.image = convolve2(self.Object, self.psf).astype('float32')
        self.image = self.add_noise(self.image)
        self.image -= np.min(self.image)
        self.image /= np.max(self.image)
        
        self.image_sh = convolve2(self.Object_SH, 
                                  self.psf_sh).astype('float32')
        self.image_sh -= np.min(self.image_sh)
        self.image_sh /= np.max(self.image_sh)

        self.image_sh_Masked = convolve2(self.image_sh, self.Diffraction_psf)
        self.image_sh_Masked = self.add_noise(self.image_sh_Masked)
        self.image_sh_Masked -= np.min(self.image_sh_Masked)
        self.image_sh_Masked /= np.max(self.image_sh_Masked)
        
     
        self.image_sh = to_numpy(self.image_sh_Masked)
        self.image = to_numpy(self.image)
        
        # some subapertures will not be visible in the real shack-hartmann 
        # sensor and need to be remove manually. Adapt these lines as needed 
        # for proper simulation of your specific shack-hartmann sensor.
        if self.N == 6 and remove_nonfull_subapertures:
            self.image_sh[0:int(int(680/3)), 0:int(int(680/6))] = 0
            self.image_sh[0:int(int(680/6)), 0:int(int(680/3))] = 0
            self.image_sh[0:int(int(680/3)), -int(int(680/6)):] = 0
            self.image_sh[0:int(int(680/6)), -int(int(680/3)):] = 0
            self.image_sh[-int(int(680/3)):, 0:int(int(680/6))] = 0
            self.image_sh[-int(int(680/6)):, 0:int(int(680/3))] = 0
            self.image_sh[-int(int(680/3)):, -int(int(680/6)):] = 0
            self.image_sh[-int(int(680/6)):, -int(int(680/3)):] = 0
            
        if self.N == 12:
            for i in range(12):
                for j in range(12):
                    if np.sqrt((i-5.5)**2+(j-5.5)**2) > 5:
                        self.image_sh[int(680/12*i):int(680/12*(i+1)), 
                                      int(680/12*j):int(680/12*(j+1))] = 0 
        
    def zernike_from_wavefront(self, N_zernike):
        """
        Return a fit of the first N zernike polynomials to the currently 
        loaded wavefront
        
        Inputs:
        
        N_zernike: Number of Zernike polynomials to be fitted to the 
                   loaded phase-screen
        
        Outputs:
        
        self.zCoeffs: Coefficients of the calculated Zernike polynomials
        """
        # Make sure a phase sceen is loaded        
        if not hasattr(self, 'phase_screen'):
            raise Exception('No phase screen found')
            return
        
        # Load array of Zernike Coefficients
        try:
            zernike_inv = np.load('wavefronts/' + str(N_zernike) + '_' 
                                  + str(self.phase_screen.shape[0]) 
                                  + '_inv.npy')
            # zernike_inv = cp.array(zernike_inv.astype('float16'))
            # print('Zernike wavefront imported')
        except:
            zernike = np.array(zernikeArray(N_zernike, 
                                            self.phase_screen.shape[0]), 
                               dtype = 'float32')
            zernike = np.reshape(zernike, [N_zernike,
                                           self.phase_screen.shape[0]**2])
            zernike_inv = np.linalg.pinv(zernike)
            np.save('wavefronts/'+ str(N_zernike) + '_' 
                    + str(self.phase_screen.shape[0]) 
                    + '_inv.npy', zernike_inv)
            zernike_inv = np.array(zernike_inv)
            del zernike
            # print('Zernike wavefront calculated and saved')
        
        phi = np.reshape(self.phase_screen, [self.phase_screen.shape[0]**2])
        self.zCoeffs = to_numpy(np.dot(phi,zernike_inv))
        return self.zCoeffs
        
    def wavefront_to_128(self):
        """
        Crop the wavefront to a size of 128x128, as this is used for training
        
        of the neural network 
        
        Inputs:
        
        None (loads variables from class or file)
        
        Outputs:
        
        None (variables loaded into class)
        """
        Pupil_func = numpy.array(circle(self.res_SH/2, 
                                        self.res_SH), dtype = 'uint8')
        downres2 = numpy.around(numpy.arange(0, self.res_SH, 
                                             self.res_SH/128)).astype('uint16')
        self.wavefront = self.wavefront[0:self.res, 0:self.res]
        for q in range(0, 5):
            self.wavefront -= np.mean(self.wavefront)
            self.wavefront *= Pupil_func
            
        y_curr = numpy.zeros([128, 128])
        
        for q in range(0, 5):
            y_curr += 1/5*self.wavefront[downres2+q,][:,downres2+q]
        
        return y_curr
    
    def plot_image(self, noise = True):
        """
        Make a nice plot of the images
        
        Inputs:
        
        None (loads variables from class or file)
        
        Outputs:
        
        None (variables loaded into class)
        """
        # Check if images are already generates, otherwise do so
        if not (hasattr(self, 'image') and hasattr(self, 'image_sh')):
            warnings.warn("No image found, making default image")
            self.load_object()
        
        # Find out wether to use noise or not
        if noise:
            image = self.image_noise
            image_sh = self.image_sh_noise
        else:
            image = self.image
            image_sh = self.image_sh
        
        # Plot the images
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(to_numpy(image), cmap='gray')
        ax[0].set_title('Main image')
        ax[0].axis('off')
        ax[1].imshow(to_numpy(image_sh), cmap='gray')
        ax[1].set_title('Shack-Hartmann image')
        ax[1].axis('off')
        plt.show()
        
    def plot_psf(self, noise = False):
        """
        Make a nice plot of the images
        
        Inputs:
        
        None (loads variables from class or file)
        
        Outputs:
        
        None (variables loaded into class)
        """
        # Check if images are already generates, otherwise do so
        if not (hasattr(self, 'psf') and hasattr(self, 'psf_sh_total')):
            warnings.warn("No psf found, making default psf")
            self.make_psf()
        
        # Find out wether to use noise or not
        if noise:
            image = self.psf_noise
            image_sh = self.psf_sh_total_noise
        else:
            image = self.psf
            image_sh = self.psf_sh_total
        
        # Plot the images
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(to_numpy(image), cmap='gist_heat')
        ax[0].set_title('Main psf')
        ax[0].axis('off')
        ax[1].imshow(to_numpy(image_sh), cmap='gist_heat')
        ax[1].set_title('Shack-Hartmann psf')
        ax[1].axis('off')
        plt.show()

    def shack_hartmann_misallignment(self, max_trans = 0, 
                                     max_rotation = 0, max_defocus = 0):
        """
        Add misallignments to the wavefront sensor, 
        TO BE REMOVED IN FUTURE VERSIONS
        
        The microlens array (MLA) will randomly be translated in x and 
        y a max of max_trans [pixels]
        
        The MLA will have an angle of maximally max_rotation [rad]
        
        One edge or corner of the MLA will be out of focus by max_defocus
        
       
        
        Note that these transformations assume that the MLA will stay 
        in the same position and the sensor will move.
        
        This means that the psf does not change (except with the defocus) 
        and the subaperture images will appear rotated
        """
        x = int((np.random.rand()*2-1)*max_trans)
        y = int((np.random.rand()*2-1)*max_trans)
        z_rot = (np.random.rand()*2-1)*max_rotation
        
        if x or y:
            if hasattr(self, 'image_sh'):
                image_sh = np.zeros_like(self.image_sh)
                image_sh[x*(x>0):x*(x<0)-1, y*(y>0):y*(y<0)-1] = (
                    self.image_sh[-x*(x<0):-x*(x>0)-1, -y*(y<0):-y*(y>0)-1])
                self.image_sh = image_sh
            psf_sh_total_V2 = np.zeros_like(self.psf_sh_total_V2)
            psf_sh_total_V2[x*(x>0):x*(x<0)-1, y*(y>0):y*(y<0)-1] = (
            self.psf_sh_total_V2[-x*(x<0):-x*(x>0)-1, -y*(y<0):-y*(y>0)-1])
            self.psf_sh_total_V2 = psf_sh_total_V2
        
        if max_rotation:
            if hasattr(self, 'image_sh'):
                image_sh = rotate(self.image_sh, z_rot)
                crop = int((image_sh.shape[0]-self.image_sh.shape[0])/2)
                self.image_sh = image_sh[crop:crop+self.res_SH,crop:crop
                                         +self.res_SH]
            psf_sh_total_V2 = rotate(self.psf_sh_total_V2, z_rot)
            crop = int((psf_sh_total_V2.shape[0]
                        -self.psf_sh_total_V2.shape[0])/2)
            self.psf_sh_total_V2 = psf_sh_total_V2[crop:crop+self.res_SH,
                                                   crop:crop+self.res_SH]
        
        if max_defocus:
            warnings.warn("Defocus misallignment feature not yet added")
         
        if hasattr(self, 'image_sh'):
            temp = np.random.poisson(5, [self.res_SH, self.res_SH])
            self.image_sh_noise = self.image_sh+temp/(40+np.random.rand()*120)
        self.psf_sh_total_V2 = (self.psf_sh_total_V2
        +np.random.poisson(5, [self.res_SH, self.res_SH])
        /(40+np.random.rand()*120))

    def remove_ptt(self, mode = 'phase_screen'):
        """
        Remove the piston, tip and tilt terms from given varianble 
        (usually used for wavefronts)
        
        Input:
        
        mode: 'phase_screen' or 'wavefront' or 'wavefront_0' or any variable 
        
               you may want to remove these modes from as long as theyre square
        
               and excist in the loaded class.
        
        Outputs:
        
        None (variables loaded into class)
        """
        # Check if mode exists
        if not hasattr(self, mode):
            raise Exception('Requested variable is currently not loaded')
            return
        
        # Store the shape of variable
        shape = eval('self.' + mode + '.shape[0]')
        
        # Generated matrices if not already loaded
        if (not hasattr(self, 'B') or not hasattr(self, 'B_inv') 
            or mode != self.remove_ptt_mode):
            self.B = numpy.array(zernikeArray(3, shape), 
                                 dtype = 'float32').reshape([3, shape*shape])
            self.B_inv = numpy.linalg.pinv(self.B)
        
        # Calculate the presence of each mode and remove them
        self.__dict__[mode] = to_numpy(self.__dict__[mode])
        x = numpy.dot(self.__dict__[mode].reshape([shape*shape]), 
                      numpy.array(self.B_inv))
        temp = numpy.dot(x, numpy.array(self.B)).reshape([shape, shape])
        self.__dict__[mode] -= temp
        # Store the mode, so that if a different mode is selected next time 
        # it is known that the matrices need to be calculated again
        self.remove_ptt_mode = mode
        return
