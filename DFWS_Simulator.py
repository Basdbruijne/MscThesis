#   Deconvolution from wavefront Sensing Simulator!!
#   Created by Bas de Bruijne, aug 26, 2020
#   For questions, contact BasdBruijne@gmail.com
#   References:
#   1 - Jason D. Schmidt, "Numerical Simulation of Optical Wave Propagation with Examples in MATLAB"


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
cupy = 0

def to_numpy(x):
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
    c = np.real(np.fft.ifft2(np.fft.fft2(a_pad)*np.fft.fft2(b_pad)))[int(np.floor(edge)):-int(np.ceil(edge))+1, int(np.floor(edge)):-int(np.ceil(edge))+1]      
    return c

def ft2(g, delta):
    # Modified function for fourier transform, see reference 1
    return fft.fftshift(fft.fft2(fft.fftshift(g)))*delta**2

def ift2(G, delta):
    # Modified function for inverse fourier transform, see reference 1
    N = G.shape[0]
    return fft.ifftshift(fft.ifft2(fft.ifftshift(G)))*(N*delta)**2

def myconv2(A, B, delta):
    # Modified function for convolution, see reference 1
    N = A.shape[0]
    return ift2(ft2(A, delta) * ft2(B, delta), 1/(N*delta))

def random(randomness, percentage):
    # Returns a number between 1-percentage and 1+percentage
    # Input: Randomness [0 or 1]
    # Output: Percentage [0 - 100]
    if randomness:
        out = np.random.rand()*percentage/50+(1-percentage/100)
    else:
        out = 1
    return out

def step_down(rr,f11, f12, f21, f22):
    # Function used for wavefront_Kolmogorov
    rr56=rr**(5/6);
    sq47=0.6687*rr56;
    o_c = (f11+ f12 +f21 + f22)*0.25 + (0.7804*rr56)*randn();
    o_11_21=(f11+ f21)*0.5 + sq47*randn();
    o_11_12=(f11+ f12)*0.5 + sq47*randn();
    o_12_22=(f12+ f22)*0.5 + sq47*randn();
    o_21_22=(f21+ f22)*0.5 + sq47*randn();

    return o_c,o_11_12,o_11_21,o_12_22,o_21_22

class DFWS:
    # This class (Deconvolution From wavefront Sensing) is design to simulate a DFWS adaptive optics system
    # The different attributes represent the interaction of the two components with the wavefront
    def __init__(self, D, N, Res, Res_SH, cupy_req = 0, randomness = 0, temp = .48):
        global np, fft, cupy, rotate
        if cupy_req:
            import cupy as np
            from cupy import fft
            from cupyx.scipy.ndimage import rotate
            cupy = 1
        else:
            pass
        #Initialize DFWS
        # D = Telescope diameter
        # N = Amount of rows of square Shack-Hartmann sensor
        # Res = Resolution of main/SH sensor
        # d = Pixel pitch of main/SH sensor
        
        # Setup variables
        self.D = D
        self.D_SH = D/N
        self.N = int(np.ceil(N))
        self.res = int(Res)
        self.res_SH = int(Res_SH)
        self.p = 5.20e-6
        self.p_SH = 5.20e-6
        self.res_subap = int(self.res_SH/self.N)
        self.NA_SH = 1.46*.6/(2*4.2)
        self.randomness = randomness
        self.phase_screens = [None, None, None, None]
        self.cupy_req = cupy_req
        # Generate pupil functions
        self.pad_main = int(self.res*0.48) #int(self.res*.18)
        self.pupil_func = numpy.array(aotools.functions.pupil.circle(Res/2, Res+self.pad_main*2, circle_centre=(0, 0), origin='middle'), dtype = 'int8')# <- good for no abberation
        
        # Inits for SH psf function   
        factor = 3
        res = int(int(self.res_SH/2*factor)*2+2)
        pupil = int(self.res_SH/2*factor)
        self.phase_screen_pad = int((res-2*pupil)/2)
        mask = numpy.array(aotools.functions.pupil.circle(pupil*.9, res), dtype='int8') #, circle_centre=(numpy.random.rand()*5-2.5, numpy.random.rand()*5-2.5)))
        
        if res%2:
            res_phi_l = int((numpy.ceil(res*numpy.sqrt(2)/2)*2))-1
            crop = int((res_phi_l-res)/2)
        else:
            res_phi_l = int((numpy.ceil(res*numpy.sqrt(2)/2)*2))
            crop = int((res_phi_l-res)/2)
        phi_l = numpy.array(aotools.functions.zernike.zernikeArray([4], res_phi_l), dtype='float32')[0,crop:crop+res, crop:crop+res]
        copy_index = numpy.floor(2*pupil/6*numpy.arange(0, 7)).astype('uint16')
      
        x2 = numpy.linspace(0, 1, 2*pupil+1)
        y2 = numpy.linspace(0, 1, 2*pupil+1)
        f = interp2d(numpy.linspace(0, 1, 7), numpy.linspace(0, 1, 7), phi_l[copy_index, :][:, copy_index], kind='linear')
        phi_SH = numpy.pad(f(x2, y2), ([int((res-pupil*2)/2), int((res-pupil*2)/2)-1], [int((res-pupil*2)/2), int((res-pupil*2)/2)-1]))
        phi_SH = np.array(phi_SH)
        phi_SH -= np.mean(phi_SH)
        phi_SH *= 927 #Tune this parameter for the spacing of the spots of the SH-sensor
        
        self.pupil_func_SH_v4 = mask
        self.phi_SH = to_numpy(phi_SH)

        
    def wavefront_from_zernike(self, zCoeffs = None):
        self.phase_screen_Original = None
        #Generate wavefront from Zernike Coefficients
        # Input: zCoeffs, list of Zernike coefficients
        # Also makes psf
        
        if zCoeffs is None:
            warnings.warn("No coefficiens provided, using flat wavefront.")
            self.zCoeffs =to_numpy( np.zeros([2], dtype='float16'))
            del zCoeffs
        else:
            self.zCoeffs = to_numpy(np.array(zCoeffs, dtype='float16'))  # Convert the array to numpy/cupy
            del zCoeffs # Delete the original to free up memory
            
        # Check if wavefront modes are already loaded or present in file
        Res = self.res#max(self.res, self.res_SH)
        if not hasattr(self, 'Zs'):
            file = 'wavefronts/' + str(self.zCoeffs.shape[0]) + '_' + str(Res) + '.npy'
            try:
                self.Zs = to_numpy(np.load(file))
            except:
                if not os.path.exists('wavefronts'):
                    os.makedirs('wavefronts')
                self.Zs = to_numpy(np.array(aotools.functions.zernike.zernikeArray(self.zCoeffs.shape[0], Res), dtype = 'float16'))
                numpy.save(file, self.Zs.astype('float16'))

        # Check if loaded wavefront modes are of the right size
        if self.Zs.shape[0] != self.zCoeffs.shape[0]:
            file = 'wavefronts/' + str(self.zCoeffs.shape[0]) + '_' + str(Res) + '.npy'
            try:
                self.Zs = to_numpy(np.load(file))
            except:
                pass
            if self.Zs.shape[0] != self.zCoeffs.shape[0]:
                if not os.path.exists('wavefronts'):
                    os.makedirs('wavefronts')
                self.Zs = to_numpy(np.array(aotools.functions.zernike.zernikeArray(self.zCoeffs.shape[0], Res), dtype = 'float16'))
                numpy.save(file, self.Zs.astype('float16'))
            
        # Make wavefront from wavefrond modes and coefficients
        phase_screen = np.tensordot(np.array(self.zCoeffs), np.array(self.Zs), axes=1).astype('float16')
        
        # Make the wavefront zero-mean 
        pupil = np.array(aotools.functions.pupil.circle(phase_screen.shape[0]/2, phase_screen.shape[0]), dtype='int8')
        phase_screen *= pupil
        phase_screen -= np.mean(phase_screen)
        phase_screen *= pupil

        self.phase_screen = to_numpy(phase_screen)
    
    def load_wavefront(self, Phi):
        self.phase_screen_Original = None
        self.phase_screen = np.array(Phi)
    
    def wavefront_from_disk(self, alpha, factor = 1, screen = 1):
        #Imports the wavefront given a phase screen
        
        self.phase_screen_Original = None
        # Load the phase screen from file and calculate the amount of pixels fitting in the aperture
        if self.phase_screens[screen] is None:
            self.phase_screens[screen] = factor*np.array(scipy.io.loadmat('PhaseScreen'+ str(screen) + '.mat')['phase1'], dtype='float32')

        self.phase_screen = self.phase_screens[screen]
        phase_screen_Spacing = 0.02032e-3 #[m]
        Cropped_size = int(np.floor(self.D/phase_screen_Spacing))
        
        # Given rotational angle alpha, find out where to crop the screen
        y = np.sin(alpha)*(self.phase_screen.shape[0]/2-Cropped_size/2)+self.phase_screen.shape[0]/2
        x = np.cos(alpha)*(self.phase_screen.shape[0]/2-Cropped_size/2)+self.phase_screen.shape[0]/2
        phase_screen_Crop = self.phase_screen[int(y-Cropped_size/2):int(y+Cropped_size/2), int(x-Cropped_size/2):int(x+Cropped_size/2)]
        pupil = np.array(aotools.functions.pupil.circle(phase_screen_Crop.shape[0]/2, phase_screen_Crop.shape[0]), dtype='int8')
        
        # Optional: Rotate and flip/mirror the wavefront randomly to increase variaty
        if True:
            phase_screen_Crop_Rotate = rotate(phase_screen_Crop, np.random.rand()*360).astype('float32')
            margin = int((phase_screen_Crop_Rotate.shape[0]-phase_screen_Crop.shape[0])/2)
            phase_screen_Crop = phase_screen_Crop_Rotate[margin:margin+phase_screen_Crop.shape[0], margin:margin+phase_screen_Crop.shape[0]]

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
        
        self.phase_screen = phase_screen_Crop.astype('float32')
    
    def milk_phaseScreen(self):
        #This function rotates and mirrors the existing phase screen randomly, this way you can have a new fasescreen quicker
        
        if self.phase_screen_Original is None:
            if not hasattr(self, 'phase_screen'):
                print('Please generate a phase screen first')
                raise
            self.phase_screen_Original = self.phase_screen
        
        if not hasattr(self, 'Milk_PhaseScreen_pupil'):
            self.Milk_PhaseScreen_pupil = np.array(aotools.functions.pupil.circle(self.phase_screen.shape[0]/2, self.phase_screen.shape[0]), dtype='int8')
        
        marginx = np.random.randint(0, self.Kolmogorov_Screen_Big.shape[0]-self.res)
        marginy = np.random.randint(0, self.Kolmogorov_Screen_Big.shape[1]-self.res)
        self.phase_screen_Original = self.Kolmogorov_Screen_Big[marginx:marginx+self.res, marginy:marginy+self.res]
        
        phase_screen_Crop_Rotate = rotate(np.array(self.phase_screen_Original), np.random.rand()*360).astype('float32')
        margin = int((phase_screen_Crop_Rotate.shape[0]-self.phase_screen_Original.shape[0])/2)
        self.phase_screen = phase_screen_Crop_Rotate[margin:margin+self.phase_screen_Original.shape[0], margin:margin+self.phase_screen_Original.shape[0]].astype('float32')
    
        if np.random.rand() < .3:
            self.phase_screen = self.phase_screen[::-1, :]
        if np.random.rand() < .3:
            self.phase_screen = self.phase_screen[:, ::-1]
        if np.random.rand() < .3:
            self.phase_screen = self.phase_screen.T
        
        self.phase_screen *= self.Milk_PhaseScreen_pupil
        self.phase_screen = to_numpy(self.phase_screen)
        
    def wavefront_kolmogorov(self, r0):
        #Make a phase screen based on Kolmogorov statistics
        # r0 is fried parameter. Only the ratio between r0 and D (see init) is used
        # Code made by Gleb Vdovin, converted from matlab to python be me
        # with reference to : R.G. Lane A Glindemann, C. Dainty
        # "Simulation of a Kolmogorov phase screen"
        # Waves in Random Media 2. 1992, pp 209-224.
        # This code is very flow when using cupy, so only numpy is used here
        #
        self.phase_screen_Original = None
        n_cycles = numpy.ceil(numpy.log(self.res)/numpy.log(2)).astype('uint16')
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
                
                    ph[int((i1+i1+step)/2),int((j1+j1+step)/2)],ph[i1,int((j1+j1+step)/2)], ph[int((i1+i1+step)/2),j1], ph[int((i1+i1+step)/2),j1+step], ph[i1+step,int((j1+j1+step)/2)] = step_down(1./(2**(it-1)),ph[i1,j1], ph[i1,j1+step], ph[i1+step,j1], ph[i1+step,j1+step])
    
            step=int(step/2)
        ph = np.array(ph)
        self.Kolmogorov_Screen_Big = (ph*(nmax/self.res)**(5/6)*dr0**(5/6)).astype('float32')
        self.phase_screen = np.array((ph[0:self.res,0:self.res]*(nmax/self.res)**(5/6)*dr0**(5/6)).astype('float32'))
        self.phase_screen *= np.array(aotools.functions.pupil.circle(self.res/2, self.res))
        self.phase_screen -= np.mean(self.phase_screen)
        self.phase_screen *= np.array(aotools.functions.pupil.circle(self.res/2, self.res))
        self.phase_screen = to_numpy(self.phase_screen)
        
    def make_psf(self, no_SH = False, no_main = False, no_main_wavefront = False):
        #Make the psf for the two images, new new version
        
        # Make sure a phase sceen is loaded        
        if not hasattr(self, 'phase_screen'):
            warnings.warn("No phase screen found, making default wavefront")
            self.phase_screen = np.zeros([2050, 2050], dtype = 'float32')
            return
        
        # The SH sensor and main sensor require different shapes of phase-screen
        # these lines of code make sure that they are of the right shape.
        
        if not no_SH:
            # Make the SH psf:
            self.psf_sh = numpy.zeros([self.res_SH, self.res_SH], dtype = 'float16')
            
            # Adjust the phase screen resolution
            if self.phase_screen.shape[0] < self.phi_SH.shape[0]:
                self.phase_screen = to_numpy(self.phase_screen)
                x2 = numpy.linspace(0, 1,  self.phi_SH.shape[0])
                y2 = numpy.linspace(0, 1,  self.phi_SH.shape[0])
                f = interp2d(numpy.linspace(0, 1, self.phase_screen.shape[0]), numpy.linspace(0, 1, self.phase_screen.shape[0]),  self.phase_screen, kind='linear')
                phase_screen = numpy.array(f(x2, y2))
                self.phase_screen = np.array(self.phase_screen, dtype = 'float32')
            elif self.phase_screen.shape[0] == self.phi_SH.shape[0]:
                phase_screen = numpy.pad(self.phase_screen.shape, [self.phase_screen_pad])
            else:
                downsample =  np.floor(np.arange(0, self.phase_screen.shape[0], self.phase_screen.shape[0]/self.phi_SH.shape[0])).astype('uint16')
                phase_screen = to_numpy(self.phase_screen[downsample, ][:, downsample])
            
            # Make the psf and adjust the resolution again
            psf_sh = fft.fftshift(np.abs(fft.fft2(np.array(self.pupil_func_SH_v4)*np.exp(complex(1j)*np.array(self.phi_SH)+complex(1j)*np.array(phase_screen))))**2)
            downsample =  np.floor(np.arange(0, psf_sh.shape[0], psf_sh.shape[0]/self.res_SH)).astype('int')
            self.psf_sh = (psf_sh[downsample, ][:, downsample]+psf_sh[downsample+1, ][:, downsample+1]+psf_sh[downsample+2, ][:, downsample+2])
            self.psf_sh -= np.min(self.psf_sh)
            self.psf_sh /= np.max(self.psf_sh)
            self.psf_sh = self.psf_sh.astype('float16')
            self.phase_screen = to_numpy(self.phase_screen)
            self.phase_screen_SH = to_numpy(phase_screen)
        
        # Make the Main sensor psf
        # Adjust the phase screen resolution
        if not no_main_wavefront:
            self.wavefront = numpy.zeros([self.res, self.res], dtype = 'float32')
            if self.phase_screen.shape[0] < self.wavefront.shape[0]:
                # x2 = np.linspace(0, 1,  self.wavefront.shape[0])
                # y2 = np.linspace(0, 1,  self.wavefront.shape[0])
                # f = interp2d(np.linspace(0, 1, self.phase_screen.shape[0]), np.linspace(0, 1, self.phase_screen.shape[0]), self.phase_screen, kind='linear')
                # self.wavefront = np.array(f(x2, y2))
                
                self.phase_screen = to_numpy(self.phase_screen)
                x2 = numpy.linspace(0, 1,  self.wavefront.shape[0])
                y2 = numpy.linspace(0, 1,  self.wavefront.shape[0])
                f = interp2d(numpy.linspace(0, 1, self.phase_screen.shape[0]), numpy.linspace(0, 1, self.phase_screen.shape[0]),  self.phase_screen, kind='linear')
                self.wavefront = numpy.array(f(x2, y2), dtype = 'float32')
            elif self.phase_screen.shape[0] == self.wavefront.shape[0]:
                self.wavefront = self.phase_screen
            else:
                downsample =  numpy.floor(numpy.arange(0, self.phase_screen.shape[0], self.phase_screen.shape[0]/self.wavefront.shape[0])).astype('uint16')
                self.wavefront = to_numpy(self.phase_screen[downsample, ][:, downsample])
                
        if not no_main:
            # Make the psf and adjust the resolution again
            wavefront_Pad = numpy.pad(self.wavefront, self.pad_main)[0:self.pupil_func.shape[0], 0:self.pupil_func.shape[0]]
            psf = fft.fftshift(np.abs(fft.fft2(np.array(self.pupil_func)*np.exp(complex(1j)*np.array(wavefront_Pad))))**2)
            psf -= np.min(psf)
            psf /= np.max(psf)
            self.psf = to_numpy(psf[self.pad_main:-self.pad_main, self.pad_main:-self.pad_main].astype('float16'))
            
    def load_object(self, Object = None):
        #Load an object and calculate resulting image
        
        # Check if object is given and of right dimensions, otherwise generate a template
        if Object.shape[0] < self.res:
            pad_size = (self.res-Object.shape[0])/2
            Object = np.pad(Object, [int(np.floor(pad_size)), int(np.ceil(pad_size))]).astype('float32')
        if Object is None or Object.shape[0] > self.res:
            warnings.warn("No object provided or object is of wrong dimension, using the default object.")
            Object1 = np.array(aotools.functions.pupil.circle(int(self.res/5), self.res))
            Object2 = np.zeros([self.res, self.res])
            Object2[0:int(self.res/2), 0:int(self.res/2)] = 1; Object2[int(self.res/2):, int(self.res/2):] = 1
            Object = Object1*Object2
        
        # Save the object and match resolution for the subapertures
        self.Object = Object.astype('float32')
        Object = np.pad(Object, 210) # This extra padding is just to make the magnifications of the different images match my particular setup
        # Object = np.pad(Object, 151) # This extra padding is just to make the magnifications of the different images match my particular setup
        downres = np.around(np.arange(0, Object.shape[0], Object.shape[0]/(self.res_SH/self.N))).astype('uint16')
        self.Object_SH = Object[downres, :]
        self.Object_SH = self.Object_SH[:, downres].astype('float32')
            
    def make_image(self):
        if not (hasattr(self, 'psf_sh') and hasattr(self, 'psf')):
            warnings.warn("No psf found, making psf from default wavefront")
            self.wavefront_from_zernike()
         
        if not hasattr(self, 'image_sh_Mask'):
            self.image_sh_Mask = np.zeros([self.res_SH, self.res_SH], dtype = 'float16')
            for i in range(int(self.res_SH/2*(.85-.0)), int(self.res_SH/2*(.85+.05))):
                self.image_sh_Mask += np.array(aotools.functions.pupil.circle(i, self.res_SH))
            self.image_sh_Mask = self.image_sh_Mask/np.max(self.image_sh_Mask)
        
        if not hasattr(self, 'Diffraction_psf'):
            self.Diffraction_psf = np.zeros([113, 113], dtype = 'float32')
            self.Diffraction_psf[56, :] = .002
            self.Diffraction_psf[:, 56] = .002
            self.Diffraction_psf[56, 56] = 1
            
        # Make the image using the convolve2 function
        self.image = convolve2(self.Object, self.psf).astype('float32')
        self.image -= np.min(self.image)
        self.image /= np.max(self.image)
        # self.image_noise = self.image+np.random.poisson(5, self.image_sh.shape)/500
        self.image_sh = convolve2(self.Object_SH, self.psf_sh).astype('float32')
        self.image_sh -= np.min(self.image_sh)
        self.image_sh /= np.max(self.image_sh)

        self.image_sh_Masked = convolve2(self.image_sh, self.Diffraction_psf)#*self.image_sh_Mask
        self.image_sh_Masked /= np.max(self.image_sh_Masked)
        self.image_sh_Masked += np.random.poisson(5, self.image_sh.shape)/800
     
    def zernike_from_wavefront(self, N_zernike):
        #Return a fit of the first N zernike polynomials to the currently loaded wavefront
        
        # Make sure a phase sceen is loaded        
        if not hasattr(self, 'phase_screen'):
            raise Exception('No phase screen found')
            return
        
        # Load array of Zernike Coefficients
        try:
            zernike_inv = np.load('wavefronts/' + str(N_zernike) + '_' + str(self.phase_screen.shape[0]) + '_inv.npy')
            # zernike_inv = cp.array(zernike_inv.astype('float16'))
            # print('Zernike wavefront imported')
        except:
            zernike = np.array(aotools.functions.zernike.zernikeArray(N_zernike, self.phase_screen.shape[0]), dtype = 'float32')
            zernike = np.reshape(zernike, [N_zernike, self.phase_screen.shape[0]**2])
            zernike_inv = np.linalg.pinv(zernike)
            np.save('wavefronts/'+ str(N_zernike) + '_' + str(self.phase_screen.shape[0]) + '_inv.npy', zernike_inv)
            zernike_inv = np.array(zernike_inv)
            del zernike
            # print('Zernike wavefront calculated and saved')
        
        phi = np.reshape(self.phase_screen, [self.phase_screen.shape[0]**2])
        self.zCoeffs = np.dot(phi,zernike_inv)
        return self.zCoeffs
        
    def wavefront_to_128(self):
        #Reduce the resolution of the wavefront to 128x128 pixels to decrease the size 
        #exported wavefront. 
        
        Pupil_func = numpy.array(aotools.functions.pupil.circle(340, 680), dtype = 'uint8')
        downres2 = numpy.around(numpy.arange(0, 680, 680/128)).astype('uint16')
        self.wavefront = self.wavefront[0:self.res, 0:self.res]
        for q in range(0, 5):
            self.wavefront -= np.mean(self.wavefront)
            self.wavefront *= Pupil_func
            
        y_curr = numpy.zeros([128, 128])
        
        for q in range(0, 5):
            y_curr += 1/5*self.wavefront[downres2+q,][:,downres2+q]
        
        return y_curr
    
    def plot_image(self, noise = True):
        #Make a nice plot of the images
        
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
        #Make a nice plot of the images
        
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

    def shack_hartmann_misallignment(self, max_trans = 0, max_rotation = 0, max_defocus = 0):
        #Add misallignments to the wavefront sensor
        # The microlens array (MLA) will randomly be translated in x and y a max of max_trans [pixels]
        # The MLA will have an angle of maximally max_rotation [rad]
        # One edge or corner of the MLA will be out of focus by max_defocus
        #
        # Note that these transformations assume that the MLA will stay in the same position and the sensor will move.
        # This means that the psf does not change (except with the defocus) and the subaperture images will appear rotated
        x = int((np.random.rand()*2-1)*max_trans)
        y = int((np.random.rand()*2-1)*max_trans)
        z_rot = (np.random.rand()*2-1)*max_rotation
        
        if x or y:
            if hasattr(self, 'image_sh'):
                image_sh = np.zeros_like(self.image_sh)
                image_sh[x*(x>0):x*(x<0)-1, y*(y>0):y*(y<0)-1] = self.image_sh[-x*(x<0):-x*(x>0)-1, -y*(y<0):-y*(y>0)-1]
                self.image_sh = image_sh
            psf_sh_total_V2 = np.zeros_like(self.psf_sh_total_V2)
            psf_sh_total_V2[x*(x>0):x*(x<0)-1, y*(y>0):y*(y<0)-1] = self.psf_sh_total_V2[-x*(x<0):-x*(x>0)-1, -y*(y<0):-y*(y>0)-1]
            self.psf_sh_total_V2 = psf_sh_total_V2
        
        if max_rotation:
            if hasattr(self, 'image_sh'):
                image_sh = rotate(self.image_sh, z_rot)
                crop = int((image_sh.shape[0]-self.image_sh.shape[0])/2)
                self.image_sh = image_sh[crop:crop+self.res_SH,crop:crop+self.res_SH]
            psf_sh_total_V2 = rotate(self.psf_sh_total_V2, z_rot)
            crop = int((psf_sh_total_V2.shape[0]-self.psf_sh_total_V2.shape[0])/2)
            self.psf_sh_total_V2 = psf_sh_total_V2[crop:crop+self.res_SH,crop:crop+self.res_SH]
        
        if max_defocus:
            warnings.warn("Defocus misallignment feature not yet added")
         
        if hasattr(self, 'image_sh'):
            self.image_sh_noise = self.image_sh+np.random.poisson(5, [self.res_SH, self.res_SH])/(40+np.random.rand()*120)
        self.psf_sh_total_V2 = self.psf_sh_total_V2+np.random.poisson(5, [self.res_SH, self.res_SH])/(40+np.random.rand()*120)
