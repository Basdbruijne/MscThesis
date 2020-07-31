##Bas' common functions:
try:
    import cupy as np
except:
    import numpy as np
    print("Cupy is not detected. Numpy is used instead. All functions should work, but not on optimal speed")
       
def convolve2(a, b):
    # Check the type of input (numpy/cupy)
    cupy = "cupy" in str(a.data)
    
    if not cupy:
        a = np.array(a)
        b = np.array(b)
    
    # Make sure matrices are square
    if a.shape[0] != a.shape[1] or b.shape[0] != b.shape[1]:
        raise Exception('Please enter square matrices')
    
    # Add padding to the matrices
    a_pad = np.pad(a, [0, b.shape[0]-1], mode='constant')
    b_pad = np.pad(b, [0, a.shape[0]-1], mode='constant')
    
    
    edge = int(np.minimum(a.shape[0], b.shape[0])/2)
    c = np.real(np.fft.ifft2(np.fft.fft2(a_pad)*np.fft.fft2(b_pad)))[edge:-edge, edge:-edge]
    
    if not cupy:
        c = np.asnumpy(c)
        
    return c

def deconvolve(image, psf, mode = 'LR', iterations=10):
    cupy = "cupy" in str(image.data)
    
    if not cupy:
        image = np.array(image)
        psf = np.array(psf)
    
    if image.shape[0] != image.shape[1] or psf.shape[0] != psf.shape[1]:
        raise Exception('Non-square images are not (yet) supported')
        
    # Initialize the variables and make sure the psf is normalized
    psf /= np.sum(psf)
    objectt = np.zeros(image.shape)
    psf_mirror = psf[::-1, ::-1]
    
    # Define the Fourier transform counterparts of the variables
    i_F = np.fft.fft2(np.pad(image, [0, psf.shape[0]-1], mode='reflect'))
    psf_F = np.fft.fft2(np.pad(psf, [0, image.shape[0]-1], mode='constant'))
    psf_mirror_F = np.fft.fft2(np.pad(psf_mirror, [image.shape[0]-1, 0], mode='constant'))
    o_F = np.fft.fft2( np.pad(objectt, [0, psf.shape[0]-1], mode='constant')) #uncomment for zero intial guess

    if mode == 'LR':
        tau = 1.5
        for n in range(iterations):
            d = (i_F-np.multiply(psf_F, o_F))*psf_mirror_F
            o_F += tau*d
            
    elif mode == 'Regularization':
        if iterations != 10:
            print('Regularization mode does not work iteratively')
            
        o_F = i_F/(psf_F+1e-4)

    else:
        raise Exception('Deconvolution method not recognized')
    
    output = np.abs(np.fft.ifft2(o_F))
    
    if not cupy:
        output = np.asnumpy(output)
        
    return output

def fact(x):
    if x%1:
        print('Input is not an integer')
    a = 1
    while x:
        a *= x
        x = x-1
    return(a)
