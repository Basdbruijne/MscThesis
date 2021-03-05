Ground based telescope imaging suffers from interference from the earth's atmosphere. Fluctuations in the refractive index of the air delay incoming light randomly, resulting in blurred images. A deconvolution from wavefront sensing system is an adaptive optics system that measures the modes in which the light is corrupted (i.e. the wavefront) and corrects it using a process called deconvolution. The wavefront is measured using a wavefront sensor, which consists of an array of microlenses combined with an imaging sensor. Each microlens casts an image of the object unto the imaging sensor, resulting in a collection of images that are differently aberrated depending on their location on the sensor. Conventionally, the wavefront is calculated by measuring the shifts of each microlens image and integrating these shifts over the aperture. This method, however, discards information about the higher order deformations of the microlens images. 

In this thesis, a novel method of wavefront reconstruction has been developed which makes use of artificial neural networks in order to extract this higher order information. In order to do this, each of the micro lens images in the wavefront sensor is normalized, which is done using a blind deconvolution algorithm called TIP. After the normalization, the microlens images are reduced to what they would look like if a point source was observed, instead of the object. With the influence of the object removed, an artificial neural network is used for the estimation of the wavefront. 

By using this method, the wavefront can be reconstructed with twice the turbulence strength compared to what is possible with conventional methods. Combining this method with an image deconvolution step results in a real-time image correction system that works up to $10 Hz$.

This repository contains the python code used in my Thesis. All the python files start with a discription of what the file does and how the files relate to eachother.

------------

    ├── CNN                     <- Folder contraining the neural network
    |
    ├── Objects                 <- Folder containing example objects to be loaded by DFWS_Simulator
    ├── Wavefronts              <- Folder containing pre-calculated matrices for DFWS_Simulator to use (will be generated automatically)
    |
    ├── DFWS_Simulator.py       <- This file contains a class DFWS (deconvolution from wavefront sensing) which has all the tools to simulate a DFWS system
    ├── dfws_solver.py          <- All the function needed to extract information from the DFWS class
    ├── dfws_simulation.py      <- An example of how to interact with DFWS_Simulator and dfws_solver
    ├── Unet_trainer.py         <- The program that trains the neural networks (training data not uploaded in this folder)
    
--------

Example output of dfws_solver.py:

Estimating the true object:
![alt text](https://github.com/Basdbruijne/MscThesis/blob/master/Reconstruction.png?raw=true)

Estimating the wavefront using deep learning wavefront sensing:

![alt text](https://github.com/Basdbruijne/MscThesis/blob/master/Wavefront.png?raw=true)
 
