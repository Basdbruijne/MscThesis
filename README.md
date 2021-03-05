This project is a deconvolution from wavefront sensing (DFWS) simulation as well as the software to be implemented in a real telescope.
Telescopes have limited image quality due to atmospheric turbulence that interferes with the light. The influence of the atmospheric turbulence can be 
measured using a wavefront sensor. A DFWS telescope has two imaging sensors: the wavefront sensor and the main sensor. From the wavefront sensor, it is
possible to predict how the main sensors data is corrupted. This corruption can be undone using deconvolution.

In this project, it is shown that AI can be used for this task, independent of what the telescope is pointed at. This new way of reading out the wavefront sensor
is proven to work at least twice as well as conventional ways. A link to the final report will follow.

This folder contains the python code used in my Thesis.

All the python files start with a discription of what the file does and how the files relate to eachother.

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
 
