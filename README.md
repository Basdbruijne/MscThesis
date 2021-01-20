This folder contains the python code used in my Thesis.

------------


    ├── NeuralNetworks	        <- Folder containting the neural networks
    |   └── Unet_set_01_12
    |       └── Batch_Size_<xx>_Loss_<yy>_Val_Loss_<zz>	 <- New Unet-ish network, trained with batch size xx, loss yy [mae] and validation loss <zz>
    |       └── Bekendam_...                             <- Bekendams network archetecture
    |       └── Hu_...                                   <- Hus network archetecture
    |
    ├── Objects                 <- Folder containing example objects to be loaded by DFWS_Simulator
    ├── Wavefronts              <- Folder containing pre-calculated matrices for DFWS_Simulator to use (will be generated automatically)
    |
    ├── DFWS_Simulator.py       <- This file contains a class DFWS (deconvolution from wavefront sensing) which has all the tools to simulate a DFWS system
    ├── dfws_solver.py          <- All the function needed to extract information from the DFWS class
    ├── dfws_simulation.py      <- An example of how to interact with DFWS_Simulator and dfws_solver
    ├── Unet_trainer_V2.py      <- The program that trains the neural networks (training data not uploaded in this folder)
    ├── PhaseScreen<i>.mat      <- The phasescreen information of the turbulence simulators in the lab, can be used by DFWS_Simulator
--------

