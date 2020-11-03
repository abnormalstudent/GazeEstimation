# GazeEstimation
dlib + Pytorch pipeline for gaze estimation

`SynthesEyes.ipynb` contains step-by-step implementation of 
training environment for [SyntesEyes](https://www.cl.cam.ac.uk/research/rainbow/projects/syntheseyes/) dataset.

Also, evaluation was done on traing set and it was shown, that 
MAE of current architeture of regressor-neural-network is around 0.91, which happens to be a lot, because
it measures AE between euler's angles in camera coordinate system.
