# GazeEstimation
dlib + Pytorch pipeline for gaze estimation

`SynthesEyes.ipynb` contains step-by-step implementation of 
training environment for [SynthesEyes](https://www.cl.cam.ac.uk/research/rainbow/projects/syntheseyes/) dataset.

Also, evaluation was done on test set and it was shown, that 
MAE of current architecture of regressor-neural-network is around 0.91, which happens to be a lot, because
it measures AE between euler's angles in camera coordinate system.

## Tried approaches
| Model                                  | Test Error                    | Amount of epochs                       | Model size     |
|:---------------------------------------|:-----------------------------:|:--------------------------------------:|:---------------|
| GazeNet (7 conv, 1 dense, w/ BN)       |           0.91                | 70                                     | 8.7 Mb         |


## ToDo

• Use iris features given in the dataset 

• Implement pupil center detection using another dense layer

• Apply augmentation <s> <b> only if </b> model works bad during inference time </s> 

