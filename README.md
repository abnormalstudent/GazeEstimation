# GazeEstimation
dlib + Pytorch pipeline for gaze estimation

`SynthesEyes.ipynb` contains step-by-step implementation of 
training environment for [SynthesEyes](https://www.cl.cam.ac.uk/research/rainbow/projects/syntheseyes/) dataset.

## Tried approaches
| Model                                  | Test Error                    | Amount of epochs                       | Model size     |
|:---------------------------------------|:-----------------------------:|:--------------------------------------:|:---------------|
|1 : GazeNet (7 conv, 1 dense, w/ BN)       |           0.91                | 70                                     | 8.7 Mb         |

## 1 : GazeNet (7 conv, 1 dense, w/ BN) 
Test error is quite big because it represents L1 loss with respect to euler's angles in screen space 
(yaw and pitch, reference point located at pupil center) 
## ToDo

• Use iris features given in the dataset 

• Implement pupil center detection using another dense layer

• Apply augmentation <s> <b> only if </b> model works bad during inference time </s> 

Just took a look at learning curves, model is too weak, so

• Implement hourglass 

