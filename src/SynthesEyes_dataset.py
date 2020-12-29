import tqdm
import pickle

import numpy as np
import torch

from cv2 import imread

def vector_to_angles(gaze_vector):
    """ 
    Input : gaze vector in camera coordinate system
    
    Returns :  yaw and pitch in camera coordinate system"""
    x, y, z = gaze_vector

    yaw = np.arctan2(x, z)
    pitch = np.arctan2(y, z)
    return yaw, pitch

class GazeOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, pickles_path, transform=None):
        self.transform = transform
        self.size = len(images_path)
        # Store images in pytorch-like shape
        image_shape = (3, 80, 120) 
        
        self.images = np.empty((self.size, *(image_shape)), dtype=np.float32)
        self.gazes = np.empty((self.size, 2), dtype=np.float32)
        self.pupil_landmarks = np.empty((self.size, 16), dtype=np.int)
        
        pickle_data = None
        for index, (image_path, data_path) in tqdm.tqdm(enumerate(zip(images_path, pickles_path))):
            with open(data_path, 'rb') as file:
                pickle_data = pickle.load(file)
            # Swap channels since opev-cv loads everything in BGR format scale it as well, becuase 
            # images values have 'uint8' type
            self.images[index] = imread(image_path)[:, :, [2, 1, 0]].transpose(2, 0, 1) / 255
            self.gazes[index] = vector_to_angles(pickle_data['look_vec'])
            self.pupil_landmarks[index] = np.array(pickle_data['ldmks']['ldmks_pupil_2d']).flatten()
            
    def __getitem__(self, index):
        if self.transform is not None:
            return (self.transform(image=self.images[index])["image"], self.gazes[index], self.pupil_landmarks[index])
        else:
            return (self.images[index], self.gazes[index], self.pupil_landmarks[index])       
        
    def __len__(self):
        return self.size