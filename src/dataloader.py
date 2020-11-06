import numpy as np
import torch

from dataset import GazeOnlyDataset

def create_loader(images_path, pickles_path, 
                  batch_size = 32,
                  split_ratio = 0.9,
                  num_workers= 0,
                  pin_memory= False):
    size = len(images_path)
    assert size == len(pickles_path), \
    "Length of images {} doens't match associated data length {}".format(size, len(pickles_path))
    
    assert (split_ratio >= 0 and split_ratio < 1), "Split ration must lay in range of [0, 1)"
    
    right_border = np.int(split_ratio * size)
    
    train_images_path = images_path[: right_border]
    train_pickles_path = pickles_path[: right_border]
    
    test_images_path = images_path[right_border : ]
    test_pickles_path = pickles_path[right_border : ]
    
    train_data = GazeOnlyDataset(train_images_path, train_pickles_path)
    test_data = GazeOnlyDataset(test_images_path, test_pickles_path)
    
    train_loader = torch.utils.data.DataLoader(
        train_data,
        shuffle=True,
        batch_size=batch_size,
        num_workers=0,
        drop_last=True,
        pin_memory=pin_memory
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_data,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory
    )
    return train_loader, test_loader