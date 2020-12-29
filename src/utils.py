import matplotlib.pyplot as plt

import torch

import numpy as np

radians_to_degrees = 180.0 / np.pi


def draw_histories(train, test):
    fig = plt.figure(figsize=(8, 8))
    plt.plot(list(map(lambda x : x[0], train)), list(map(lambda x : x[1], train)), color='blue', label='Train loss')
    plt.plot(list(map(lambda x : x[0], test)), list(map(lambda x : x[1], test)), color='darkorange', label= 'Validation loss')
    plt.xlabel("Epoch")
    plt.ylabel("Mean absolute error")
    plt.ylim(0, 0.5)
    plt.legend()
    plt.savefig("Hourglass_network_loss.jpg")
    
def angular_error(a, b):
    """Calculate angular error (via cosine similarity)."""
    a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
    b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-vсерии)alues (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))

    return np.arccos(similarity) * radians_to_degrees


def mean_angular_error(a, b):
    return np.mean(angular_error(a, b))

def torch_angular_error(y_true, y_pred):
    def angles_to_unit_vectors(y):
        sin = torch.sin(y)
        cos = torch.cos(y)
        return torch.stack([
            torch.multiply(cos[:, 0], sin[:, 1]),
            sin[:, 0],
            torch.multiply(cos[:, 0], cos[:, 1]),
        ], dim=1)
    v_true = angles_to_unit_vectors(y_true)
    v_pred = angles_to_unit_vectors(y_pred)
    return torch_angular_error_from_vector(v_true, v_pred)

def torch_angular_error_from_vector(v_true, v_pred):
    v_true_norm = torch.sqrt(torch.sum(torch.square(v_true), axis=1))
    v_pred_norm = torch.sqrt(torch.sum(torch.square(v_pred), axis=1))

    sim = torch.div(torch.sum(torch.multiply(v_true, v_pred), axis=1),
                    torch.multiply(v_true_norm, v_pred_norm))

    # Floating point precision can cause sim values to be slightly outside of
    # [-1, 1] so we clip values
    sim = torch.clip(sim, -1.0 + 1e-6, 1.0 - 1e-6)

    ang = radians_to_degrees * torch.acos(sim)
    return torch.mean(ang)