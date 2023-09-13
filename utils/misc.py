from typing import Tuple
import numpy as np
import cv2
import os
from scipy.io import savemat


def overlay(image, mask, color: Tuple[int, int, int] = (255, 0, 0), alpha = 0.5,   resize: Tuple[int, int] = (1024, 1024)):
    mask_ = np.expand_dims(mask, 2).repeat(3, axis=2)
    masked =np.ma.MaskedArray(image, mask=mask_ , fill_value=color)
    overlay = masked.filled()
    combined = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return combined


def save_mat(file, i, dir, folder_name):
    folder = os.path.join(dir, folder_name)
    if not os.path.exists(folder):
        os.mkdir(folder)
    savemat(os.path.join(folder , f'{i}.mat'), {'data': file})
    return None
