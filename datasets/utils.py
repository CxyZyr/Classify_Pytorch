import random
import numpy as np
import cv2

from PIL import Image
from skimage import transform
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from .augmentation import *
import matplotlib.pyplot as plt


def HeadPipeline(info,test_mode,data_aug,input_size):
    path = info
    trans = [RandomBlur(p=0.2),
            RandomCropEdge(p=0.3),
            RandomRotation(p=0.3),
            RandomLight(p=0.3),
            RandomHSV(p=0.3),
            RandomGray(p=0.1)]
    image = cv2.imread(path)
    if image is None:
        raise OSError('{} is not found'.format(path))
    if data_aug:
        for tran in trans:
            image = tran(image)
    image = cv2.resize(image, (input_size, input_size))
    image = np.array(image)
    # image = image[:, :, ::-1]
    image = (image - [104, 117, 123]) * 0.0078125
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    if not test_mode and random.random() > 0.5:
        image = np.flip(image, axis=2).copy()
    return image


