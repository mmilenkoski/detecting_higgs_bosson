# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import numpy as np

def compute_loss(y, tx, w):
    return np.sum(((y-np.sum((tx*w),axis = 1))**2))/(2*len(y))

