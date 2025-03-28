import numpy as np
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine


def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance, inv_comp):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """

    # To generate mask to mask out the dominant motion from image 1 to image 2

    # initialize mask
    mask = np.ones(image1.shape, dtype=bool)

    # check if inverse composition is required, get transform from image 1 to image 2 accordingly
    if inv_comp:
        M = InverseCompositionAffine(
            It=image1, It1=image2, threshold=threshold, num_iters=num_iters
        )
        M = np.linalg.inv(M)
    else:
        M = LucasKanadeAffine(
            It=image1, It1=image2, threshold=threshold, num_iters=num_iters
        )
        M = np.linalg.inv(M)

    # warp image according to chosen coordinate system for affine matrix (x, y)
    image1_warped = affine_transform(image1.T, M).T

    # get difference
    diff = np.absolute(image2 - image1_warped)

    # threshold to get mask
    mask = np.where(diff > tolerance, 1, 0)

    # Dilate mask for visibility
    mask = binary_dilation(mask).astype(mask.dtype)
    mask = binary_erosion(mask).astype(mask.dtype)

    return mask
