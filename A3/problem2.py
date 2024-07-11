from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as tf
import torch.optim as optim

from utils import flow2rgb
from utils import rgb2gray
from utils import read_flo
from utils import read_image

np.random.seed(seed=2022)


def numpy2torch(array):
    """ Converts 3D numpy (H,W,C) ndarray to 3D PyTorch (C,H,W) tensor.

    Args:
        array: numpy array of shape (H, W, C)
    
    Returns:
        tensor: torch tensor of shape (C, H, W)
    """

    return tensor


def torch2numpy(tensor):
    """ Converts 3D PyTorch (C,H,W) tensor to 3D numpy (H,W,C) ndarray.

    Args:
        tensor: torch tensor of shape (C, H, W)
    
    Returns:
        array: numpy array of shape (H, W, C)
    """
    
    return array


def load_data(im1_filename, im2_filename, flo_filename):
    """Loading the data. Returns 4D tensors. You may want to use the provided helper functions.

    Args:
        im1_filename: path to image 1
        im2_filename: path to image 2
        flo_filename: path to the ground truth flow
    
    Returns:
        tensor1: torch tensor of shape (B, C, H, W)
        tensor2: torch tensor of shape (B, C, H, W)
        flow_gt: torch tensor of shape (B, C, H, W)
    """

    return tensor1, tensor2, flow_gt


def evaluate_flow(flow, flow_gt):
    """Evaluate the average endpoint error w.r.t the ground truth flow_gt.
    Excludes pixels, where u or v components of flow_gt have values > 1e9.

    Args:
        flow: torch tensor of shape (B, C, H, W)
        flow_gt: torch tensor of shape (B, C, H, W)
    
    Returns:
        aepe: torch tensor scalar 
    """

    return aepe


def visualize_warping_practice(im1, im2, flow_gt):
    """ Visualizes the result of warping the second image by ground truth.

    Args:
        im1: torch tensor of shape (B, C, H, W)
        im2: torch tensor of shape (B, C, H, W)
        flow_gt: torch tensor of shape (B, C, H, W)
    
    Returns:

    """

    return


def warp_image(im, flow):
    """ Warps given image according to the given optical flow.

    Args:
        im: torch tensor of shape (B, C, H, W)
        flow: torch tensor of shape (B, C, H, W)
    
    Returns:
        x_warp: torch tensor of shape (B, C, H, W)
    """
    
    return x_warp


def energy_hs(im1, im2, flow, lambda_hs):
    """ Evalutes Horn-Schunck energy function.

    Args:
        im1: torch tensor of shape (B, C, H, W)
        im2: torch tensor of shape (B, C, H, W)
        flow: torch tensor of shape (B, C, H, W)
        lambda_hs: float
    
    Returns:
        energy: torch tensor scalar
    """

    return energy


def estimate_flow(im1, im2, flow_gt, lambda_hs, learning_rate, num_iter):
    """
    Estimate flow using HS with Gradient Descent.
    Displays average endpoint error.
    Visualizes flow field.

    Args:
        im1: torch tensor of shape (B, C, H, W)
        im2: torch tensor of shape (B, C, H, W)
        flow_gt: torch tensor of shape (B, C, H, W)
        lambda_hs: float
        learning_rate: float
        num_iter: int
    
    Returns:
        aepe: torch tensor scalar
    """
    
    return aepe

# Example usage in main()
# Feel free to experiment with your code in this function
# but make sure your final submission can execute this code
def main():

    # Loading data
    im1, im2, flow_gt = load_data("data/frame10.png", "data/frame11.png", "data/flow10.flo")

    # Parameters
    lambda_hs = 0.002
    num_iter = 500

    # Warping_practice
    visualize_warping_practice(im1, im2, flow_gt)

    # Gradient descent
    learning_rate = 18
    estimate_flow(im1, im2, flow_gt, lambda_hs, learning_rate, num_iter)


if __name__ == "__main__":
    main()
