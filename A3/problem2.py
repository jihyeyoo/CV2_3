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
    tensor = torch.tensor(array.transpose(2, 0, 1))
    return tensor


def torch2numpy(tensor):
    """ Converts 3D PyTorch (C,H,W) tensor to 3D numpy (H,W,C) ndarray.

    Args:
        tensor: torch tensor of shape (C, H, W)
    
    Returns:
        array: numpy array of shape (H, W, C)
    """
    array = tensor.numpy().transpose(1, 2, 0)
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
    im1 = rgb2gray(read_image(im1_filename))
    im2 = rgb2gray(read_image(im2_filename))
    flow_gt = read_flo(flo_filename)

    tensor1 = numpy2torch(im1).unsqueeze(0).float()
    tensor2 = numpy2torch(im2).unsqueeze(0).float()
    flow_gt = numpy2torch(flow_gt).unsqueeze(0).float()

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
    valid = (flow_gt[:, 0, :, :] < 1e9) & (flow_gt[:, 1, :, :] < 1e9)
    diff = flow - flow_gt
    epe = torch.sqrt(diff[:, 0, :, :] ** 2 + diff[:, 1, :, :] ** 2)
    aepe = torch.mean(epe[valid])
    return aepe


def visualize_warping_practice(im1, im2, flow_gt):
    """ Visualizes the result of warping the second image by ground truth.

    Args:
        im1: torch tensor of shape (B, C, H, W)
        im2: torch tensor of shape (B, C, H, W)
        flow_gt: torch tensor of shape (B, C, H, W)
    
    Returns:

    """
    im2_warp = warp_image(im2, flow_gt)
    im1_np = torch2numpy(im1[0])
    im2_warp_np = torch2numpy(im2_warp[0])
    diff = np.abs(im1_np - im2_warp_np)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(im1_np, cmap='gray')
    plt.title('Image 1')
    plt.subplot(1, 3, 2)
    plt.imshow(im2_warp_np, cmap='gray')
    plt.title('Warped Image 2')
    plt.subplot(1, 3, 3)
    plt.imshow(diff, cmap='gray')
    plt.title('Difference')
    plt.show()

    return


def warp_image(im, flow):
    """ Warps given image according to the given optical flow.

    Args:
        im: torch tensor of shape (B, C, H, W)
        flow: torch tensor of shape (B, C, H, W)
    
    Returns:
        x_warp: torch tensor of shape (B, C, H, W)
    """
    B, C, H, W = im.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float().to(im.device)
    flow = flow.permute(0, 2, 3, 1)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1) + flow

    grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / (W - 1) - 1.0
    grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / (H - 1) - 1.0

    x_warp = tf.grid_sample(im, grid, align_corners=True)

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
    im2_warp = warp_image(im2, flow)
    brightness_constancy = (im2_warp - im1) ** 2
    u, v = flow[:, 0, :, :], flow[:, 1, :, :]
    u_dx, u_dy = torch.gradient(u)
    v_dx, v_dy = torch.gradient(v)
    smoothness = u_dx ** 2 + u_dy ** 2 + v_dx ** 2 + v_dy ** 2
    energy = torch.sum(brightness_constancy) + lambda_hs * torch.sum(smoothness)

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
    flow = torch.zeros_like(flow_gt, requires_grad=True)
    optimizer = optim.SGD([flow], lr=learning_rate)

    initial_aepe = evaluate_flow(flow, flow_gt)
    print(f'Initial AEPE: {initial_aepe.item()}')

    for i in range(num_iter):
        optimizer.zero_grad()
        energy = energy_hs(im1, im2, flow, lambda_hs)
        energy.backward()
        optimizer.step()

        if i % 50 == 0 or i == num_iter - 1:
            current_aepe = evaluate_flow(flow, flow_gt)
            print(f'Iteration {i + 1}/{num_iter}, AEPE: {current_aepe.item()}')

    aepe = evaluate_flow(flow, flow_gt)
    print(f'Final AEPE: {aepe.item()}')

    flow_rgb = flow2rgb(torch2numpy(flow[0].detach()))
    plt.figure()
    plt.imshow(flow_rgb)
    plt.title('Estimated Flow')
    plt.show()

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
