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
    """ 3D numpy (H,W,C) ndarray -> 3D PyTorch (C,H,W) tensor.

    Args:
        array: numpy array of shape (H, W, C)
    
    Returns:
        tensor: torch tensor of shape (C, H, W)
    """
    tensor=torch.tensor(array.transpose(2, 0, 1))
    return tensor


def torch2numpy(tensor):
    """ 3D PyTorch (C,H,W) tensor -> 3D numpy (H,W,C) ndarray.

    Args:
        tensor: torch tensor of shape (C, H, W)
    
    Returns:
        array: numpy array of shape (H, W, C)
    """
    array=tensor.numpy().transpose(1, 2, 0)
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
   
        
    I2_w = warp_image(im2, flow_gt)

    # tensor->numpy
    I1_np = torch2numpy(im1.squeeze(0))
    I2_w_np = torch2numpy(I2_w.squeeze(0))

    diff = np.abs(I1_np - I2_w_np)

    # visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(1, 3, 1)
    plt.title('I1')
    plt.imshow(I1_np, cmap='gray')
    
    plt.subplot(1, 3, 2)
    plt.title('I2_Warped')
    plt.imshow(I2_w_np, cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title('Difference')
    plt.imshow(diff, cmap='gray')
    
    plt.show()
    return


def warp_image(im, flow):
    """ Warps given image(im) according to the given optical flow.

    Args:
        im: torch tensor of shape (B, C, H, W)
        flow: torch tensor of shape (B, C, H, W)
    
    Returns:
        x_warp: torch tensor of shape (B, C, H, W)
    """
    B, C, H, W = im.size()
    
    # make grid (store warped image)
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1).to(flow.device)  # (B, H, W, 2)
    grid = grid + flow.permute(0, 2, 3, 1) #(B, C, H, W)->(B, H, W, C)

    grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0

    x_warp = tf.grid_sample(im, grid, mode='bilinear', padding_mode='border', align_corners=True)

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
    u = flow[0, 0, :, :]
    v = flow[0, 1, :, :]

    im2_warp = warp_image(im2, flow)
    brightness_constancy = (im2_warp - im1) ** 2
    
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
    B, C, H, W = im1.shape

    # flow initailiztion (zero tensor)
    flow = torch.zeros(B, 2, H, W, requires_grad=True)

    optimizer = optim.SGD([flow], lr=learning_rate)

    for i in range(num_iter):
        optimizer.zero_grad()
        energy = energy_hs(im1, im2, flow, lambda_hs)
        energy.backward()
        optimizer.step()

        if i % 50 == 0 or i == num_iter - 1:
            aepe = evaluate_flow(flow, flow_gt)
            print(f'Iteration {i+1}/{num_iter}, Energy: {energy.item()}, AEPE: {aepe.item()}')

    # final AEPE
    aepe = evaluate_flow(flow, flow_gt)

    # visualize flow
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
