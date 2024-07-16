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
    # Load, convert to grayscale
    im1 = read_image(im1_filename)
    im2 = read_image(im2_filename)
    im1 = rgb2gray(im1)
    im2 = rgb2gray(im2)

    # Load gt optical flow
    flow_gt = read_flo(flo_filename)

    # Convert all loaded data to 4D PyTorch tensors
    tensor1 = numpy2torch(im1)
    tensor2 = numpy2torch(im2)
    flow_gt = numpy2torch(flow_gt)

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
    valid_flow = (flow_gt.abs() <= 1e9).all(dim=1, keepdim=True)

    valid_flow_estimated = flow[valid_flow]
    valid_flow_gt = flow_gt[valid_flow]
    
    epe = torch.norm(valid_flow_estimated - valid_flow_gt, dim=1)
    aepe = epe.mean()
    
    return aepe


def visualize_warping_practice(im1, im2, flow_gt):
    """ Visualizes the result of warping the second image by ground truth.

    Args:
        im1: torch tensor of shape (B, C, H, W)
        im2: torch tensor of shape (B, C, H, W)
        flow_gt: torch tensor of shape (B, C, H, W)
    
    Returns:

    """
    if im1.dim() == 3:
        im1 = im1.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
    if im2.dim() == 3:
        im2 = im2.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
    if flow_gt.dim() == 3:
        flow_gt = flow_gt.unsqueeze(0)  # (2, H, W) -> (1, 2, H, W)
        
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
    if im.dim() == 3:
        im = im.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
    if flow.dim() == 3:
        flow = flow.unsqueeze(0)  # (2, H, W) -> (1, 2, H, W)

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
    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]

    B, C, H, W = im1.shape
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
    grid_x = grid_x.float()
    grid_y = grid_y.float()

    new_x = grid_x + u
    new_y = grid_y + v

    # select valid one
    valid_x = (new_x >= 0) & (new_x < W)
    valid_y = (new_y >= 0) & (new_y < H)
    valid = valid_x & valid_y

    # warp valid one
    new_x = new_x[valid].view(B, -1)
    new_y = new_y[valid].view(B, -1)
    im2_warped = tf.grid_sample(im2, torch.stack((new_y, new_x), dim=-1).unsqueeze(0).unsqueeze(0), align_corners=True)

    # bright constancy E1
    E1 = ((im1 - im2_warped) ** 2).sum()

    # gradient E2
    flow_dx = tf.pad(flow[:, :, :, 1:] - flow[:, :, :, :-1], (0, 1), 'replicate')
    flow_dy = tf.pad(flow[:, :, 1:, :] - flow[:, :, :-1, :], (0, 0, 0, 1), 'replicate')

    E2 = lambda_hs * (flow_dx ** 2 + flow_dy ** 2).sum()

    # total E
    energy = E1+E2
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
    flow_rgb = flow2rgb(flow.detach().cpu().numpy())
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('estimated flow')
    plt.imshow(flow_rgb)
    plt.subplot(1, 2, 2)
    plt.title('gt flow')
    plt.imshow(flow2rgb(flow_gt.cpu().numpy()))
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
