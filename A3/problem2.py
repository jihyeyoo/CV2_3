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
    H, W, C = array.shape
    new_array = np.empty((C, H, W))

    for i in range(H):
        new_array[:, i, :] = array[i, :, :].T

    tensor = torch.from_numpy(new_array)

    return tensor


def torch2numpy(tensor):
    """ Converts 3D PyTorch (C,H,W) tensor to 3D numpy (H,W,C) ndarray.

    Args:
        tensor: torch tensor of shape (C, H, W)

    Returns:
        array: numpy array of shape (H, W, C)
    """
    C, H, W = tensor.shape
    array = np.empty((H, W, C))
    for i in range(H):
        array[i, :, :] = tensor[:, i, :].T

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
    flo = read_flo(flo_filename)

    tensor1 = numpy2torch(im1)
    tensor1 = tensor1.reshape((1, tensor1.shape[0], tensor1.shape[1], tensor1.shape[2]))

    tensor2 = numpy2torch(im2)
    tensor2 = tensor2.reshape((1, tensor2.shape[0], tensor2.shape[1], tensor2.shape[2]))

    flow_gt = numpy2torch(flo)
    flow_gt = flow_gt.reshape((1, flow_gt.shape[0], flow_gt.shape[1], flow_gt.shape[2]))

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

    # find indices of valid values.
    valid_idx = (flow_gt <= 1e9)
    valid_idx = valid_idx[0, 0] & valid_idx[0, 1]

    # calculate EPE of each pixel and then AEPE.
    dist = torch.pow(flow[0, :, valid_idx] - flow_gt[0, :, valid_idx], 2)
    dist = dist[0] + dist[1]

    aepe = torch.sqrt(dist).sum() / valid_idx.sum()

    return aepe


def visualize_warping_practice(im1, im2, flow_gt):
    """ Visualizes the result of warping the second image by ground truth.

    Args:
        im1: torch tensor of shape (B, C, H, W)
        im2: torch tensor of shape (B, C, H, W)
        flow_gt: torch tensor of shape (B, C, H, W)

    Returns:

    """

    im2_w = warp_image(im2, flow_gt)

    im1_np = torch2numpy(im1[0])
    im2_w_np = torch2numpy(im2_w[0])
    diff = im2_w_np - im1_np

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(im1_np, "gray", interpolation='nearest')
    plt.title("Image 1")

    plt.subplot(1, 3, 2)
    plt.imshow(im2_w_np, "gray", interpolation='nearest')
    plt.title("Image 2 (Warped)")

    plt.subplot(1, 3, 3)
    plt.imshow(diff, "gray", interpolation='nearest')
    plt.title("Difference")

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
    B, C, H, W = flow.shape
    flow_reshaped = torch.zeros((B, H, W, C), dtype=flow.dtype)

    for i in range(H):
        flow_reshaped[0, i, :, :] = flow[0, :, i, :].T

    # add its x, y indices to each flow values
    x_offset = torch.tile(torch.arange(W), (H, 1))
    y_offset = torch.tile(torch.arange(H).reshape(H, 1), (1, W))
    flow_reshaped[flow_reshaped > 1e9] = 0
    flow_reshaped[0, :, :, 0] += x_offset
    flow_reshaped[0, :, :, 1] += y_offset

    # normalize flow so the value ranges from [-1, 1]
    flow_reshaped[0, :, :, 0] /= W - 1
    flow_reshaped[0, :, :, 0] = (flow_reshaped[0, :, :, 0] - 0.5) * 2

    flow_reshaped[0, :, :, 1] /= H - 1
    flow_reshaped[0, :, :, 1] = (flow_reshaped[0, :, :, 1] - 0.5) * 2

    x_warp = tf.grid_sample(im.float(), flow_reshaped.float())

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

    flow_h, flow_v = flow[0, 0], flow[0, 1]

    # gradients of horizontal/vertical flow
    hor_grad_y, hor_grad_x = torch.gradient(flow_h)
    ver_grad_y, ver_grad_x = torch.gradient(flow_v)

    prior_term = hor_grad_x ** 2 + hor_grad_y ** 2 + ver_grad_x ** 2 + ver_grad_y ** 2
    im2_w = warp_image(im2, flow)
    energy = torch.pow(im2_w - im1, 2) + lambda_hs * prior_term
    energy = energy.sum()

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

    flow_estimated = torch.zeros(flow_gt.shape, requires_grad=True)
    aepe_before = evaluate_flow(flow_estimated.detach(), flow_gt)
    print(f'AEPE before: {aepe_before}\n')

    optimizer = optim.SGD([flow_estimated], lr=learning_rate)
    for i in range(num_iter):
        optimizer.zero_grad()
        energy = energy_hs(im1, im2, flow_estimated, lambda_hs)
        energy.backward()
        optimizer.step()

        # if i % 50 == 0:
        #     aepe = evaluate_flow(flow_estimated.detach(), flow_gt)
        #     print(f'intermediate AEPE({i}): {aepe}, energy: {energy}\n')

    aepe = evaluate_flow(flow_estimated.detach(), flow_gt)
    print(f'AEPE after: {aepe}')

    flow_rgb = flow2rgb(torch2numpy(flow_estimated[0].detach()))

    plt.figure()
    plt.imshow(flow_rgb)
    plt.title("Estimated Flow")
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
