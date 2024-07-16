import math
import gco
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

np.random.seed(seed=2022)

def mrf_denoising_nllh(x, y, sigma_noise):
    """Elementwise negative log likelihood.

      Args:
        x: candidate denoised image
        y: noisy image
        sigma_noise: noise level for Gaussian noise

      Returns:
        A `nd.array` with dtype `float32/float64`.
    """
    nllh = (1 / (2 * sigma_noise**2)) * (x - y)**2

    assert (nllh.dtype in [np.float32, np.float64])
    return nllh

def edges4connected(height, width):
    """Construct edges for 4-connected neighborhood MRF.
    The output representation is such that output[i] specifies two indices
    of connected nodes in an MRF stored with row-major ordering.

      Args:
        height, width: size of the MRF.

      Returns:
        A `nd.array` with dtype `int32/int64` of size |E| x 2.
    """
    edges = []
    for i in range(height):
        for j in range(width):
            idx = i * width + j
            # right
            if j + 1 < width: # if the current pixel is not the last column
                edges.append([idx, idx + 1])  
            # bottom
            if i + 1 < height:
                edges.append([idx, idx + width])  
    edges = np.array(edges, dtype=np.int64)

    assert (edges.shape[0] == 2 * (height*width) - (height+width) and edges.shape[1] == 2)
    assert (edges.dtype in [np.int32, np.int64])
    return edges

def my_sigma():
    return 5

def my_lmbda():
    return 5

def generate_unary_and_pairwise(noisy, denoised, alpha, edges, lmbda):
    edge_idx = np.array(edges)
    different_pixels = denoised[edge_idx[:, 0]] != denoised[edge_idx[:, 1]]
    pairwise_row = edge_idx[different_pixels, 0]
    pairwise_col = edge_idx[different_pixels, 1]

    # pairwise
    data = np.full(len(pairwise_row), lmbda)
    pairwise = csr_matrix((data, (pairwise_row, pairwise_col)), shape=(denoised.size, denoised.size))

    # unary (2xN)
    unary = []
    unary.append(mrf_denoising_nllh(noisy, denoised, my_sigma()).flatten())

    target_img = np.full_like(denoised, alpha)
    unary.append(mrf_denoising_nllh(noisy, target_img, my_sigma()).flatten())

    return np.array(unary), pairwise


def alpha_expansion(noisy, init, edges, candidate_pixel_values, s, lmbda):
    denoised = init.flatten()
    noisy = noisy.flatten()
    
    for alpha in candidate_pixel_values:
        unary, pairwise = generate_unary_and_pairwise(noisy, denoised, alpha, edges, lmbda)
        labels = gco.graphcut(unary, pairwise)

        if labels.sum() == 0:
            break

        denoised[labels == 1] = alpha

    denoised = denoised.reshape(init.shape)
    assert (np.equal(denoised.shape, init.shape).all())
    assert (denoised.dtype == init.dtype)
    return denoised


def compute_psnr(img1, img2):
    """Computes PSNR between img1 and img2"""
    vmax = 255.0
    mse = np.sum((img1 - img2) ** 2) / (img1.size)
    
    if mse == 0:
        return float('inf')
    
    psnr = 10 * np.log10((vmax ** 2) / mse)
    return psnr

def show_images(i0, i1):
    """
    Visualize estimate and ground truth in one Figure.
    Only show the area for valid gt values (>0).
    """

    # Crop images to valid ground truth area
    row, col = np.nonzero(i0)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(i0, "gray", interpolation='nearest')
    plt.subplot(1,2,2)
    plt.imshow(i1, "gray", interpolation='nearest')
    plt.show()

# Example usage in main()
# Feel free to experiment with your code in this function
# but make sure your final submission can execute this code
if __name__ == '__main__':
    # Read images
    noisy = ((255 * plt.imread('data/la-noisy.png')).squeeze().astype(np.int32)).astype(np.float32)
    gt = (255 * plt.imread('data/la.png')).astype(np.int32)
    
    lmbda = my_lmbda()
    s = my_sigma()

    # Create 4 connected edge neighborhood
    edges = edges4connected(noisy.shape[0], noisy.shape[1])

    # Candidate search range
    labels = np.arange(0, 255)

    # Graph cuts with random initialization
    random_init = np.random.randint(low=0, high=255, size=noisy.shape)
    estimated = alpha_expansion(noisy, random_init, edges, labels, s, lmbda)
    show_images(noisy, estimated)
    psnr_before = compute_psnr(noisy, gt)
    psnr_after = compute_psnr(estimated, gt)
    print(psnr_before, psnr_after)