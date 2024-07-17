'''
Authors:
Group #13
- Jinjoo Ha (jinjoo.ha@stud.tu-darmstadt.de)
- Jihye Yoo (gudong0918@ewhain.net)
- Hyewon Joo (joohw1004@ewhain.net)
'''

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
    nllh = 0.5 * ((x - y) ** 2) / (sigma_noise ** 2)
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
            if i < height - 1:
                edges.append([i * width + j, (i + 1) * width + j])
            if j < width - 1:
                edges.append([i * width + j, i * width + (j + 1)])
    edges = np.array(edges, dtype=np.int64)
    assert (edges.shape[0] == 2 * (height*width) - (height+width) and edges.shape[1] == 2)
    assert (edges.dtype in [np.int32, np.int64])
    return edges

def my_sigma():
    return 5

def my_lmbda():
    return 5

def generate_unary(noisy, denoised, alpha, s):
    target_img = np.full(denoised.shape, alpha)
    unary = np.stack([
        mrf_denoising_nllh(noisy, denoised, s).flatten(),
        mrf_denoising_nllh(noisy, target_img, s).flatten()
    ])
    return unary

def generate_pairwise(img, edges, lmbda):
    row_ind = []
    col_ind = []

    for e in edges:
        if img[e[0]] != img[e[1]]:
            row_ind.append(e[0])
            col_ind.append(e[1])
    data = np.full(len(row_ind), lmbda)
    pairwise = csr_matrix((data, (row_ind, col_ind)), shape=(img.size, img.size))
    return pairwise

def alpha_expansion(noisy, init, edges, candidate_pixel_values, s, lmbda):
    """ Run alpha-expansion algorithm.

      Args:
        noisy: Given noisy grayscale image.
        init: Image for denoising initilisation
        edges: Given neighboor of MRF.
        candidate_pixel_values: Set of labels to consider
        s: sigma for likelihood estimation
        lmbda: Regularization parameter for Potts model.

      Runs through the set of candidates and iteratively expands a label.
      If there have been recorded changes, re-run through the complete set of candidates.
      Stops, if there are no changes in the labelling.

      Returns:
        A `nd.array` of type `int32`. Assigned labels minimizing the costs.
    """
    height, width = noisy.shape
    denoised = init.flatten()
    noisy_flat = noisy.flatten()

    for alpha in candidate_pixel_values:
        unary = generate_unary(noisy_flat, denoised, alpha, s)
        pairwise = generate_pairwise(denoised, edges, lmbda)

        labels = gco.graphcut(unary, pairwise)

        if labels.sum() == 0:
            break

        denoised[labels == 1] = alpha

    denoised = denoised.reshape((height, width))
    assert (np.equal(denoised.shape, init.shape).all())
    assert (denoised.dtype == init.dtype)
    return denoised

def compute_psnr(img1, img2):
    """Computes PSNR b/w img1 and img2"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    pixel_max = 255.0
    psnr = 10 * math.log10((pixel_max ** 2) / mse)
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