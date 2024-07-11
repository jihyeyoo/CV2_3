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
    nllh = (x - y) ** 2 / (2 * sigma_noise ** 2)

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
            current_idx = i * width + j
            # add the connection with the node below
            if i < height - 1:
                edges.append([current_idx, (i + 1) * width + j])

            # add the connection with right node
            if j < width - 1:
                edges.append([current_idx, current_idx + 1])

    edges = np.array(edges)
    assert (edges.shape[0] == 2 * (height * width) - (height + width) and edges.shape[1] == 2)
    assert (edges.dtype in [np.int32, np.int64])
    return edges


def my_sigma():
    return 5


def my_lmbda():
    return 5


def generate_unary(noisy, denoised, alpha):
    unary = []
    unary.append(mrf_denoising_nllh(noisy, denoised, my_sigma()).flatten())

    target_img = np.empty(denoised.shape)
    target_img.fill(alpha)
    unary.append(mrf_denoising_nllh(noisy, target_img, my_sigma()).flatten())

    return np.array(unary)


def generate_pairwise(img, edges, lmbda):
    row_ind = []
    col_ind = []

    for e in edges:
        if img[e[0]] != img[e[1]]:
            row_ind.append(e[0])
            col_ind.append(e[1])
    data = np.empty(len(row_ind))
    data.fill(lmbda)
    pairwise = csr_matrix((data, (np.array(row_ind), np.array(col_ind))), shape=(img.size, img.size))
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

    denoised = init.flatten()
    noisy = noisy.flatten()
    for alpha in candidate_pixel_values:
        unary = generate_unary(noisy, denoised, alpha)
        pairwise = generate_pairwise(denoised, edges, lmbda)

        labels = gco.graphcut(unary, pairwise)

        if labels.sum() == 0:
            print(f'stopping at {alpha}')
            break

        denoised[labels == 1] = alpha

    denoised = denoised.reshape(init.shape)
    assert (np.equal(denoised.shape, init.shape).all())
    assert (denoised.dtype == init.dtype)
    return denoised


def compute_psnr(img1, img2):
    """Computes PSNR b/w img1 and img2"""

    mse = (img2 - img1) ** 2 / img1.size
    mse = mse.sum()

    v_max = 255
    psnr = 10 * np.log10(v_max ** 2 / mse)

    return psnr


def show_images(i0, i1):
    """
    Visualize estimate and ground truth in one Figure.
    Only show the area for valid gt values (>0).
    """

    # Crop images to valid ground truth area
    row, col = np.nonzero(i0)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(i0, "gray", interpolation='nearest')
    plt.subplot(1, 2, 2)
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

    # maxflow compile error: solved by doing 'pip install numpy==1.26.4'