from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rice
from skimage.metrics import structural_similarity as ssim

try:
    # Package scikit-rmt installed
    from skrmt.ensemble import WishartEnsemble
    from skrmt.ensemble.spectral_law import MarchenkoPasturDistribution
    from skrmt.ensemble.utils import get_bins_centers_and_contour
except ModuleNotFoundError:
    # Running within `simulations` directory
    import sys
    sys.path.append("..")
    from skrmt.ensemble import WishartEnsemble
    from skrmt.ensemble.spectral_law import MarchenkoPasturDistribution
    from skrmt.ensemble.utils import get_bins_centers_and_contour


def norm_img_0_255(img: np.ndarray):
    """Normalizes a 2D image (numpy array) pixel intensities between 0 and 255.
    The value 0 corresponds to the least intense pixel, and 255 to the pixel with
    the highest intensity.
    """
    min_val = np.min(img)
    max_val = np.max(img)
    return 255 * (img - min_val) / (max_val - min_val)


def denoise_local_mppca(X: np.ndarray, sigma: float) -> np.ndarray:
    """Denoises local region represented by X using Marchenko-Pastur PCA
    denoising algorithm.

    Args:
        X (np.dnarray): M times N matrix representing a local region to denoise.
            M is the number of different measurements, and N the number of pixels
            in the region to denoise.
        sigma (float): Marchenko-Pastur parameter, that approximates the level of noise.

    Returns:
        np.ndarray: denoised matrix X.
    """
    # M := number of measurements
    # N := number of pixels
    (M, N) = X.shape
    wre = WishartEnsemble(beta=1, p=M, n=N, sigma=sigma)
    wre.matrix = X

    # Principal Component Analysis (PCA) via SVD
    U, S, Vh = np.linalg.svd((1/np.sqrt(N)) * wre.matrix, full_matrices=False)

    # nullifying noisy eigenvalues
    denoised_S = np.where(S <= np.sqrt(wre.lambda_plus), 0, S)

    # reconstructing denoised X
    denoised_X = np.sqrt(N) * np.dot(U * denoised_S, Vh)

    return denoised_X


def denoise_mppca_by_rows(snapshots: np.ndarray, sigma: float) -> np.ndarray:
    """Denoises a set of images (snapshots) which are all noisy measurements
    of the same region of interest. For example, a set of MRI images focused on
    the same brain region.
    This function rearranges the different measurements and denoises each row
    at a time.

    Args:
        snapshots (np.ndarray): set of images which are all noisy measurements
            of the same region of interest. 3D dimensional numpy array of size
            (N images, height, width).
        sigma (float): Marchenko-Pastur parameter, that approximates the level of noise.

    Returns:
        np.ndarray: denoised snapshots.
    """
    n_snapshots, img_height, img_width = snapshots.shape
    print(f"Denoising {n_snapshots} snapshots (sigma = {sigma}).")

    denoised_snapshots = np.zeros_like(snapshots)
    for i in range(img_height):  # this can be parallelized
        rows = snapshots[:,i,:]
        denoised_X = denoise_local_mppca(X=rows, sigma=sigma)
        denoised_snapshots[:,i,:] = denoised_X
    
    return denoised_snapshots


def denoise_mppca(snapshots: np.ndarray, sigma: float, window_size: int = 16) -> np.ndarray:
    """Performns PCA-based image denoising using the Marchenko-Pastur law.
    Denoises a set of images (snapshots) which are all noisy measurements
    of the same region of interest. For example, a set of MRI images focused on
    the same brain region.
    This function iterates over patches or windows of the image of size
    `window_size x window_size`. The denoised pixels contained in several windows
    are averaged.

    Args:
        snapshots (np.ndarray): set of images which are all noisy measurements
            of the same region of interest. 3D dimensional numpy array of size
            (N images, height, width).
        sigma (float): Marchenko-Pastur parameter, that approximates the level of noise.

    Returns:
        np.ndarray: denoised snapshots.
    """
    n_snapshots, img_height, img_width = snapshots.shape
    print(f"Denoising {n_snapshots} snapshots of size {img_height}x{img_width} (sigma = {sigma}).")

    denoised_snapshots = np.zeros_like(snapshots)

    for i in range(img_height - window_size + 1):
        for j in range(img_width - window_size + 1):
            locally_denoised_sss = np.zeros_like(snapshots)
            local_window = snapshots[:,i:i+window_size,j:j+window_size]
            local_X = np.reshape(local_window, (n_snapshots, window_size**2))

            denoised_local_X = denoise_local_mppca(X=local_X, sigma=sigma)

            denoised_local_window = np.reshape(denoised_local_X, (n_snapshots, window_size, window_size))
            locally_denoised_sss[:,i:i+window_size,j:j+window_size] = denoised_local_window

            denoised_snapshots += locally_denoised_sss

    for i in range(img_height):
        for j in range(img_width):
            # average by the number of times a pixel has been denoised
            average_by = min(i+1, img_height-i, window_size) * min(j+1, img_width-j, window_size)
            denoised_snapshots[:,i,j] /= average_by

    return denoised_snapshots


def apply_foreground_masks(snapshots: np.ndarray, fg_masks: np.ndarray) -> np.ndarray:
    assert snapshots.shape == fg_masks.shape

    mask_snapshots = []
    for idx, den_ss in enumerate(snapshots):
        fg_m = fg_masks[idx]
        mask_snapshots.append(fg_m * den_ss)
    mask_snapshots = np.asarray(mask_snapshots)

    assert mask_snapshots.shape == snapshots.shape
    return mask_snapshots

def normalize_imgs_0_255(snapshots: np.ndarray) -> np.ndarray:
    norm_snapshots = []
    for ss in snapshots:
        norm_snapshots.append(norm_img_0_255(img=ss))
    return np.asarray(norm_snapshots)


def plot_local_mppca(X: np.ndarray, sigma: float) -> None:
    """Plots the local Marchenko-Pastur PCA denoising process. This function
    represents the eigenvalue distirbution of the local region being denoised.
    The eigenvalues greater than `lambda_max` are the signal-carrying eigenvalues,
    and the smaller ones are considered pure noise, so they are removed in the
    denoising process.

    Args:
        X (np.dnarray): M times N matrix representing a local region to denoise.
            M is the number of different measurements, and N the number of pixels
            in the region to denoise.
        sigma (float): Marchenko-Pastur parameter, that approximates the level of noise.
    """
    wre = WishartEnsemble(beta=1, p=X.shape[0], n=X.shape[1], sigma=sigma)
    wre.matrix = (1/wre.n) * np.matmul(X, X.T)
    eigvals = wre.eigvals()

    # absolute histogram for all eigenvalues
    global_interval = (0.01, eigvals.max())
    wre.plot_eigval_hist(bins=40, interval=global_interval, density=False, normalize=False)

    # histogram for noise-related eigenvalues
    noise_interval = (wre.lambda_minus, wre.lambda_plus)
    wre.plot_eigval_hist(bins=40, interval=noise_interval, density=False, normalize=False)

    # computing normalized histogram
    observed, bin_edges = wre.eigval_hist(
        bins=40, interval=noise_interval, density=True, normalize=False
    )
    centers = np.asarray(get_bins_centers_and_contour(bin_edges))

    mpd = MarchenkoPasturDistribution(ratio=wre.ratio, sigma=sigma)

    height = mpd.pdf(centers)
    width = bin_edges[1]-bin_edges[0]
    # plotting normalized histogram
    plt.bar(bin_edges[:-1], observed, width=width, align='edge')
    # pdf
    plt.plot(centers, height, color='red', linewidth=2)
    plt.show()


def snr(ref_img: np.ndarray, test_img: np.ndarray) -> float:
    """
    Computes Signal-to-Noise ratio measured in dB

    Input:
        - ref_img (np.ndarray): 2D numpy array.
        - test_img (np.ndarray): 2D numpy array.
    
    Returns:
        float: signal-to-noise ratio.
    
    Reference:
        - "SNR, PSNR, RMSE, MAE". Daniel Sage at the Biomedical Image Group, EPFL, Switzerland.
            http://bigwww.epfl.ch/sage/soft/snr/
        - D. Sage, M. Unser, "Teaching Image-Processing Programming in Java".
            IEEE Signal Processing Magazine, vol. 20, no. 6, pp. 43-52, November 2003.
            http://bigwww.epfl.ch/publications/sage0303.html
    """
    # checking they are 2D images
    assert len(ref_img.shape) == 2
    # checking both images have the same size
    assert ref_img.shape == test_img.shape

    numerator = np.sum(ref_img**2)
    denominator = np.sum((ref_img - test_img)**2)
    return 10 * np.log10(numerator/denominator)


def psnr(ref_img: np.ndarray, test_img: np.ndarray) -> float:
    """
    Computes Peak Signal-to-Noise ratio measured in dB

    Input:
        - ref_img (np.ndarray): 2D numpy array.
        - test_img (np.ndarray): 2D numpy array.
    
    Returns:
        float: peak signal-to-noise ratio.
    
    Reference:
        - https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    # checking they are 2D images
    assert len(ref_img.shape) == 2
    # checking both images have the same size
    assert ref_img.shape == test_img.shape

    mse = 1/(np.prod(ref_img.shape)) * np.sum((ref_img - test_img)**2)
    return 20 * np.log10(255.0 / np.sqrt(mse))


def average_snr(ref_img: np.ndarray, test_imgs: List[np.ndarray]) -> float:
    snrs = [snr(ref_img=ref_img, test_img=t_img) for t_img in test_imgs]
    return np.mean(snrs)

def average_psnr(ref_img: np.ndarray, test_imgs: List[np.ndarray]) -> float:
    psnrs = [psnr(ref_img=ref_img, test_img=t_img) for t_img in test_imgs]
    return np.mean(psnrs)

def average_ssim(ref_img: np.ndarray, test_imgs: List[np.ndarray]) -> float:
    ssims = [ssim(im1=ref_img, im2=t_img, data_range=255) for t_img in test_imgs]
    return np.mean(ssims)

class ImgNoiseCorruptor:
    """Generates corrupted images from a given original image by
    injecting different types of noise.
    
    Attributes:
        original_img (np.ndarray): 2d numpy array representing the original image
                of size (n_rows, n_cols) = (height, width).
    """
    DEFAULT_MAX_PIXEL_DISPLACEMENT = 4
    
    def __init__(
        self,
        original_img: np.ndarray,
        max_pixel_displacement: int = DEFAULT_MAX_PIXEL_DISPLACEMENT
    ):
        self.original_img = original_img
        self.max_pixel_displacement = max_pixel_displacement

    def _set_seed(self, seed: int = None) -> None:
        if seed is not None:
            np.random.seed(seed)

    def generate_gauss_noisy_imgs(
        self,
        n_snapshots: int,
        sigma: float,
        seed: int = None,
    ) -> np.ndarray:
        """Generates `n_snapshots` noisy images of the given original image.

        Args:
            n_snapshots (int): number of noisy images to generate.
            sigma (float): gaussian standard deviation to corrupt the original image.

        Returns:
            np.ndarray: numpy array of size (n_snapshots, height, width).
        """
        self._set_seed(seed)

        gaussian_noise_mtcs = [np.random.randn(*self.original_img.shape) for _ in range(n_snapshots)]
        snapshots = [
            self.original_img + sigma * randn_mtx
            for randn_mtx in gaussian_noise_mtcs
        ]
        return np.stack(snapshots, axis=0)

    def displace_img_horizontally(
        self, img: np.ndarray = None, max_displacement: int = None, seed: int = None
    ) -> np.ndarray:
        self._set_seed(seed)
        if img is None:
            img = self.original_img
        if max_displacement is None:
            max_displacement = self.max_pixel_displacement

        n_pixels_to_move = np.random.choice(np.arange(max_displacement+1), size=1)[0]
        if n_pixels_to_move == 0:
            return img

        zero_img = np.zeros(img.shape)
        move_to_right = np.random.choice([True, False], size=1)[0]
        if move_to_right:
            return np.hstack((zero_img[:,:(2*n_pixels_to_move)], img[:,n_pixels_to_move:-n_pixels_to_move]))
        else:
            return np.hstack((img[:,n_pixels_to_move:-n_pixels_to_move], zero_img[:,-(2*n_pixels_to_move):]))

    def displace_img_vertically(
        self, img: np.ndarray = None, max_displacement: int = None, seed: int = None
    ) -> np.ndarray:
        self._set_seed(seed)
        if img is None:
            img = self.original_img
        if max_displacement is None:
            max_displacement = self.max_pixel_displacement

        n_pixels_to_move = np.random.choice(np.arange(max_displacement+1), size=1)[0]
        if n_pixels_to_move == 0:
            return img

        zero_img = np.zeros(img.shape)
        move_up = np.random.choice([True, False], size=1)[0]
        if move_up:
            return np.vstack((zero_img[:(2*n_pixels_to_move),:], img[n_pixels_to_move:-n_pixels_to_move,:]))
        else:
            return np.vstack((img[n_pixels_to_move:-n_pixels_to_move,:], zero_img[:(2*n_pixels_to_move),:]))
        
    def generate_displaced_imgs(
        self, n_snapshots: int,  max_displacement: int = None, seed: int = None
    ) -> np.ndarray:
        self._set_seed(seed)
        if max_displacement is None:
            max_displacement = self.max_pixel_displacement

        snapshots = []
        for _ in range(n_snapshots):
            snapsht = np.copy(self.original_img)
            vdispl_snapsht = self.displace_img_vertically(img=snapsht)
            displ_snapsht = self.displace_img_horizontally(img=vdispl_snapsht)
            snapshots.append(displ_snapsht)

        return snapshots

    def add_rician_noise_fg(
        self, rice_b: float, sigma: float, img: np.ndarray = None, center_dist: bool = False, seed: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Adds Rician noise to the foreground part of the image.
        
        Returns: tuple with the corrupted img and the used foreground mask.
        """
        self._set_seed(seed)
        if img is None:
            img = self.original_img

        fg_mask = img > 0.0    
        n_rician_samples = np.prod(img.shape)
        rician_samples = rice.rvs(b=rice_b, scale=sigma, size=n_rician_samples)
        if center_dist:
            rice_dist_mean = rice.mean(b=rice_b, scale=sigma)
            rician_samples -= rice_dist_mean
        rician_noise_mtx = np.reshape(rician_samples, img.shape)
        rician_noisy_img = img + rician_noise_mtx
        # Returning the image corrupted just in the foreground (ignoring the background)
        return fg_mask * rician_noisy_img, fg_mask

    def generate_rician_noisy_displaced_imgs(
        self,
        n_snapshots: int,
        sigma: float,
        rice_b: float,
        max_displacement: int = None,
        center_noise_dist: bool = False,
        seed: int = None,
    ) -> np.ndarray:
        if seed is not None: np.random.seed(seed)

        displaced_snapshots = self.generate_displaced_imgs(
            n_snapshots=n_snapshots, max_displacement=max_displacement
        )

        fg_masks = []
        corrupted_snapshots = []
        for displ_snapsh in displaced_snapshots:
            rician_noised_ss, fg_m = self.add_rician_noise_fg(
                img=displ_snapsh, rice_b=rice_b, sigma=sigma, center_dist=center_noise_dist
            )
            corrupted_snapshots.append(rician_noised_ss)
            fg_masks.append(fg_m)

        return np.stack(corrupted_snapshots, axis=0), np.stack(fg_masks, axis=0)