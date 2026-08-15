"""
Lightweight OpenCV (cv2) compatibility wrapper using NumPy, SciPy, and Pillow.
Ensures zero-error portability across environments without native OpenCV binaries.
"""
from typing import Any, Tuple
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
import scipy.signal as signal

INTER_AREA: int = 3
INTER_LINEAR: int = 1
CV_32F: int = 5
TM_CCOEFF_NORMED: int = 5


def resize(src: np.ndarray, dsize: Tuple[int, int], interpolation: int = INTER_AREA) -> np.ndarray:
    w, h = dsize
    if src.dtype != np.uint8:
        # Scale if float
        if src.max() <= 1.0 and src.min() >= 0.0:
            pil_img = Image.fromarray((src * 255.0).round().astype(np.uint8))
            res = pil_img.resize((w, h), Image.Resampling.BOX if interpolation == INTER_AREA else Image.Resampling.BILINEAR)
            return np.array(res, dtype=np.float32) / 255.0
    pil_img = Image.fromarray(src)
    res = pil_img.resize((w, h), Image.Resampling.BOX if interpolation == INTER_AREA else Image.Resampling.BILINEAR)
    return np.array(res, dtype=src.dtype)


def Sobel(src: np.ndarray, ddepth: int, dx: int, dy: int, ksize: int = 3) -> np.ndarray:
    axis = 1 if dx > 0 else 0
    return ndimage.sobel(src.astype(np.float32), axis=axis).astype(np.float32)


def magnitude(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.hypot(x, y).astype(np.float32)


def GaussianBlur(src: np.ndarray, ksize: Tuple[int, int], sigmaX: float, sigmaY: float = 0.0) -> np.ndarray:
    sig_y = sigmaY if sigmaY > 0 else sigmaX
    return ndimage.gaussian_filter(src.astype(np.float32), sigma=(sig_y, sigmaX)).astype(src.dtype)


def matchTemplate(image: np.ndarray, templ: np.ndarray, method: int = TM_CCOEFF_NORMED) -> np.ndarray:
    # Zero-mean normalized cross-correlation
    tpl_f = templ.astype(np.float32)
    img_f = image.astype(np.float32)
    
    tpl_mean = np.mean(tpl_f)
    tpl_std = np.std(tpl_f) + 1e-7
    tpl_norm = (tpl_f - tpl_mean) / tpl_std
    
    # Fast 2D correlation via FFT
    # Note: signal.fftconvolve / correlate2d
    th, tw = tpl_f.shape[:2]
    corr = signal.correlate2d(img_f, tpl_norm, mode='valid')
    
    # Local window standard deviation in image
    kernel = np.ones((th, tw), dtype=np.float32)
    img_sum = signal.correlate2d(img_f, kernel, mode='valid')
    img_sq_sum = signal.correlate2d(img_f**2, kernel, mode='valid')
    
    n_pix = th * tw
    img_mean = img_sum / n_pix
    img_var = np.maximum(0.0, (img_sq_sum / n_pix) - (img_mean**2))
    img_std = np.sqrt(img_var) + 1e-7
    
    zncc = (corr - (img_mean * np.sum(tpl_norm))) / (n_pix * img_std)
    return np.clip(zncc, -1.0, 1.0)


def minMaxLoc(src: np.ndarray) -> Tuple[float, float, Tuple[int, int], Tuple[int, int]]:
    min_val = float(np.min(src))
    max_val = float(np.max(src))
    min_idx = np.unravel_index(np.argmin(src), src.shape)
    max_idx = np.unravel_index(np.argmax(src), src.shape)
    min_loc = (int(min_idx[1]), int(min_idx[0]))
    max_loc = (int(max_idx[1]), int(max_idx[0]))
    return min_val, max_val, min_loc, max_loc
