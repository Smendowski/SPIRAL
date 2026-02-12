from src.transformations.typing import Array1D, ArrayImage
from src.transformations.ts2i import named
from skimage.transform import resize
import numpy as np
import pywt


def _cwt_univariate_default(ts: Array1D, wavelet: str, size: int) -> ArrayImage:
    ts = np.ravel(ts).astype(np.float32)
    T = ts.shape[0]

    S = min(64, max(16, T // 2))
    S = max(2, min(int(S), T))
    scales = np.arange(1, S + 1, dtype=np.float32)

    coeffs, _ = pywt.cwt(ts, scales, wavelet=wavelet, sampling_period=1.0)
    img = np.abs(coeffs).astype(np.float32)

    img = np.log1p(img)

    img = resize(img, (size, size), anti_aliasing=True, preserve_range=True).astype(
        np.float32
    )

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = np.clip(img, 0, 1)
    img_rgb = np.stack([img, img, img], axis=0)
    return img_rgb


@named("RWT")
def ricker_wavelet_transform(window: Array1D, size: int) -> ArrayImage:
    return _cwt_univariate_default(window, wavelet="mexh", size=size)


@named("MWT")
def morlet_wavelet_transform(window: Array1D, size: int) -> ArrayImage:
    return _cwt_univariate_default(window, wavelet="morl", size=size)
