from pyts.image import GramianAngularField
from src.transformations.typing import Array1D, ArrayImage
import numpy as np
from skimage.transform import resize
from src.transformations.ts2i import named


@named("GASF")
def gramian_angular_summation_field(window: Array1D, size: int) -> ArrayImage:
    gasf = GramianAngularField(method="summation")
    window = window.reshape(1, -1)
    img = gasf.fit_transform(window).squeeze(0)

    # Normalize before resize [-1, 1] -> [0, 1]
    img = (img + 1) / 2
    img = np.clip(img, 0, 1)

    img = resize(img, (size, size), anti_aliasing=True, preserve_range=True).astype(
        np.float32
    )

    return np.stack([img, img, img], axis=0)


@named("GADF")
def gramian_angular_difference_field(window: Array1D, size: int) -> ArrayImage:
    gasf = GramianAngularField(method="difference")
    window = window.reshape(1, -1)
    img = gasf.fit_transform(window).squeeze(0)

    # Normalize before resize [-1, 1] -> [0, 1]
    img = (img + 1) / 2
    img = np.clip(img, 0, 1)

    img = resize(img, (size, size), anti_aliasing=True, preserve_range=True).astype(
        np.float32
    )

    return np.stack([img, img, img], axis=0)
