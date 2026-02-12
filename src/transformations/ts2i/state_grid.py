from src.transformations.typing import Array1D, ArrayImage
import numpy as np
from src.transformations.ts2i import named
from skimage.transform import resize


@named("SG")
def state_grid(window: Array1D, size: int) -> ArrayImage:
    img = resize(window, (size, size), anti_aliasing=True, preserve_range=True).astype(
        np.float32
    )

    img_rgb = np.stack([img, img, img], axis=0)
    return img_rgb
