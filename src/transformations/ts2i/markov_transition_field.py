from pyts.image import MarkovTransitionField
from src.transformations.typing import Array1D, ArrayImage
import numpy as np
from skimage.transform import resize
from src.transformations.ts2i import named


@named("MTF")
def markov_transition_field(window: Array1D, size: int) -> ArrayImage:
    window = np.ravel(window)[None, :]

    mtf = MarkovTransitionField()

    img = mtf.fit_transform(window)
    img = img.squeeze(0)

    img = resize(img, (size, size), anti_aliasing=True, preserve_range=True).astype(
        np.float32
    )

    img = np.clip(img, 0, 1)

    img_rgb = np.stack([img, img, img], axis=0)  # (C, H, W)
    return img_rgb
