from pyts.image import RecurrencePlot
from skimage.transform import resize
import numpy as np
from src.transformations.ts2i import named
from src.transformations.typing import Array1D, ArrayImage


@named("RP")
def recurrence_plot(window: Array1D, size: int) -> ArrayImage:
    rp = RecurrencePlot(threshold="point")
    img = rp.fit_transform(np.ravel(window)[None, :]).squeeze(0)

    img = resize(img, (size, size), anti_aliasing=True, preserve_range=True).astype(
        np.float32
    )
    img = np.clip(img, 0, 1)

    img_rgb = np.stack([img, img, img], axis=0)
    return img_rgb
