from src.transformations.typing import Array1D, ArrayImage
from src.transformations.ts2i import named
import numpy as np
from skimage.draw import line, disk


def draw_line_series(series: np.ndarray, size: int) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.float32)
    T = len(series)

    t_orig = np.linspace(0, 1, T)
    t_new = np.linspace(0, 1, size)
    s_res = np.interp(t_new, t_orig, series).astype(np.float32)
    s_res = np.clip(s_res, 0, 1)

    y = (1.0 - s_res) * (size - 1)
    x = np.arange(size, dtype=int)
    y_int = np.rint(y).astype(int)

    for i in range(size - 1):
        rr, cc = line(y_int[i], x[i], y_int[i + 1], x[i + 1])
        img[rr, cc] = 1.0

    for i in range(size):
        rr, cc = disk((y_int[i], x[i]), radius=1, shape=img.shape)
        img[rr, cc] = 1.0

    return img


@named("LP")
def line_plot(window: Array1D, size: int) -> ArrayImage:
    series = np.ravel(window)
    img = draw_line_series(series, size)
    img_rgb = np.stack([img, img, img], axis=0)
    return img_rgb
