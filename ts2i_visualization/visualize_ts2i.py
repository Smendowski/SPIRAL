import matplotlib.pyplot as plt
from ts2i_visualization.time_series import ANOMALIES
import os
from src.transformations.ts2i.gramian_angular_field import (
    gramian_angular_difference_field,
    gramian_angular_summation_field,
)
from src.transformations.ts2i.random_noise import random_noise
from src.transformations.ts2i.line_plot import line_plot
from src.transformations.ts2i.state_grid import state_grid
from src.transformations.ts2i.markov_transition_field import markov_transition_field
from src.transformations.ts2i.recurrence_plot import recurrence_plot
from src.transformations.ts2i.wavelets import (
    ricker_wavelet_transform,
    morlet_wavelet_transform,
)
from src.transformations.ts2i.novel import spiral

TRANSFORMS = [
    ("RN", random_noise),
    ("LP", line_plot),
    ("SG", state_grid),
    ("GASF", gramian_angular_summation_field),
    ("GADF", gramian_angular_difference_field),
    ("MTF", markov_transition_field),
    ("RP", recurrence_plot),
    ("RWT", ricker_wavelet_transform),
    ("MWT", morlet_wavelet_transform),
    ("SPIRAL", spiral)
]

SIZE = 64
CMAP = "coolwarm"

for anom_idx, (signal, anom_title, _) in enumerate(ANOMALIES):
    for ts2i_name, ts2i_func in TRANSFORMS:
        output_dir = f"viz/{ts2i_name}"
        os.makedirs(output_dir, exist_ok=True)

        img = ts2i_func(signal, SIZE)

        img_2d = img[0]

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(img_2d, cmap=CMAP, interpolation="nearest")

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        plt.tight_layout(pad=0)

        filename = (
            f"{output_dir}/{anom_idx:02d}_{anom_title.lower().replace(' ', '_')}.pdf"
        )
        plt.savefig(filename, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()
