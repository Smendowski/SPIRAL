import matplotlib.pyplot as plt
from ts2i_visualization.time_series import ANOMALIES, x

for i, (signal, title, marker_pos) in enumerate(ANOMALIES):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(x, signal, "k-", linewidth=2.5)

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    ax.grid(alpha=0.15, linestyle="--", linewidth=0.3)

    ax.axhline(y=0, color="gray", linewidth=0.3, alpha=0.5)

    plt.savefig(
        f"{i:02d}_{title.lower().replace(' ', '_')}.pdf",
        bbox_inches="tight",
        pad_inches=0,
        dpi=1500,
    )
    plt.close()
