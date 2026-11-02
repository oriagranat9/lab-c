import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

vals = [
    (1.0225, 1636),
    (1.0987, 1663),
    (1.1817, 1693),
    (1.2734, 1719),
    (1.3803, 1757),#5
    (1.478, 1791),
    (1.5206, 1806),
    # (1.4, 1792),
    # (1.1723, 1735),#9
    (1.1723, 1685),
    (1.2406, 1706),
    (1.3370, 1740),
    (1.0572, 1644),
    (0.9864, 1621),
    (1.3337, 1738),
    (1.4346, 1772),
    (1.4815, 1791),
    (1.3167, 1736),
    (1.2727, 1718),
    (1.457, 1783),
]

z_0_px = np.mean([1545, 1549, 1548, 1459])
x = [v[0] for v in vals]
y = [v[1] - z_0_px for v in vals]

SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
TEXT_PRI = "#0b0b0b"
TEXT_MUT = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

fig, ax = plt.subplots(figsize=(7, 4.5))
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

ax.scatter(x, y, color=BLUE, s=60, linewidths=0, zorder=3)

for i, (xi, yi) in enumerate(zip(x, y)):
    ax.annotate(str(i + 1), (xi, yi), textcoords="offset points", xytext=(6, 4),
                fontsize=7, color=TEXT_MUT, zorder=4)

ax.set_xlabel("Measurement", color=TEXT_MUT, fontsize=10)
ax.set_ylabel("Pixel value", color=TEXT_MUT, fontsize=10)
ax.set_title("Measurement vs Pixel value", color=TEXT_PRI, fontsize=12, fontweight="bold", pad=12)

ax.tick_params(colors=TEXT_MUT, labelsize=9)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.spines["bottom"].set_visible(True)
ax.spines["bottom"].set_color(BASELINE)
ax.spines["left"].set_visible(True)
ax.spines["left"].set_color(BASELINE)

ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()
plt.savefig("z_dc_graph_vals.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
plt.show()
print("Saved z_dc_graph_vals.png")
