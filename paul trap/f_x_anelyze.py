from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Physical Constants
D = 26e-6              # Diameter in meters (26 um)
R = D / 2              # Radius in meters
rho_p = 510            # Density of lycopodium in kg/m^3
mu = 1.81e-5           # Dynamic viscosity of air in kg/(m*s)

# Compute Mass and Stokes' Drag Coefficient
mass = (4 / 3) * np.pi * (R**3) * rho_p
drag_coeff = 6 * np.pi * mu * R

# 2. Calibration Factors (UPDATE THESE WITH YOUR ACTUAL EXPERIMENTAL VALUES)
FPS = 30.0                   # Frames per second of your camera
METERS_PER_PIXEL = 2.3636363636363636e-06      # Conversion factor (e.g., 10 um per pixel)

# 3. Load and Clean Data
csv_path = Path(__file__).resolve().parent / 'sphere_tracking.csv'
df = pd.read_csv(csv_path)
df = df.dropna(subset=['velocity_px_per_frame', 'acceleration_px_per_frame2']).copy()

# Convert units from pixels/frames to SI units (meters and seconds)
df['v_m_s'] = df['velocity_px_per_frame'] * METERS_PER_PIXEL * FPS
df['a_m_s2'] = df['acceleration_px_per_frame2'] * METERS_PER_PIXEL * (FPS**2)

# 4. Calculate the unknown force F(X)
# Equation: F(X) = m*a + 6*pi*mu*R*v
df['F_X_newtons'] = (mass * df['a_m_s2']) + (drag_coeff * df['v_m_s'])
df['F_X_pN'] = df['F_X_newtons'] * 1e12  # Convert to picoNewtons (pN)

# 5. Compute Mean and Standard Deviation of F(X)
overall_mean_pN = df['F_X_pN'].mean()
overall_std_pN = df['F_X_pN'].std(ddof=1)
overall_mean_N = df['F_X_newtons'].mean()
overall_std_N = df['F_X_newtons'].std(ddof=1)

per_sphere_stats = (
    df.groupby('sphere_id')['F_X_pN']
    .agg(count='count', mean_pN='mean', std_pN='std', min_pN='min', max_pN='max')
)
frame_mean = df.groupby('frame')['F_X_pN'].mean()

print("=== Overall Force F(X) Statistics (All Spheres) ===")
print(f"Mean F(X): {overall_mean_pN:.4f} pN  ({overall_mean_N:.4e} N)")
print(f"Std  F(X): {overall_std_pN:.4f} pN  ({overall_std_N:.4e} N)")
print(f"Mean across sphere means: {per_sphere_stats['mean_pN'].mean():.4f} ± {per_sphere_stats['mean_pN'].std(ddof=1):.4f} pN\n")

print("=== Per-Sphere Force F(X) Statistics [pN] ===")
print(per_sphere_stats.to_string(float_format=lambda x: f"{x:.4f}"))
print("\n=== Full Summary Statistics ===")
print(df[['F_X_pN', 'v_m_s', 'a_m_s2']].describe())

# 6. Visualize Force F(X) for All Spheres
# Validated categorical palette (from dataviz reference palette)
PALETTE = [
    '#2a78d6',  # 1: blue
    '#eb6834',  # 2: orange
    '#1baf7a',  # 3: aqua
    '#eda100',  # 4: yellow
    '#e87ba4',  # 5: magenta
    '#008300',  # 6: green
    '#6250d6',  # 7: violet
    '#e34948',  # 8: red
]
SURFACE = '#fcfcfb'
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRID_COLOR = '#e1e0d9'
AXIS_COLOR = '#c3c2b7'

fig, (ax_ts, ax_bar) = plt.subplots(
    1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [2.4, 1]}
)
fig.patch.set_facecolor(SURFACE)
for ax in (ax_ts, ax_bar):
    ax.set_facecolor(SURFACE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(AXIS_COLOR)
    ax.spines['bottom'].set_color(AXIS_COLOR)
    ax.tick_params(colors=INK_MUTED, labelsize=9.5)

# Left panel: F(X) over frame for all spheres + overall mean and ±1 SD band
ax_ts.grid(True, color=GRID_COLOR, linewidth=0.8, linestyle='-')
ax_ts.set_axisbelow(True)

# Overall ±1 SD band (subtle 10% opacity wash) and overall mean reference line
frames_sorted = np.sort(df['frame'].unique())
ax_ts.fill_between(
    frames_sorted,
    overall_mean_pN - overall_std_pN,
    overall_mean_pN + overall_std_pN,
    color='#2a78d6',
    alpha=0.10,
    linewidth=0,
    label=f'±1 SD band (±{overall_std_pN:.2f} pN)',
)

for i, (sphere_id, group) in enumerate(df.groupby('sphere_id')):
    group_sorted = group.sort_values('frame')
    color = PALETTE[i % len(PALETTE)]
    s_mean = group_sorted['F_X_pN'].mean()
    s_std = group_sorted['F_X_pN'].std(ddof=1)
    ax_ts.plot(
        group_sorted['frame'],
        group_sorted['F_X_pN'],
        label=f'Sphere {sphere_id} (μ={s_mean:.2f}, σ={s_std:.2f} pN)',
        color=color,
        linewidth=1.8,
        solid_joinstyle='round',
        solid_capstyle='round',
        alpha=0.9,
    )

ax_ts.axhline(
    overall_mean_pN,
    color=INK_PRIMARY,
    linewidth=1.5,
    linestyle='-',
    label=f'Overall Mean ({overall_mean_pN:.2f} pN)',
)

ax_ts.set_ylim(-115, 158)
ax_ts.set_xlabel('Frame', color=INK_SECONDARY, fontsize=10.5)
ax_ts.set_ylabel('Force F(X) [pN]', color=INK_SECONDARY, fontsize=10.5)
ax_ts.set_title(
    f'Force F(X) Over Time Across All Spheres\nOverall Mean = {overall_mean_pN:.2f} pN, Std Dev = {overall_std_pN:.2f} pN',
    color=INK_PRIMARY,
    fontsize=12,
    fontweight='semibold',
    loc='left',
    pad=12,
)
legend = ax_ts.legend(
    loc='upper left',
    frameon=True,
    facecolor=SURFACE,
    edgecolor=GRID_COLOR,
    framealpha=0.95,
    fontsize=8.5,
    ncol=2,
)
for text in legend.get_texts():
    text.set_color(INK_SECONDARY)

# Right panel: Per-sphere Mean F(X) (ranked horizontal bars, single series color + direct labels)
stats_sorted = per_sphere_stats.sort_values('mean_pN', ascending=True)
y_pos = np.arange(len(stats_sorted))
labels = [f'Sphere {sid}' for sid in stats_sorted.index]
means = stats_sorted['mean_pN'].values

ax_bar.grid(True, axis='x', color=GRID_COLOR, linewidth=0.8, linestyle='-')
ax_bar.set_axisbelow(True)
ax_bar.axvline(
    overall_mean_pN,
    color=INK_MUTED,
    linewidth=1.2,
    linestyle='-',
    zorder=2,
)
bars = ax_bar.barh(
    y_pos,
    means,
    height=0.36,
    color='#2a78d6',
    edgecolor='none',
    zorder=3,
)

for bar, (sid, row) in zip(bars, stats_sorted.iterrows()):
    w = bar.get_width()
    ax_bar.text(
        w + 0.22,
        bar.get_y() + bar.get_height() / 2,
        f"{row['mean_pN']:.2f} pN (σ={row['std_pN']:.1f})",
        va='center',
        ha='left',
        color=INK_PRIMARY,
        fontsize=8.5,
        zorder=4,
        bbox=dict(facecolor=SURFACE, edgecolor='none', pad=1.5),
    )

ax_bar.text(
    overall_mean_pN,
    -0.52,
    f'Overall mean: {overall_mean_pN:.2f} pN',
    color=INK_SECONDARY,
    fontsize=8.5,
    va='top',
    ha='center',
    bbox=dict(facecolor=SURFACE, edgecolor='none', pad=1.0),
)

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(labels, color=INK_PRIMARY, fontsize=9.5)
ax_bar.set_ylim(-0.75, len(stats_sorted) - 0.35)
ax_bar.set_xlim(0, max(means) * 1.62)
ax_bar.set_xlabel('Mean Force F(X) [pN]', color=INK_SECONDARY, fontsize=10.5)
ax_bar.set_title(
    'Mean Force F(X) by Sphere\nRanked highest to lowest with per-sphere SD (σ)',
    color=INK_PRIMARY,
    fontsize=12,
    fontweight='semibold',
    loc='left',
    pad=12,
)

plt.tight_layout()
out_plot = Path(__file__).resolve().parent / 'f_x_analysis.png'
plt.savefig(out_plot, dpi=150, bbox_inches='tight', facecolor=SURFACE)
plt.show()
