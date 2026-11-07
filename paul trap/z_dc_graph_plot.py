import numpy as np
import matplotlib.pyplot as plt

dc_vals = [0.8776, 0.8878, 0.883, 0.8883, 0.8887, 0.8839, 0.8809]

dc_avg = np.mean(dc_vals)
# print("Average of dc_vals:", dc_avg)

px_to_mm = (5 / 8112)

z_0_measure = [1545, 1549, 1548, 1459]
z_0_px_avg = np.mean(z_0_measure)
z_0_px_std = np.std(z_0_measure, ddof=1)

print("z mm: ", z_0_px_avg * px_to_mm, "z std: ", z_0_px_std * px_to_mm)

# dc,z_zmin,z_max
vals = [
    (1.0225, 1636, 1781),
    (1.0987, 1663, 1905),
    (1.1817, 1693, 2029),
    (1.2734, 1719, 2171),
    (1.3803, 1757, 2326),  # 5
    (1.478, 1791, 2480),
    (1.5206, 1806, 2538),
    # (1.4, 1792),
    # (1.1723, 1735),#9
    (1.1723, 1685, 2022),
    (1.2406, 1706, 2123),
    (1.3370, 1740, 2268),
    (1.0572, 1644, 1840),  #
    (0.9864, 1621, 1734),
    (1.3337, 1738, 2263),
    (1.4346, 1772, 2415),
    (1.4815, 1791, 2489),
    (1.3167, 1736, 2238),
    (1.2727, 1718, 2175),
    (1.457, 1783, 2457),
]
x = [v[0] for v in vals]
# y min values with z_0 reduced
y_min = [v[1] for v in vals]
y_max = [v[2] for v in vals]
y_avg = [np.average([v[1], v[2]]) for v in vals]

plt.scatter(x, y_min, label="y_min")
plt.scatter(x, y_max, label="y_max")
plt.scatter(x, y_avg, label="y_avg")
for i in range(len(x)):
    plt.annotate(str(i), (x[i], y_min[i]), textcoords="offset points", xytext=(0, 5), ha="center")
    plt.annotate(str(i), (x[i], y_max[i]), textcoords="offset points", xytext=(0, 5), ha="center")
    plt.annotate(str(i), (x[i], y_avg[i]), textcoords="offset points", xytext=(0, 5), ha="center")
plt.title("Measurement vs Pixel value")
plt.xlabel("Measurement")
plt.ylabel("Pixel value")
plt.legend()
plt.grid(True)
plt.show()


def v_dc(v_mon):
    return 82 * (v_mon - 0.06)


def e_dc(e_mon):
    return 4850 * (e_mon - 0.06)
