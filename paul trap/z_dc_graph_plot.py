import numpy as np
import matplotlib.pyplot as plt

px_to_mm = (5 / 8112)

z_0_px = np.mean([1545, 1549, 1548, 1459])

print("z mm: ", np.mean(z_0_px) * px_to_mm)
dc_vals = [0.8776, 0.8878, 0.883, 0.8883, 0.8887, 0.8839, 0.8809]

dc_avg = np.mean(dc_vals)
print("Average of dc_vals:", dc_avg)

vals = [
    (1.0225, 1636),
    (1.0987, 1663),
    (1.1817, 1693),
    (1.2734, 1719),
    (1.3803, 1757),  # 5
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
x = [v[0] for v in vals]
# y min values with z_0 reduced
y = [v[1] - z_0_px for v in vals]


plt.scatter(x, y)
plt.title("Measurement vs Pixel value")
plt.xlabel("Measurement")
plt.ylabel("Pixel value")
plt.grid(True)
plt.show()
