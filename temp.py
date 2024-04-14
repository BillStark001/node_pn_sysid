import numpy as np
import matplotlib.pyplot as plt

N = 51

x = np.linspace(0, N-1, N)

# 绘制各种窗函数
rect_window = np.ones(N)
plt.plot(x, rect_window, label='Rectangular Window')

# 汉明窗
hamming_window = np.hamming(N)
plt.plot(x, hamming_window, label='Hamming Window')

# 汉宁窗
hann_window = np.hanning(N)
plt.plot(x, hann_window, label='Hann Window')

# 布莱克曼窗
blackman_window = np.blackman(N)
plt.plot(x, blackman_window, label='Blackman Window')

# 巴特利特窗
bartlett_window = np.bartlett(N)
plt.plot(x, bartlett_window, label='Bartlett Window')

# 凯泽窗
kaiser_window = np.kaiser(N, 14)
plt.plot(x, kaiser_window, label='Kaiser Window')

plt.legend()

plt.show()