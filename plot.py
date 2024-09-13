import matplotlib.pyplot as plt
import numpy as np
import os
import sys

pts = np.loadtxt(sys.argv[1])
plt.scatter(*pts.T, s=5)
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.gca().set_aspect('equal')
plt.show()
plt.close()