
import numpy as np
import matplotlib.pyplot as plt

x = ['CC-OPF','LDT-CC','WCC-LDT']
y = np.array([4475.0, 4570.0, 4550.8])

plt.plot(x, y)
plt.ylabel("Total Cost [$]")

plt.show()