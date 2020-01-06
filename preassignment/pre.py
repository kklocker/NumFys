import numpy as np
import matplotlib.pyplot as plt
import time

a = time.time()

n = 10**np.linspace(0,6)
for i in n:
    plt.plot(np.cumsum(np.random.uniform(-1,1,int(i))))
b = time.time()
print(b-a)
plt.show()
