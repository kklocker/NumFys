import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import trange

N =200000 # Number of random walks
n = 200 # Max number of steps / shape of t-array
t = np.arange(n)

rand = np.random.uniform(-1,1,(N,n))
random_walks = np.cumsum(rand,axis = 1) - rand

p = np.zeros(n)

for i in trange(N,desc="1"):
    for j in range(n):
        if random_walks[i,1] > 0:
            if random_walks[i,j] <0:
                p[j] += 1
                break
        if random_walks[i,1] < 0:
            if random_walks[i,j] >0:
                p[j] += 1
                break

logt = np.log(t)[2:]
alpha = np.log(p[2:])
print(logt[:2])

coeffs = np.polyfit(logt, alpha,1)
print(logt*coeffs[0]  + coeffs[1])
print(alpha)
print(coeffs)

plt.figure()
plt.subplot(121)
plt.plot(p[2:], label = "Probability")

plt.subplot(122)
plt.plot(alpha, label = "Alpha")
plt.plot(logt*coeffs[0] + coeffs[1], label = "Regression")
#plt.xscale("log")

plt.legend()
plt.show()

# for i in tqdm(range(N)):


