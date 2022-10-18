import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10)
y = np.ones(10)
labels = np.arange(0, 10)
print(labels)
print(len(labels))

plt.plot(x, y)
for i in range(0,len(labels) ):
    plt.annotate(labels[i], (x[i], y[i]))
plt.show()
#plt.savefig("./test.png")
plt.close()
