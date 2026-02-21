import matplotlib.pyplot as plt
import numpy as np

cases = ['a','b','c','d','e']
lab = [0.082734,0.455495,0.338541,1.861825,1.947038]
hpc = [0.270000,1.190000,1.070000,5.240000,3.870000]
w, x = 0.4, np.arange(len(cases))

plt.bar(x-w/2, lab, w, label='Lab PC')
plt.bar(x+w/2, hpc, w, label='HPC')
plt.xticks(x, cases)
plt.xlabel('Problem Index')
plt.ylabel('Execution Time (s)')
plt.legend()
plt.show()
