import numpy as np
import matplotlib.pyplot as plt

cost_rate = 0.85
Qc = 0
discount = 0.99
lr = 1e-3
batch_size = 10
ep_length = 1000
N_step = int(1e5)
qc_vals = np.zeros((N_step,))
for i in range(N_step):
    for j in range(batch_size):
        cost_violation = np.random.choice([0, 1], p=[1-cost_rate, cost_rate])
        mask = np.random.choice([1, 0], p=[1-1/ep_length, 1/ep_length])
        taget_cost = cost_violation + mask * discount * Qc
        loss = taget_cost - Qc
        Qc += lr * loss
    qc_vals[i] = Qc

plt.figure()
plt.plot(qc_vals)
plt.show()
stop=0