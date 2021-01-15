import numpy as np

corr_obs = 10
num_ant_txrx = 1
data_equalized = np.array([1+1j, 1-1j, -1+1j, -1-1j, 1-1j, -1-1j, -1+1j, 1+1j])
print('Data Shape: ', data_equalized.shape[0])
print(data_equalized)
x = np.zeros([1, 1, 8], dtype=complex)
for m in range(num_ant_txrx):
    for p in range(corr_obs):
        print('m:', m, 'p:', p)
        if m == 0 and p == 0:
            x[m, p, :] = x[m, p, :] + data_equalized
            print(x.shape)
        else:
            x = np.vstack((x[m, :], data_equalized))
            x = x[np.newaxis, :, :]
            print(x)
            print(x.shape)
