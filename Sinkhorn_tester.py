import numpy as np
import ot
import ot.plot
import ot.gpu
from matplotlib import pyplot as pl

#%% parameters and data generation

n = 2048  # nb samples

mu_s = np.array([0, 0])
cov_s = np.array([[1, 0], [0, 1]])

mu_t = np.array([0, 0])
cov_t = np.array([[1, -.8], [-.8, 1]])

xs = ot.datasets.make_2D_samples_gauss(n, mu_s, cov_s)
xt = ot.datasets.make_2D_samples_gauss(n, mu_t, cov_t)

a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples

# loss matrix
M = ot.dist(xs, xt)
# M += 1e-4
# M /= M.max()
# M /= 100.0

print(M.max())
print(M.min())
print(M.mean())

# #%% plot samples

# pl.figure(1)
# pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
# pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
# pl.legend(loc=0)
# pl.title('Source and target distributions')
# pl.show()

# pl.figure(2)
# pl.imshow(M, interpolation='nearest')
# pl.title('Cost matrix M')
# pl.show()

#%% sinkhorn

print("Start")
# reg term
lambd = 1e-3

# Gs = ot.gpu.sinkhorn(a, b, M, lambd, numItermax=200)
Gs = ot.bregman.sinkhorn_stabilized(a, b, M, lambd, numItermax=200)

print("End")

pl.figure(5)
pl.imshow(Gs, interpolation='nearest')
pl.title('OT matrix sinkhorn')

pl.figure(6)
ot.plot.plot2D_samples_mat(xs, xt, Gs, color=[.5, .5, 1], thr = 1e-2)
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
pl.legend(loc=0)
pl.title('OT matrix Sinkhorn with samples')

pl.show()

