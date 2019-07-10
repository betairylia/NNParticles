import numpy as np
import ot
import ot.gpu
import ot.plot

from matplotlib import pyplot as pl

def Sinkhorn_dist(x, y, a, b, M):
    
    bs = a.shape[0]
    ns = a.shape[1]
    nt = b.shape[1]

    transport_mat = np.zeros((bs, ns, nt), np.float32)

    # M = M / M.max()
    # print(M.max())
    # print(M.min())
    # print(M.mean())
    # print(M.sum())
    # M /= M.sum() * 100.0
    M /= 100.0

    for bid in range(bs):
        transport_mat[bid, :, :] = ot.gpu.sinkhorn(a[bid], b[bid], M[bid] / M[bid].max(), 3e-2, numItermax = 100, stopThr = 1e-2)


    # xs = x[0, :, :2]
    # xt = y[0, :, :2]
    # Gs = transport_mat[0]

    # pl.figure(1)
    # pl.imshow(Gs, interpolation='nearest')
    # pl.title('OT matrix sinkhorn')

    # # pl.figure(2)
    # # ot.plot.plot2D_samples_mat(xs, xt, Gs, color=[.5, .5, 1], thr = 1e-2)
    # # pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
    # # pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
    # # pl.legend(loc=0)
    # # pl.title('OT matrix Sinkhorn with samples')
    # pl.show()

    return transport_mat.astype(np.float32)

