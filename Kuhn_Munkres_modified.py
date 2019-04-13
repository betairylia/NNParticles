import numpy as np
from scipy.spatial import distance
import random
import time

# Try to find an augmented path
def try_match(u, Lx, Ly, S, T, weights, match, slack, n):

    S[u] = 1
    
    for v in range(n):
        
        if(T[v] == 1):
            continue
        
        t = Lx[u] + Ly[v] - weights[u, v]

        # If this edge is compact
        if(abs(t) < 0.0001):
            T[v] = 1
            if(match[v] == -1 or try_match(match[v], Lx, Ly, S, T, weights, match, slack, n)):
                # We found an augmented path!
                match[v] = u
                return True
        
        else:
            slack[v] = min(slack[v], t)
    
    return False

def update(Lx, Ly, S, T, weights, match, slack, n):

    d = 999999999 # INF
    
    for i in range(n):
        if(T[i] == 0):
            d = min(d, slack[i])
    
    for i in range(n):
        if(S[i] == 1):
            Lx[i] -= d
        if(T[i] == 1):
            Ly[i] += d

def KM(predict, groundtruth, length, maxl):
    
    batches = predict.shape[0]
    result_array = np.zeros((batches, maxl[0]), dtype = np.int32)

    for batch_idx in range(batches):

        # Init variables
        n = int(length[batch_idx])

        weights = np.zeros((maxl[0], maxl[0]))
        match = np.zeros((maxl[0]), dtype = np.int32)
        Lx = np.zeros((maxl[0]))
        Ly = np.zeros((maxl[0]))
        slack = np.zeros((maxl[0]))
        S = np.zeros((maxl[0]), dtype = np.int32)
        T = np.zeros((maxl[0]), dtype = np.int32)

        # Calculate weights
        for i in range(maxl[0]):
            for j in range(n):
                # weights[i, j] = min(10000.0, 1.0 / distance.euclidean(predict[batch_idx, i, 0:3], groundtruth[batch_idx, i, 0:3]))
                # weights[i, j] = distance.euclidean(predict[batch_idx, i, 0:3], groundtruth[batch_idx, j, 0:3])
                # weights[i, j] = 0 - distance.euclidean(predict[batch_idx, i, 0:3], groundtruth[batch_idx, j, 0:3]) # L2 distance
                weights[i, j] = 0 - distance.cityblock(predict[batch_idx, i, 0:3], groundtruth[batch_idx, j, 0:3]) # L1 distance
            for j in range(n, maxl[0]):
                weights[i, j] = - 999999999 # INF
        
        # Init Lx & Ly
        for i in range(maxl[0]):
            Lx[i] = 0
            Ly[i] = 0
            match[i] = -1
            for j in range(maxl[0]):
                Lx[i] = max(Lx[i], weights[i, j])
        
        # KM Main algorithm
        for i in range(maxl[0]):
            for j in range(maxl[0]):
                slack[j] = 999999999 # INF
            while True:
                S = np.zeros((maxl[0]))
                T = np.zeros((maxl[0]))

                if(try_match(i, Lx, Ly, S, T, weights, match, slack, maxl[0])):
                    break
                else:
                    update(Lx, Ly, S, T, weights, match, slack, maxl[0])
        
        # Write results
        for i in range(maxl[0]):
            result_array[batch_idx, match[i]] = i
    
    return result_array

def test(length, sigma):

    test_predict = np.zeros((16, length, 6))
    test_gt = np.zeros((16, length, 6))
    test_noise = np.zeros((16, length, 6))
    test_single = np.zeros((16, length, 6))
    batch_length = np.zeros((16))
    maxl = np.zeros((1))

    maxl = length

    # fill data
    for batch_idx in range(16):
        
        n = random.randint(length // 4, length)
        # n = length
        batch_length[batch_idx] = n

        order = np.arange(length)
        np.random.shuffle(order)

        for i in range(n):
            rData = np.random.rand(6) * 20 - 10
            test_gt[batch_idx, i, 0:6] = rData
        
        for i in range(length):
            noise = np.random.normal(0.0, sigma, (6,))
            test_predict[batch_idx, i, 0:6] = test_gt[batch_idx, order[i], 0:6] + noise

            if i < n:
                test_noise[batch_idx, i, 0:6] = test_gt[batch_idx, i, 0:6] + noise
                test_single[batch_idx, i, 0:6] = test_gt[batch_idx, 0, 0:6]
    
    raw_Loss = np.sum(np.absolute(test_gt - test_predict))
    min_Loss = np.sum(np.absolute(test_gt - test_noise))
    matched_order = KM(test_predict, test_gt, batch_length, [maxl, maxl])
    single_order = KM(test_single, test_gt, batch_length, [maxl, maxl])
    
    KM_Loss = 0
    Single_Loss = 0

    for bi in range(16):
        for i in range(int(batch_length[bi])):
            KM_Loss += np.sum(np.absolute(test_predict[bi, i] - test_gt[bi, matched_order[bi, i]]))
            Single_Loss += np.sum(np.absolute(test_single[bi, i] - test_gt[bi, single_order[bi, i]]))

    print("Raw Loss = %.8f, KM Loss = %.8f, Minimize Loss = %.8f, Single Loss = %.8f" % (raw_Loss, KM_Loss, min_Loss, Single_Loss))
