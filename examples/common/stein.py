import numpy as np
import sys

def gauss_mmd(x, y, sigma=1):
    d = x.shape[1]
    # do this computation in blocks to avoid heavy memory requirements
    blk = 500

    print('KXX:')
    KXX = 0.
    for i in range(0, x.shape[0], blk):
        sys.stdout.write(f"row index {i}/{x.shape[0]}    \r")
        sys.stdout.flush()
        for j in range(0, x.shape[0], blk):
            xx_diffs = x[i:(i+blk), np.newaxis, :] - x[np.newaxis, j:(j+blk), :]
            xx_sq_dists = (xx_diffs**2).sum(axis=2)
            KXX += np.exp(-xx_sq_dists/(2.*sigma**2)).sum()
    KXX /= x.shape[0]**2
    sys.stdout.write("\n")
    sys.stdout.flush()
    print(KXX)

    print('KYY:')
    KYY = 0.
    for i in range(0, y.shape[0], blk):
        sys.stdout.write(f"row index {i}/{y.shape[0]}    \r")
        sys.stdout.flush()
        for j in range(0, y.shape[0], blk):
            yy_diffs = y[i:(i+blk), np.newaxis, :] - y[np.newaxis, j:(j+blk), :]
            yy_sq_dists = (yy_diffs**2).sum(axis=2)
            KYY += np.exp(-yy_sq_dists/(2.*sigma**2)).sum()
    KYY /= y.shape[0]**2
    sys.stdout.write("\n")
    sys.stdout.flush()
    print(KYY)


    print('KXY:')
    KXY = 0.
    for i in range(0, x.shape[0], blk):
        sys.stdout.write(f"row index {i}/{x.shape[0]}    \r")
        sys.stdout.flush()
        for j in range(0, y.shape[0], blk):
            xy_diffs = x[i:(i+blk), np.newaxis, :] - y[np.newaxis, j:(j+blk), :]
            xy_sq_dists = (xy_diffs**2).sum(axis=2)
            KXY += np.exp(-xy_sq_dists/(2.*sigma**2)).sum()
    KXY /= (x.shape[0]*y.shape[0])
    sys.stdout.write("\n")
    sys.stdout.flush()
    print(KXY)

    ## K(X,X)
    #xx_diffs = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    #xx_sq_dists = (xx_diffs**2).sum(axis=2)
    #kernel_xx = np.exp(-xx_sq_dists/(2.*sigma**2))

    ## K(Y,Y)
    #yy_diffs = y[:, np.newaxis, :] - y[np.newaxis, :, :]
    #yy_sq_dists = (yy_diffs**2).sum(axis=2)
    #kernel_yy = np.exp(-yy_sq_dists/(2.*sigma**2))

    ## K(X, Y)
    #xy_diffs = x[:, np.newaxis, :] - y[np.newaxis, :, :]
    #xy_sq_dists = (xy_diffs**2).sum(axis=2)
    #kernel_xy = np.exp(-xy_sq_dists/(2.*sigma**2))

    #print('original')
    #print(kernel_xx.sum()/x.shape[0]**2 + kernel_yy.sum()/y.shape[0]**2 - 2.*kernel_xy.sum()/(x.shape[0]*y.shape[0]))
    #print('new')
    #print(KXX/x.shape[0]**2 + KYY/y.shape[0]**2 - 2*KXY/(x.shape[0]*y.shape[0]))

    #return kernel_xx.sum()/x.shape[0]**2 + kernel_yy.sum()/y.shape[0]**2 - 2.*kernel_xy.sum()/(x.shape[0]*y.shape[0])
    return KXX + KYY - 2*KXY

def imq_mmd(x, y, sigma=1, beta=0.5):
    d = x.shape[1]
    # do this computation in blocks to avoid heavy memory requirements
    blk = 500

    print('KXX:')
    KXX = 0.
    for i in range(0, x.shape[0], blk):
        sys.stdout.write(f"row index {i}/{x.shape[0]}    \r")
        sys.stdout.flush()
        for j in range(0, x.shape[0], blk):
            xx_diffs = x[i:(i+blk), np.newaxis, :] - x[np.newaxis, j:(j+blk), :]
            xx_sq_dists = (xx_diffs**2).sum(axis=2)
            KXX += (1./(xx_sq_dists/(2.*sigma**2) + 1.)**beta).sum()
    KXX /= x.shape[0]**2
    sys.stdout.write("\n")
    sys.stdout.flush()
    print(KXX)


    print('KYY:')
    KYY = 0.
    for i in range(0, y.shape[0], blk):
        sys.stdout.write(f"row index {i}/{y.shape[0]}    \r")
        sys.stdout.flush()
        for j in range(0, y.shape[0], blk):
            yy_diffs = y[i:(i+blk), np.newaxis, :] - y[np.newaxis, j:(j+blk), :]
            yy_sq_dists = (yy_diffs**2).sum(axis=2)
            KYY += (1./(yy_sq_dists/(2.*sigma**2) + 1.)**beta).sum()
    KYY /= y.shape[0]**2
    sys.stdout.write("\n")
    sys.stdout.flush()
    print(KYY)

    print('KXY:')
    KXY = 0.
    for i in range(0, x.shape[0], blk):
        sys.stdout.write(f"row index {i}/{x.shape[0]}    \r")
        sys.stdout.flush()
        for j in range(0, y.shape[0], blk):
            xy_diffs = x[i:(i+blk), np.newaxis, :] - y[np.newaxis, j:(j+blk), :]
            xy_sq_dists = (xy_diffs**2).sum(axis=2)
            KXY += (1./(xy_sq_dists/(2.*sigma**2) + 1.)**beta).sum()
    KXY /= (x.shape[0]*y.shape[0])
    sys.stdout.write("\n")
    sys.stdout.flush()
    print(KXY)

    ## K(X,X)
    #xx_diffs = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    #xx_sq_dists = (xx_diffs**2).sum(axis=2)
    #kernel_xx = 1./(xx_sq_dists/(2.*sigma**2) + 1.)**beta

    ## K(Y,Y)
    #yy_diffs = y[:, np.newaxis, :] - y[np.newaxis, :, :]
    #yy_sq_dists = (yy_diffs**2).sum(axis=2)
    #kernel_yy = 1./(yy_sq_dists/(2.*sigma**2) + 1.)**beta

    ## K(X, Y)
    #xy_diffs = x[:, np.newaxis, :] - y[np.newaxis, :, :]
    #xy_sq_dists = (xy_diffs**2).sum(axis=2)
    #kernel_xy = 1./(xy_sq_dists/(2.*sigma**2) + 1.)**beta

    #return kernel_xx.sum()/x.shape[0]**2 + kernel_yy.sum()/y.shape[0]**2 - 2.*kernel_xy.sum()/(x.shape[0]*y.shape[0])
    return KXX + KYY - 2*KXY



def _gauss_stein(x, y, score_x, score_y, sigma, beta): # beta is unused here, but we keep for consistent api
    _, p = x.shape
    d = x[:, None, :] - y[None, :, :]
    dists = (d ** 2).sum(axis=-1)
    k = np.exp(-dists / sigma**2 / 2)
    scalars = score_x.dot(score_y.T)
    scores_diffs = score_x[:, None, :] - score_y[None, :, :]
    diffs = (d * scores_diffs).sum(axis=-1)
    der2 = p - dists / sigma**2
    return k * (scalars + diffs / sigma**2 + der2 / sigma**2)

def _imq_stein(x, y, score_x, score_y, sigma, beta):
    _, p = x.shape
    d = x[:, None, :] - y[None, :, :]
    dists = (d ** 2).sum(axis=-1)
    res = 1 + dists /(2.*sigma**2)
    kxy = res ** (-beta)
    scores_d = score_x[:, None, :] - score_y[None, :, :]
    temp = d * scores_d
    dkxy = 2 * beta /(2.*sigma**2) * (res) ** (-beta - 1) * temp.sum(axis=-1)
    d2kxy = 2 * (
        beta / (2.*sigma**2) * (res) ** (-beta - 1) * p
        - 2 * beta * (beta + 1) /(2.*sigma**2)** 2 * dists * res ** (-beta - 2)
    )
    return score_x.dot(score_y.T) * kxy + dkxy + d2kxy


def _stein_blocked(x, scores, _ksd, sigma, beta):
    # do this computation in blocks to avoid heavy memory requirements
    # and since computing the matrix of scores can be expensive even with block computation,
    # we will do this iteratively using a running stochastic estimate (with std error estimation to know when to quit)
    blk = 500
    KSD = 0.
    for i in range(0, x.shape[0], blk):
        sys.stdout.write(f"row index {i}/{x.shape[0]}    \r")
        sys.stdout.flush()
        for j in range(0, x.shape[0], blk):
            KSD += _ksd(x[i:(i+blk), :], x[j:(j+blk), :], scores[i:(i+blk),:], scores[j:(j+blk),:], sigma, beta).sum()
    sys.stdout.write("\n")
    sys.stdout.flush()
    KSD /= (x.shape[0] ** 2)
    return KSD

def _stochastic_stein_blocked(x, score_estimator, _ksd, sigma, beta):
    # do this computation in blocks to avoid heavy memory requirements
    # and since computing the matrix of scores can be expensive even with block computation,
    # we will do this iteratively using a running stochastic estimate (with std error estimation to know when to quit)
    blk = 5000
    ct = 0.
    mom1 = 0.
    mom2 = 0.
    while True:
        print(f'KSD estimation iteration {ct+1}')
        # need two iid subsamples of the data to estimate the score-score/score-data inner products
        # note: if you just use one subsample, you're essentially computing the average KSD over subsampled posteriors (no good)
        # if you use use two subsamples, you're getting an unbiased estimate of the full posterior KSD
        scores_1 = score_estimator(x, blk)
        scores_2 = score_estimator(x, blk)
        KSD = 0.
        for i in range(0, x.shape[0], blk):
            sys.stdout.write(f"row index {i}/{x.shape[0]}    \r")
            sys.stdout.flush()
            for j in range(0, x.shape[0], blk):
                KSD += _ksd(x[i:(i+blk), :], x[j:(j+blk), :], scores_1[i:(i+blk),:], scores_2[j:(j+blk),:], sigma, beta).sum()
        sys.stdout.write("\n")
        sys.stdout.flush()
        KSD /= (x.shape[0] ** 2)
        ct += 1
        mom1 += KSD
        mom2 += KSD**2
        KSD_est = mom1/ct
        KSD_std = np.sqrt(mom2/ct - (mom1/ct)**2)/np.sqrt(ct)
        print(f'KSD estimate: {KSD_est} std err estimate: {KSD_std}')
        if ct > 1 and np.fabs(KSD_std/KSD_est) < 0.1:
            break
    return mom1/ct

def gauss_stein(x, scores, sigma=1, beta=None):
    return _stein_blocked(x, scores, _gauss_stein, sigma, beta)

def imq_stein(x, scores, sigma=1, beta=0.5):
    return _stein_blocked(x, scores, _imq_stein, sigma, beta)
