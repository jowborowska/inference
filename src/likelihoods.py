import numpy as np


def ln_chi_squared(data, mean, variance):
    loglike = - 0.5 * np.sum((data - mean)**2 / variance + np.log(variance))

    # print(np.sum((data - mean)**2 / variance))
    # print(np.array2string(data, separator=','))
    # print(np.array2string(mean, separator=','))
    # print(np.array2string(np.sqrt(variance), separator=','))
    # # print(data, mean)
    # # print(np.sqrt(variance))
    if np.isfinite(loglike):
        return loglike
    else:
        return -np.infty


def ln_chi_squared_cov(data, mean, cov_mat):
    try: 
        inv_cov_mat = np.linalg.inv(cov_mat)
    except:
        return -np.infty
    cov_det = np.linalg.det(cov_mat)

    if cov_det <= 0:
        return -np.infty
    loglike = - 0.5 * (
        np.matmul((data - mean), np.matmul(inv_cov_mat, (data - mean)))
        + np.log(cov_det)
    )
    if np.isfinite(loglike):
        return loglike
    else:
        return -np.infty
