import numpy as np
import sys
import time
import numba # type: ignore
from scipy.special import iv # type: ignore

from worm import gen_samples_measure

@numba.njit(parallel=True)
def get_results(K_list, L, num_samples, bessels):
    """
    Main run script that doesn't save MC data
    along the way.

    Usage: python main.py [L] [num_samples]
    """
    num_K = len(K_list)
    results = np.zeros((3, num_K))

    for i in numba.prange(num_K):
        bessel = bessels[i]
        K = K_list[i]
        results[0, i] = K
        rho_s, chi = gen_samples_measure(K, L, num_samples, bessel)
        results[1, i] = rho_s
        results[2, i] = chi

    return results

if __name__ == '__main__':
    L = int(sys.argv[1])
    num_samples = int(sys.argv[2])

    K_region1 = np.linspace(0.85, 1, 3, endpoint=False)
    K_region2 = np.array([1, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09])
    K_region3 = np.array([1.1, 1.102, 1.104, 1.106, 1.108, 1.11, 1.112, 1.114, 1.116, 1.118])
    K_region4 = np.linspace(1.12, 1.2, 8, endpoint=False)
    K_region5 = np.linspace(1.2, 2, 16, endpoint=False)

    K_list = np.concatenate((K_region1, K_region2, K_region3, K_region4, K_region5))
    num_K = len(K_list)

    filename = f"data/results_{L}_{num_samples}.npy"

    # Store Bessel functions
    bessels = np.zeros((num_K, 20))
    for i in range(num_K):
        bessels[i] = np.array([iv(J, K_list[i]) for J in range(20)])

    start_time = time.time()
    results = get_results(K_list, L, num_samples, bessels)

    with open(filename, 'wb') as f:
        np.save(f, results)
    end_time = time.time()

    print(f"Done!")
    print(f"Elapsed time: {np.round(end_time - start_time, 2)} seconds")