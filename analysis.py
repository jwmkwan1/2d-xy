import numpy as np
import sys
from tqdm import trange # type: ignore

from worm import winding_x, winding_y

def analyze(flows, sources, sinks, K, L):
    """
    Calculates superfluid stiffness and magnetic
    susceptibility given MC data.
    """
    num_samples = np.shape(flows)[0]
    z_space = 0
    w_squared = 0

    # First 25% of samples are thermalization
    for i in trange(num_samples // 4, num_samples):
        if np.all(sources[i] == sinks[i]):
            z_space += 1
            wx = winding_x(flows[i], L)
            wy = winding_y(flows[i], L)
            w_squared += (wx**2 + wy**2)

    # Compute superfluid stiffness
    rho_s = (w_squared / z_space) / (2 * K)

    # Compute susceptibility
    chi = z_space / (num_samples - num_samples // 5)
    
    return rho_s, chi

def do_analysis(K, L, num_samples):
    """
    Perform data analysis on a specific run
    """
    filename = f"data/samples_{K}_{L}_{num_samples}.npy"

    with open(filename, 'rb') as f:
        flows = np.load(f)
        sources = np.load(f)
        sinks = np.load(f)

    print("Starting analysis:")
    rho_s, chi = analyze(flows, sources, sinks, K, L)

    print("Finished")

    print("Results: (K, L, rho_s, chi)")
    print(f"{K}, {L}, {rho_s}, {chi}")

    return rho_s, chi


if __name__ == '__main__':
    """
    Usage: python analysis.py [K_min] [K_max] [num_K] [L] [num_samples]
    """
    K_min = float(sys.argv[1])
    K_max = float(sys.argv[2])
    num_K = int(sys.argv[3])
    L = int(sys.argv[4])
    num_samples = int(sys.argv[5])

    results = np.zeros((3, num_K))
    K_list = np.linspace(K_min, K_max, num_K, endpoint=False)

    for i in range(num_K):
        K_sample = np.round(K_list[i], 3)
        rho_s, chi = do_analysis(K_sample, L, num_samples)

        # Store results
        results[0, i] = K_sample
        results[1, i] = rho_s
        results[2, i] = chi

    filename = f"data/results_{L}_{num_samples}.npy"

    with open(filename, 'wb') as f:
        np.save(f, results)