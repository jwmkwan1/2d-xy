import numpy as np
import sys
from tqdm import trange # type: ignore

# To compute stiffness, need winding numbers
def winding_x(flows, L):
    """
    Calculates the winding in the x-direction
    for a closed flow configuration.
    """
    total_winding = 0
    for i in range(L):
        total_winding += flows[0,i,0]
    return total_winding

def winding_y(flows, L):
    """
    Calculates the winding in the y-direction
    for a closed flow configuration.
    """
    total_winding = 0
    for i in range(L):
        total_winding += flows[i,0,1]
    return total_winding

def analyze(flows, sources, sinks, K, L):
    """
    Calculates superfluid stiffness and magnetic
    susceptibility given MC data.
    """
    num_samples = np.shape(flows)[0]
    z_space = 0
    w_squared = 0

    # First 20% of samples are thermalization
    for i in trange(num_samples // 5, num_samples):
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
    K = float(sys.argv[1])
    L = int(sys.argv[2])
    num_samples = int(sys.argv[3])

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
    Usage: python analysis.py [K] [L] [num_samples]
    """
    K = float(sys.argv[1])
    L = int(sys.argv[2])
    num_samples = int(sys.argv[3])

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