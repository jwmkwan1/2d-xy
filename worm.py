import numpy as np
import sys
from tqdm import trange # type: ignore

from scipy.special import iv # type: ignore

# Main Monte Carlo update function
def mc_update(flows, source, sink, K, L):
    """
    Implements local Metropolis algorithm on flow configuration,
    with defects source and sink. Uses casework.

    flows: (L, L, 2) array
    """
    flows_prime = np.copy(flows)
    new_source = np.copy(source)
    new_sink = np.copy(sink)

    defect_index = np.random.choice([0, 1])     # chooses which defect to shift
    shift_dir = np.random.choice([0, 1, 2, 3])  # (0, 1, 2, 3) = (x, y, -x, -y)

    # Source shifts
    if defect_index == 0:
        if shift_dir == 0:
            new_source[0] = (source[0] + 1) % L
            Jold = flows[source[0], source[1], 0]
            Jnew = Jold - 1
            flows_prime[source[0], source[1], 0] = Jnew
        elif shift_dir == 1:
            new_source[1] = (source[1] + 1) % L
            Jold = flows[source[0], source[1], 1]
            Jnew = Jold - 1
            flows_prime[source[0], source[1], 1] = Jnew
        elif shift_dir == 2:
            new_source[0] = (source[0] - 1) % L
            Jold = flows[(source[0]-1) % L, source[1], 0]
            Jnew = Jold + 1
            flows_prime[(source[0]-1) % L, source[1], 0] = Jnew
        else:
            new_source[1] = (source[1] - 1) % L
            Jold = flows[source[0], (source[1]-1) % L, 1]
            Jnew = Jold + 1
            flows_prime[source[0], (source[1]-1) % L, 1] = Jnew

    # Sink shifts
    else:
        if shift_dir == 0:
            new_sink[0] = (sink[0] + 1) % L
            Jold = flows[sink[0], sink[1], 0]
            Jnew = Jold + 1
            flows_prime[sink[0], sink[1], 0] = Jnew
        elif shift_dir == 1:
            new_sink[1] = (sink[1] + 1) % L
            Jold = flows[sink[0], sink[1], 1]
            Jnew = Jold + 1
            flows_prime[sink[0], sink[1], 1] = Jnew
        elif shift_dir == 2:
            new_sink[0] = (sink[0] - 1) % L
            Jold = flows[(sink[0]-1) % L, sink[1], 0]
            Jnew = Jold - 1
            flows_prime[(sink[0]-1) % L, sink[1], 0] = Jnew
        else:
            new_sink[1] = (sink[1] - 1) % L
            Jold = flows[sink[0], (sink[1]-1) % L, 1]
            Jnew = Jold - 1
            flows_prime[sink[0], (sink[1]-1) % L, 1] = Jnew

    # Determine acceptance probability
    prob = iv(Jnew, K) / iv(Jold, K)
    prob = min(1, prob)

    if (np.random.rand() < prob):
        return flows_prime, new_source, new_sink
    else:
        return flows, source, sink

def init_flows(K, L):
    """
    Generates a random initial flow configuration.
    """
    source = np.array([0,0], dtype=np.int16)
    sink = np.array([0,0], dtype=np.int16)
    return np.zeros((L, L, 2), dtype=np.int16), source, sink

def gen_samples(K, L, num_samples):
    """
    Generates num_samples Monte Carlo samples.
    """
    flows = np.zeros((num_samples, L, L, 2), dtype=np.int16)
    sources = np.zeros((num_samples, 2), dtype=np.int16)
    sinks = np.zeros((num_samples, 2), dtype=np.int16)

    # Initialize configuration
    init_flow, init_source, init_sink = init_flows(K, L)
    flows[0] = init_flow
    sources[0] = init_source
    sinks[0] = init_sink

    for i in trange(num_samples - 1):
        new_flow, new_source, new_sink = mc_update(flows[i], sources[i], sinks[i], K, L)
        flows[i+1] = new_flow
        sources[i+1] = new_source
        sinks[i+1] = new_sink

    return flows, sources, sinks

def save_data(K, L, num_samples):
    """
    Generates and saves data
    """
    print(f"Generating samples: K = {K}, L = {L}, num_samples = {num_samples}")
    flows, sources, sinks = gen_samples(K, L, num_samples)

    print("")
    print("Saving data...")

    filename = f"data/samples_{K}_{L}_{num_samples}.npy"
    with open(filename, 'wb') as f:
        np.save(f, flows)
        np.save(f, sources)
        np.save(f, sinks)

    print("Done!")
    print("")

# Main script for generating MC data
if __name__ == '__main__':
    """
    Usage: python worm.py [K_min] [K_max] [num_K] [L] [num_samples]
    """
    K_min = float(sys.argv[1])
    K_max = float(sys.argv[2])
    num_K = int(sys.argv[3])
    L = int(sys.argv[4])
    num_samples = int(sys.argv[5])

    K_list = np.linspace(K_min, K_max, num_K, endpoint=False)

    for K in K_list:
        round_K = np.round(K, 3)
        save_data(round_K, L, num_samples)