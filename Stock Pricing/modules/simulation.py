import numpy as np

def geo_paths(S, T, r, sigma, steps, N):
    dt = T / steps
    ST = np.log(S) + np.cumsum(((r - sigma ** 2 / 2) * dt + sigma * np.sqrt(dt) * np.random.normal(size=(steps, N))),
                               axis=0)
    return np.exp(ST)

def monte_carlo_simulation(S, K, T, r, sigma, steps, N, option_type):
    paths = geo_paths(S, T, r, sigma, steps, N)
    if option_type == "Call":
        payoffs = np.maximum(paths[-1] - K, 0)
    else:
        payoffs = np.maximum(K - paths[-1], 0)
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price







