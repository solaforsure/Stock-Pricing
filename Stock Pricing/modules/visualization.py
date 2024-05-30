import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from modules.black_scholes import BlackScholesGreeks

def plot_distribution(paths, K):
    final_prices = paths[-1]
    final_prices = final_prices[~np.isnan(final_prices)]
    if len(final_prices) == 0:
        st.error("All values are NaN. Check the parameters for generating paths.")
        return

    plt.figure(figsize=(10, 5))
    n, bins, patches = plt.hist(final_prices, bins=250)
    for c, p in zip(bins, patches):
        if c > K:
            plt.setp(p, 'facecolor', 'green')
        else:
            plt.setp(p, 'facecolor', 'red')

    plt.axvline(K, color='black', linestyle='dashed', linewidth=2, label="Strike")
    plt.title("Distribution of $S_{T}$")
    plt.xlabel("$S_{T}$")
    plt.ylabel('Count')
    plt.legend()
    st.pyplot(plt)

def plot_greeks(S, K, T, r, sigma, option_type):
    deltas = [BlackScholesGreeks(s, K, T, r, sigma).delta_call() if option_type == "Call" else BlackScholesGreeks(s, K, T, r, sigma).delta_put() for s in S]
    gammas = [BlackScholesGreeks(s, K, T, r, sigma).gamma() for s in S]
    vegas = [BlackScholesGreeks(s, K, T, r, sigma).vega() for s in S]
    thetas = [BlackScholesGreeks(s, K, T, r, sigma).theta_call() if option_type == "Call" else BlackScholesGreeks(s, K, T, r, sigma).theta_put() for s in S]
    rhos = [BlackScholesGreeks(s, K, T, r, sigma).rho_call() if option_type == "Call" else BlackScholesGreeks(s, K, T, r, sigma).rho_put() for s in S]

    plt.figure(figsize=(10, 6))
    plt.plot(S, deltas, label='Delta')
    plt.plot(S, gammas, label='Gamma')
    plt.plot(S, vegas, label='Vega')
    plt.plot(S, thetas, label='Theta')
    plt.plot(S, rhos, label='Rho')
    plt.xlabel('Stock Price')
    plt.ylabel('Greek Value')
    plt.title('Option Greeks')
    plt.legend()
    st.pyplot(plt)



