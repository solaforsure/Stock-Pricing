# Stock Price Simulation, Option Greeks Analysis, and SQL Analysis

This project is a Streamlit application that performs stock price simulation, option Greeks analysis, Monte Carlo simulation, and SQL data querying. It utilizes various Python libraries such as Pandas for data manipulation, SQLAlchemy for database interactions, and Matplotlib for visualization.

## Introduction

The Black-Scholes model, developed by Fischer Black, Myron Scholes, and Robert Merton in the early 1970s, provides a theoretical estimate of European option prices. The Black-Scholes formula considers factors such as the current stock price, the option's strike price, time to expiration, the risk-free rate, and the asset's volatility to calculate the option premium.

Monte Carlo simulations are computational algorithms that rely on repeated random sampling to obtain numerical results. In the context of option pricing, Monte Carlo simulations model the behavior of stock prices under stochastic processes and estimate the value of options. This method is particularly useful for pricing complex derivatives and options with path-dependent features.

This research aims to implement European option pricing using both the Monte Carlo method and the Black-Scholes model. By gradually increasing the number of paths and the number of steps per path (reducing the step size) in the Monte Carlo simulation, we can verify that its results (prices) converge to those obtained from the Black-Scholes model.

In this tutorial, we will cover the following key areas:
- Understanding the Black-Scholes formula and its components.
- Using a library to obtain real options data with `yfinance`.
- Implementing the Black-Scholes model in Python using an object-oriented approach.
- Visualizing option prices and Greeks with impressive charts.
- Analyzing the sensitivity of option prices to various parameters.
- Using Monte Carlo simulations for option pricing.

## Installation

First, set up the Python environment and install the necessary libraries:
```sh
pip install -r requirements.txt
```

## Obtaining Real Options Data

To apply the Black-Scholes model, we need real financial data. We will use the `yfinance` library to download options data from major financial institutions such as JPMorgan Chase & Co. (JPM).

First, define a function to get options data for a given stock symbol:

```python
import yfinance as yf

def fetch_options_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    options_dates = ticker.options
    options_data = ticker.option_chain(options_dates[0])
    return options_data.calls, options_data.puts
```

### Example Usage:
```python
jpm_calls, jpm_puts = fetch_options_data('JPM')
```

## Visualizing Historical Stock Prices

```python
import matplotlib.pyplot as plt

def plot_historical_prices(stock_data, title='Historical Stock Prices'):
    plt.figure(figsize=(10, 5))
    plt.plot(stock_data['Close'])
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.grid(True)
    plt.show()

# Fetch and plot JPM stock data
jpm_stock_data = yf.download('JPM', start='2023-01-01', end='2023-12-31')
plot_historical_prices(jpm_stock_data, title='JPM Historical Stock Prices')
```

## Implementing the Black-Scholes Model

The Black-Scholes model is a mathematical model that provides a theoretical estimate of European option prices. Let's implement the Black-Scholes formula in Python.

```python
import numpy as np
from scipy.stats import norm

class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma):
        self.S = S        # Stock price
        self.K = K        # Strike price
        self.T = T        # Time to maturity (in years)
        self.r = r        # Risk-free interest rate
        self.sigma = sigma  # Volatility of the underlying asset

    def d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)
    
    def call_option_price(self):
        return (self.S * norm.cdf(self.d1()) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2()))
    
    def put_option_price(self):
        return (self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2()) - self.S * norm.cdf(-self.d1()))
```

### Example Usage:
```python
bsm = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2)
print(f"Call Option Price: {bsm.call_option_price()}")
print(f"Put Option Price: {bsm.put_option_price()}")
```

By implementing the Black-Scholes model, we can now evaluate options for a selected stock. However, to do this accurately, we need to estimate the volatility of the underlying asset (sigma). Volatility measures the variation in the prices of an asset over time, typically calculated as the annualized standard deviation of the asset's returns.

## Implementing the Monte Carlo Method

The Monte Carlo method is a statistical technique used to model the probability of different outcomes in a process that cannot easily be predicted due to the intervention of random variables. For European option pricing, the Monte Carlo method simulates the underlying asset's price paths to estimate the option's price.

```python
import numpy as np

class MonteCarloOptionPricing:
    def __init__(self, S, K, T, r, sigma, num_simulations, num_steps):
        self.S = S              # Stock price
        self.K = K              # Strike price
        self.T = T              # Time to maturity (in years)
        self.r = r              # Risk-free interest rate
        self.sigma = sigma      # Volatility of the underlying asset
        self.num_simulations = num_simulations  # Number of simulations
        self.num_steps = num_steps  # Number of steps in each simulation

    def simulate_paths(self):
        dt = self.T / self.num_steps
        paths = np.zeros((self.num_simulations, self.num_steps + 1))
        paths[:, 0] = self.S

        for t in range(1, self.num_steps + 1):
            z = np.random.standard_normal(self.num_simulations)
            paths[:, t] = paths[:, t-1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z)

        return paths

    def call_option_price(self):
        paths = self.simulate_paths()
        payoff = np.maximum(paths[:, -1] - self.K, 0)
        return np.exp(-self.r * self.T) * np.mean(payoff)

    def put_option_price(self):
        paths = self.simulate_paths()
        payoff = np.maximum(self.K - paths[:, -1], 0)
        return np.exp(-self.r * self.T) * np.mean(payoff)
```

### Example Usage:
```python
mc = MonteCarloOptionPricing(S=100, K=100, T=1, r=0.05, sigma=0.2, num_simulations=10000, num_steps=100)
print(f"Call Option Price: {mc.call_option_price()}")
print(f"Put Option Price: {mc.put_option_price()}")
```

By implementing the Monte Carlo method, we can estimate option prices by simulating numerous potential future paths for the underlying asset's price. This method allows us to verify the convergence of the Monte Carlo results to the Black-Scholes model's prices by increasing the number of simulations and steps, thus demonstrating the robustness and reliability of Monte Carlo simulations in option pricing.
## Calculating Historical Volatility

```python
def calculate_historical_volatility(stock_data, window=252):
    log_returns = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
    volatility = np.sqrt(window) * log_returns.std()
    return volatility 

jpm_volatility = calculate_historical_volatility(jpm_stock_data)
print(f"Historical Volatility of JPM: {jpm_volatility}")
```

Now that we have historical volatility, we can use it as an estimate for sigma in the Black-Scholes model.

## Visualizing Option Prices and Greeks

Option prices are not the only important indicators in options trading. "Greeks" are also crucial as they measure the sensitivity of option prices to various factors. The most common Greeks are Delta, Gamma, Theta, Vega, and Rho.

### Understanding the Greeks
- **Delta**: Measures the change in the option price with respect to the change in the underlying asset price. For call options, Delta ranges from 0 to 1, while for put options, Delta ranges from -1 to 0.
- **Gamma**: Measures the change in Delta with respect to the change in the underlying asset price. Gamma is important as it affects how Delta changes with price movements.
- **Theta**: Measures the decrease in the option's value over time (time decay). Theta is usually negative, indicating that the value decreases with time.
- **Vega**: Measures the sensitivity of the option price to the volatility of the underlying asset. Vega indicates how much the option price will change for a 1% change in volatility.
- **Rho**: Measures the sensitivity of the option price to the risk-free interest rate. For call options, higher interest rates generally increase the option's value, and the opposite is true for put options.

### Calculating and Visualizing Greeks

Let's extend the Black-Scholes model to calculate these Greeks.

```python
class BlackScholesGreeks(BlackScholesModel):
    def delta_call(self):
        return norm.cdf(self.d1())

    def delta_put(self):
        return -norm.cdf(-self.d1())
    
    def gamma(self):
        return norm.pdf(self.d1()) / (self.S * self.sigma * np.sqrt(self.T))

    def theta_call(self):
        return (-self.S * norm.pdf(self.d1()) * self.sigma / (2 * np.sqrt(self.T)) 
                - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2()))
    
    def theta_put(self):
        return (-self.S * norm.pdf(self.d1()) * self.sigma / (2 * np.sqrt(self.T)) 
                + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2()))

    def vega(self):
        return self.S * norm.pdf(self.d1()) * np.sqrt(self.T)
    
    def rho_call(self):
        return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2())
    
    def rho_put(self):
        return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2())
```

### Example Usage:
```python
bsg = BlackScholesGreeks(S=100, K=100, T=1, r=0.05, sigma=0.2)
print(f"Call Delta: {bsg.delta_call()}")
print(f"Put Delta: {bsg.delta_put()}")
```

To visualize how these Greeks change with the underlying asset's price, we can plot them.

### Defining a Range of Stock Prices

```python
stock_prices = np.linspace(80, 120, 100)
deltas = [BlackScholesGreeks(S=price, K=100, T=1, r=0.05, sigma=0.2).delta_call() for price in stock_prices]

plt.figure(figsize=(10, 5))
plt.plot(stock_prices, deltas)
plt.title('Call Option Delta vs. Stock Price')
plt.xlabel('Stock Price')
plt.ylabel('Delta')
plt.grid(True)
plt.show()
```

Similarly, we can plot Gamma, Theta, Vega, and Rho for a range of stock prices.

## Analyzing Option Price Sensitivity

The Black-Scholes model assumes that volatility and interest rates are constant, but this is not always the case in the real world. Therefore, it is crucial to analyze the sensitivity of option prices to these parameters.

Let's create a function to plot option prices against different volatilities and interest rates.

```python
def plot_option_sensitivity(bs_model, parameter, values, option_type='call'):
    prices = []
    for value in values:
        setattr(bs_model, parameter, value)
        if option_type == 'call':
            prices.append(bs_model.call_option_price())
        else:
            prices.append(bs_model.put_option_price())

    plt.figure(figsize=(10, 5))
    plt.plot(values, prices)
    plt.title(f'Sensitivity of Option Price to {parameter.capitalize()}')
    plt.xlabel(parameter.capitalize())
    plt.ylabel('Option Price')
    plt.grid(True)
    plt.show()
```

### Example Usage:

```python
volatilities = np.linspace(0.1, 0.3, 100)
plot_option_sensitivity(bsm, 'sigma', volatilities, 'call')
```

We can do the same for interest rates and put options:

```python
interest_rates = np.linspace(0.01, 0.10, 100)
plot_option_sensitivity(bsm, 'r', interest_rates, 'put')
```

## Main Application: Streamlit Interface

### main.py
```python
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
import os
from sklearn.linear_model import LinearRegression
from modules.data_fetching import fetch_stock_data
from modules.black_scholes import BlackScholesGreeks, black_scholes_call, black_scholes_put
from modules.simulation import geo_paths, monte_carlo_simulation
from modules.visualization import plot_distribution, plot_greeks
from modules.database import create_connection, load_data_to_db, initialize_database
from modules.helpers import load_config

def main():
    st.title("Stock Price Simulation, Option Greeks Analysis, and SQL Analysis")

    # Load configurations
    current_dir = os.getcwd()
    config_path = os.path.join(current_dir, "config.toml")
    config = load_config(config_path)
    db_uri = config["database"]["uri"]

    # Initialize database
    initialize_database(db_uri)

    # User input for stock symbol
    stock_symbol = st.sidebar.text_input("Enter stock symbol (e.g., AAPL):", config["defaults"]["stock_symbol"])
    stock_data = fetch_stock_data(stock_symbol)

    if stock_data.empty:
        st.error("No data found for the given stock symbol. Please check the symbol and try again.")
        return

    st.write("Stock Data:", stock_data)
    st.line_chart(stock_data['Adj_Close'])

    # Store stock data in database
    conn = create_connection(db_uri)
    load_data_to_db(stock_data, 'stocks', conn)

    # Option parameters
    S = stock_data['Adj_Close'].iloc[-1]
    K = st.sidebar.number_input("Strike Price:", value=config["defaults"]["strike_price"])
    T = st.sidebar.number_input("Time to Maturity (in years):", value=config["defaults"]["time_to_maturity"])
    r = st.sidebar.number_input("Risk-Free Rate:", value=config["defaults"]["risk_free_rate"])
    sigma = st.sidebar.number_input("Volatility (annual):", value=config["defaults"]["volatility"])
    steps = st.sidebar.number_input("Number of Steps:", value=config["defaults"]["steps"])
    N = st.sidebar.number_input("Number of Trials:", value=config["defaults"]["trials"])
    option_type = st.sidebar.selectbox("Option Type:", ["Call", "Put"])

    # Generate and plot stock price paths
    paths = geo_paths(S, T, r, sigma, steps, N)
    st.subheader("Simulated Stock Price Paths")
    plt.figure(figsize=(10, 5))
    for i in range(N):
        plt.plot(paths[:, i], lw=0.5)
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    plt.title("Simulated Stock Price Paths")
    st.pyplot(plt)

    # Plot stock price distribution at maturity
    st.subheader("Distribution of Stock Prices at Maturity")
    plot_distribution(paths, K)

    # Option Greeks analysis
    st.subheader("Option Greeks Analysis")
    stock_prices = np.linspace(S * 0.5, S * 1.5, 100)
    plot_greeks(stock_prices, K, T, r, sigma, option_type)

    # Compare Monte Carlo and Black-Scholes prices
    st.subheader("Monte Carlo Simulation vs Black-Scholes Price")
    mc_price = monte_carlo_simulation(S, K, T, r, sigma, steps, N, option_type)
    bs_price = black_scholes_call(S, K, T, r, sigma) if option_type == "Call" else black_scholes_put(S, K, T, r, sigma)
    st.write(f"Monte Carlo Simulated Price ({option_type}): {mc_price}")
    st.write(f"Black-Scholes Price ({option_type}): {bs_price}")

    # Regression analysis
    st.subheader("Regression Analysis")
    st.write("Performing regression analysis on the stock data.")
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data['Days'] = (stock_data['Date'] - stock_data['Date'].min()).dt.days
    X = stock_data['Days'].values.reshape(-1, 1)
    y = stock_data['Adj_Close'].values
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    st.write("Regression Coefficient (Slope):", reg.coef_[0])
    st.write("Regression Intercept:", reg.intercept_)
    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, color='blue', label='Actual Stock Prices')
    plt.plot(X, y_pred, color='red', label='Regression Line')
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.title("Stock Price Regression Analysis")
    plt.legend()
    st.pyplot(plt)

    # SQL analysis
    st.subheader("SQL Analysis")
    query = st.text_area("Enter SQL query here", "SELECT * FROM stocks LIMIT 10")
    if st.button("Run Query"):
        query_result = pd.read_sql_query(query, conn)
        query_result.reset_index(drop=True, inplace=True)  # Reset index and drop the original index
        query_result = query_result.loc[:, ~query_result.columns.str.contains('^index')]  # Remove any index column
        st.write("Query Result:")
        st.write(query_result)
        if 'Date' in query_result.columns and 'Volume' in query_result.columns:
            st.write("Data Visualization:")
            plt.figure(figsize=(10, 5))
            plt.plot(pd.to_datetime(query_result['Date']), query_result['Volume'])
            plt.xlabel("Date")
            plt.ylabel("Volume")
            plt.title("Volume over Time")
            plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
            st.pyplot(plt)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(pd.to_datetime(query_result['Date']), query_result['Open'], label='Open', color='blue')
            ax.plot(pd.to_datetime(query_result['Date']), query_result['High'], label='High', color='green')
            ax.plot(pd.to_datetime(query_result['Date']), query_result['Low'], label='Low', color='red')
            ax.plot(pd.to_datetime(query_result['Date']), query_result['Close'], label='Close', color='black')

            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.set_title("Stock Prices Over Time")
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)


if __name__ == "__main__":
    main()

```
# Sources
This project was developed using a variety of key sources and references to ensure accuracy and effectiveness in implementing the Black-Scholes model and Monte Carlo method for European option pricing:

- Hull, J. (2018). Options, Futures, and Other Derivatives. This book provided fundamental insights into the theoretical underpinnings of the Black-Scholes model.
- Shreve, S. E. (2004). Stochastic Calculus for Finance II: Continuous-Time Models. This text was instrumental in understanding the mathematical foundation of Monte Carlo simulations in financial contexts.
- Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering. This book offered practical approaches and advanced techniques for implementing Monte Carlo methods in option pricing.
- Relevant Research Papers and Articles. Various academic papers and online articles were referenced to stay updated on the latest developments and best practices in financial modeling.