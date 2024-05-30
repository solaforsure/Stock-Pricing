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







