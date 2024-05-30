import yfinance as yf
import pandas as pd

def fetch_stock_data(stock_symbol):
    stock_data = yf.download(stock_symbol, start="2013-01-01", end="2023-12-31")
    stock_data.reset_index(inplace=True)
    stock_data.columns = [col.replace(' ', '_') for col in stock_data.columns]
    return stock_data












