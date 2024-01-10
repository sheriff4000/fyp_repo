import yfinance as yf
import pandas as pd
import numpy as np

data = yf.download("SPY", start="2023-01-01", end="2023-11-30")
#define portfolio as an array of times [100day, 50day, 25day, 10day, out of market]
portfolio = [0, 0, 0, 0, 1]
print(data)
