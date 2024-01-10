using Pkg; Pkg.add("Suppressor"); using Suppressor: @suppress;
@suppress Pkg.add(url="https://github.com/sheriff4000/AirBorne.jl#SA_FYP")
@suppress Pkg.add(["Dates","Plots","DataFrames"])
@suppress include("./EMA.jl")
@info "Dependencies added"

# Fetch OHLCV data from YahooFinance on Apple and Google tickers
using AirBorne.ETL.YFinance: get_interday_data
using Dates: DateTime,datetime2unix
unix(x) = string(round(Int, datetime2unix(DateTime(x))))
data = get_interday_data(["AAPL","GOOG"], unix("2017-01-01"), unix("2022-01-01"))
first(data,4) # Display first 4 rows
