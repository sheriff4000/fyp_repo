using AirBorne.Engines.DEDS
using AirBorne
using AirBorne.ETL.YFinance: get_interday_data, get_chart_data, parse_intraday_raw_data
using AirBorne.Markets.StaticMarket: execute_orders!, expose_data, Order, place_order!, executeOrder_CA!
using AirBorne.Strategies.FALM: falm_initialize!, falm_trading_logic!
using AirBorne.Strategies.Markowitz: initialize! as markowitz_initialize!
using AirBorne.Strategies.Markowitz: trading_logic! as markowitz_trading_logic!
using AirBorne.Forecast
using Dates
using AirBorne.Structures: summarizePerformance,TimeEvent, ContextTypeA
using AirBorne.ETL.AssetValuation: stockValuation
using DotMaps
using Latexify
using DataFrames
using Plots
using Suppressor
using Statistics

## Helper functions
function get_VaR(data, alpha=0.05)
    returns = diff(data) ./ data[1:end-1]
    var5 = quantile(returns, alpha)
    return round(var5 * -100, digits=2)
end

function get_ES(data, alpha=0.05)
    returns = diff(data) ./ data[1:end-1]
    return mean(returns[returns .<= quantile(returns, alpha)])
end
function get_annual_returns(data, freq="1d")
    returns = diff(data) ./ data[1:end-1]
    if freq == "1d"
        power = 252
    end
    if freq == "5m"
        power = 252 * 78
    end
    if freq == "60m"
        power = 252 * 6
    end
    annual_returns = (1 + mean(returns))^power - 1
    return round(annual_returns * 100, digits=2)
end
function get_sharpe(data, riskFreeRate=0.04, freq="1d")
    if freq == "1d"
        power = 252
    end
    if freq == "5m"
        power = 252 * 78
    end
    if freq == "60m"
        power = 252 * 6
    end
    annual_std = std(diff(data) ./ data[1:end-1]) * sqrt(power)
    sharpe = (get_annual_returns(data, freq) - riskFreeRate) / annual_std
    return round(sharpe, digits=2)
end
function get_volatility(data, freq="1d")
    if freq == "1d"
        power = 252
    end
    if freq == "5m"
        power = 252 * 78
    end
    if freq == "60m"
        power = 252 * 6
    end
    returns = diff(data) ./ data[1:end-1]
    return round(std(returns) * sqrt(252) * 100, digits=2) 
end
function get_max_drawdown(data)
    returns = diff(data) ./ data[1:end-1]
    cum_returns = cumprod(1 .+ returns)
    max_drawdown = 0
    for i in eachindex(cum_returns)
        for j in i:length(cum_returns)
            drawdown = (cum_returns[j] - cum_returns[i]) / cum_returns[i]
            if drawdown < max_drawdown
                max_drawdown = drawdown
            end
        end
    end
    return round(max_drawdown * -100, digits=2) 
end

## Define Forecasters
SelfLinear1 = LinearForecaster(1; reparameterise_window=45)
ARIMA1 = ArimaForecaster(4, 1, 1; reparameterise_window=52)
Combined1 = CombinedForecaster([SelfLinear1, ARIMA1], [0.8, 0.2])
Combined2 = CombinedForecaster([LinearForecaster(7; reparameterise_window=10), ArimaForecaster(2, 1, 1; reparameterise_window=65)], [0.5, 0.5])
LongLinear = LinearForecaster(4; reparameterise_window=60)

initial_investment = 10050
## Define FALM Strategies


function init_GreedyFALM(ids, tickers)
    GreedyFALM(context) = falm_initialize!(
        context;
        initialCapital = initial_investment ,
        lookahead = 3,
        lpm_order = 2.0,
        max_lookback = 65,
        tickers = tickers,
        assetIDs = ids,
        transactionCost = 0.005,
        httype = :average,
        min_alloc_threshold = 0.9,
        min_returns_threshold= 0.0008,
        forecaster = Combined1
    )
    return GreedyFALM
end

function init_RiskyFALM(ids, tickers)
    RiskyFALM(context) = falm_initialize!(
        context;
        initialCapital = initial_investment ,
        lookahead = 2,
        lpm_order = 0.5,
        max_lookback = 65,
        tickers = tickers,
        assetIDs = ids,
        transactionCost = 0.005,
        httype = :minimum,
        min_alloc_threshold = 0.5,
        min_returns_threshold= 0.0004,
        forecaster = ARIMA1
    )
    return RiskyFALM
end

function init_SafeFALM(ids, tickers)
    SafeFALM(context) = falm_initialize!(
        context;
        initialCapital = initial_investment ,
        lookahead = 1,
        lpm_order = 6.0,
        max_lookback = 65,
        tickers = tickers,
        assetIDs = ids,
        transactionCost = 0.005,
        httype = :minimum,
        min_alloc_threshold = 0.7,
        min_returns_threshold= 0.00005,
        forecaster = LongLinear
    )
    return SafeFALM
end

function init_NeutralFALM(ids, tickers)
    NeutralFALM(context) = falm_initialize!(
        context;
        initialCapital = initial_investment ,
        lookahead = 3,
        lpm_order = 1.0,
        max_lookback = 65,
        tickers = tickers,
        assetIDs = ids,
        transactionCost = 0.005,
        httype = :average,
        min_alloc_threshold = 1.0,
        min_returns_threshold= 0.00015,
        forecaster = SelfLinear1
    )
    return NeutralFALM
end
function init_FALM1(ids, tickers)
    FALM1(context) = falm_initialize!(
        context;
        initialCapital = initial_investment ,
        lookahead = 4,
        lpm_order = 3.0,
        max_lookback = 65,
        tickers = tickers,
        assetIDs = ids,
        transactionCost = 0.005,
        httype = :average,
        min_alloc_threshold = 1.0,
        min_returns_threshold= 0.0005,
        forecaster = Combined2
    )
    return FALM1
end
function init_markowitz(growth_rate)
    Markowitz(context) = markowitz_initialize!(
        context;
        initialCapital = initial_investment,
        min_growth = growth_rate,
        horizon=65
    )
    return Markowitz
end

algorithms_dict = Dict(
    "FALM1" => init_FALM1,
    "RiskyFALM" => init_RiskyFALM,
    "GreedyFALM" => init_GreedyFALM,
    "SafeFALM" => init_SafeFALM,
    "NeutralFALM" => init_NeutralFALM,
    "MeanVariance" => init_markowitz
)
algorithms_list = [init_FALM1, init_RiskyFALM, init_GreedyFALM, init_SafeFALM, init_NeutralFALM, init_markowitz]

## Define the datasets
unix(x) = string(round(Int, datetime2unix(DateTime(x))))

tech = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
health = ["PFE", "MRK", "JNJ", "ABBV", "UNH"]
diverse = ["ACWI", "VEU", "VWO", "EFA", "BND"]
growth = ["ARKK", "QQQ", "VOO", "VGT", "IWF"]
value = ["VTV", "IVE", "SPYV", "SCHV", "VOE"]
extra = ["^GSPC"]


# getdata(tickers, start, finish, freq) = parse_intraday_raw_data(get_chart_data("^GSPC", start, finish, freq))

function getdata(tickers, start, finish, freq)
    if freq == "1d" 
        return get_interday_data(tickers, start, finish)
    end
    stocks = DataFrames.DataFrame()
    for t in tickers
        data = parse_intraday_raw_data(get_chart_data(t, start, finish, freq))
        stocks = DataFrames.vcat(stocks, data)
    end
    return stocks
end


A = (unix("2024-05-15"), unix("2024-05-22"), "5m")
B = (unix("2024-05-23"), unix("2024-05-30"), "5m")
C = (unix("2024-04-12"), unix("2024-05-12"), "60m")
D = (unix("2024-05-12"), unix("2024-06-12"), "60m")
E = (unix("2020-01-01"), unix("2022-01-01"), "1d")
F = (unix("2022-01-01"), unix("2024-01-01"), "1d")

dataset_A = Dict(
    "Technology" => getdata(tech, A...),
    "Healthcare" => getdata(health, A...),
    "Diversified Global" => getdata(diverse, A...),
    # "growth" => getdata(growth, A...),
    # "value" => getdata(value, A...),
    "SP 500" => getdata(extra, A...)
)
dataset_B = Dict(
    "Technology" => getdata(tech, B...),
    "Healthcare" => getdata(health, B...),
    "Diversified Global" => getdata(diverse, B...),
    "Growth" => getdata(growth, B...),
    # "value" => getdata(value, B...),
    "SP 500" => getdata(extra, B...)
)
dataset_C = Dict(
    "Technology" => getdata(tech, C...),
    "Healthcare" => getdata(health, C...),
    "Diversified Global" => getdata(diverse, C...),
    "Growth" => getdata(growth, C...),
    "Value" => getdata(value, C...),
    "SP 500" => getdata(extra, C...)
)
dataset_D = Dict(
    "Technology" => getdata(tech, D...),
    "Healthcare" => getdata(health, D...),
    "Diversified Global" => getdata(diverse, D...),
    "Growth" => getdata(growth, D...),
    "Value" => getdata(value, D...),
    "SP 500" => getdata(extra, D...)
)
dataset_E = Dict(
    "Technology" => getdata(tech, E...),
    "Healthcare" => getdata(health, E...),
    "Diversified Global" => getdata(diverse, E...),
    "Growth" => getdata(growth, E...),
    "Value" => getdata(value, E...),
    "SP 500" => getdata(extra, E...)
)
dataset_F = Dict(
    "Technology" => getdata(tech, F...),
    "Healthcare" => getdata(health, F...),
    "Diversified Global" => getdata(diverse, F...),
    "Growth" => getdata(growth, F...),
    "Value" => getdata(value, F...),
    "SP 500" => getdata(extra, F...)
)
algo_names = ["FALM1", "RiskyFALM", "GreedyFALM", "SafeFALM", "NeutralFALM", "Mean Variance"]
datasets = Dict("A" => dataset_A, "B" => dataset_B, "C" => dataset_C, "D" => dataset_D, "E" => dataset_E, "F" => dataset_F)

feeStructure=Vector{Dict}([Dict("FeeName" => "SaleCommission", "fixedPrice" => 0.0, "variableRate" => 0.005)])
singleExecutionFun(context, order, data) = executeOrder_CA!(context, order, data;defaultFeeStructures=feeStructure,partialExecutionAllowed=false)
my_execute_orders!(context, data) = execute_orders!(context, data; propagateBalanceToPortfolio=true, executeOrder=singleExecutionFun)
function augment_data(data)
    dollar_symbol = "FEX/USD"
    usdData = deepcopy(data[data.symbol .== data.symbol[1], :])
    usdData[!, "assetID"] .= dollar_symbol
    usdData[!, "exchangeName"] .= "FEX"
    usdData[!, "symbol"] .= "USD"
    usdData[!, [:close, :high, :low, :open]] .= 1.0
    usdData[!, [:volume]] .= 0
    out_data = vcat(data, usdData)

    return out_data
end
markowitz_expose_data(context, data) = expose_data(context, data; historical=false)
function run_algo_on_dataset(dataset, algorithm, evalEvents; markowitz_growth=0.001)
    if algorithm == init_markowitz
        test_algo = algorithm(markowitz_growth)
        algo_context = @suppress AirBorne.Engines.DEDS.run(
            dataset,
            test_algo,
            markowitz_trading_logic!,
            my_execute_orders!,
            markowitz_expose_data;
            audit=true,
            verbose=false,
            initialEvents=evalEvents
        )

        result = @suppress summarizePerformance(dataset, algo_context)
    else
        test_algo = algorithm(unique(dataset.assetID), unique(dataset.symbol))
        algo_context = @suppress AirBorne.Engines.DEDS.run(
            dataset,
            test_algo,
            falm_trading_logic!,
            my_execute_orders!,
            expose_data;
            audit=true,
            verbose=false,
            initialEvents=evalEvents
        )
        OHLCV_data = augment_data(dataset)
        result = @suppress summarizePerformance(OHLCV_data, algo_context; includeAccounts=false, riskFreeRate=0.04)
    end
    # test_algo = algorithm(unique(dataset.assetID), unique(dataset.symbol))

    return result.dollarValue
end

for (letter, set) in datasets
    println("Running tests on dataset $letter")
    sp500 = set["SP 500"].close[65:end]
    sp500_returns = diff(sp500) ./ sp500[1:end-1]
    sp500_prices = cumprod(1 .+ sp500_returns) .* (initial_investment * 0.995)
    if letter == "A" || letter == "B"
        freq = "5m"
        growth_rate = 3.8e-7
    end
    if letter == "C" || letter == "D"
        freq = "60m"
        growth_rate = 4.6e-6
    end
    if letter == "E" || letter == "F"
        freq = "1d"
        growth_rate = 0.0001
    end
    for (name, info) in set
        println("Running tests on dataset $name")
        if name == "SP 500"
            continue
        end
        ## Initialize latexify table
        # table = ["Algorithm" "Annual Return"  "Sharpe Ratio" "Max DD" "Volatility" "VaR"]'
        table = []
        ## Initialize the plot
        dates = unique(info.date)
        ticks = Dates.format.(dates, "dd-mm-yyyy")
        step_size = round(Int, length(dates) / 10)
        date_idx =1:step_size:length(ticks)
        p = plot(size=(1000, 500), title="FALM Portfolio Optimisation Performance On Dataset $(letter) with $(name) Portfolio", xlabel="Date", ylabel="Dollar Value", leftmargin=5Plots.mm, bottommargin=10Plots.mm, 
        legend=:topleft, grid=:on, frame=:box, xrotation=45, xticks=(1:step_size:length(dates),ticks[end-length(dates)+1:step_size:end]))

        for (algo_idx, algo) in enumerate(algorithms_list)
            println("Running tests on algorithm $algo_idx")
            evaluationEvents = [TimeEvent(t, "data_transfer") for t in sort(unique(info.date); rev=true)]
            if algo == init_markowitz
                series = run_algo_on_dataset(info, algo, evaluationEvents; markowitz_growth=growth_rate)
            else
                series = run_algo_on_dataset(info, algo, evaluationEvents)
            end
            
            ## Add the results to the table
            row = [algo_idx get_annual_returns(series, freq) get_sharpe(series, 0.04, freq) get_max_drawdown(series) get_volatility(series, freq) get_VaR(series)]
            if table == []
                table = row'
            else
                table = vcat(table, row')
            end
            ## Add the results to the plot
            plot!(p, series[67:end], label=algo_names[algo_idx], linewidth=2)
        end

        ## Add the results for the sp500 to table
        row = [7.0 get_annual_returns(sp500_prices, freq) get_sharpe(sp500_prices, 0.04, freq) get_max_drawdown(sp500_prices) get_volatility(sp500_prices, freq) get_VaR(sp500_prices)]
        table = vcat(table, row')

        ## Plot the sp500
        plot!(p, sp500_prices, label="S&P 500", linewidth=2, linestyle=:dash)


        ## Plot the homogenous portfolio
        tickers = unique(info.symbol)
        homogenous_weight = 1.0 / length(tickers)
        homogenous_prices = nothing
        for ticker in tickers
            stock = info[info.symbol .== ticker, :close][65:end]
            stock_returns = diff(stock) ./ stock[1:end-1]
            stock_prices = cumprod(1 .+ stock_returns) .* (0.995 * initial_investment)
            if isnothing(homogenous_prices)
                homogenous_prices = (stock_prices .* homogenous_weight)
            else
                homogenous_prices .+= (stock_prices .* homogenous_weight)
            end
        end
        plot!(p, homogenous_prices, label="Homogenous Portfolio", linewidth=2, linestyle=:dash)

        ## Add the results for the homogenous portfolio to table
        row = [8.0 get_annual_returns(homogenous_prices, freq) get_sharpe(homogenous_prices, 0.04, freq) get_max_drawdown(homogenous_prices) get_volatility(homogenous_prices, freq) get_VaR(homogenous_prices)]
        table = vcat(table, row')

        ## Save the table
        latex_table = latexify(table, env=:table, latex=false)
        open("falm_plots/test/$(name*letter)_table.tex", "w") do f
            write(f, latex_table)
        end

        ## Save the plot
        savefig(p, "falm_plots/test/$(name*letter)_plot.svg")
    end
end
