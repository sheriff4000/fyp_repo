"""
    SMA (Simple Moving Average)

    This is a standard strategy that can be implemented in several ways.

    1. Crossover Simple Moving Average: Define 2 time windows, a long one and a short one.
    If the Average during the short one is greater than over the long one this implies that the price  is going up.
    So a long position is desired, however if its smaller then this indicates a falling price and a short position 
    is desired.

    In this Strategy an optimization of hyperparameters will be available. The optimization will need an objective function and 
    maybe constraints.

    The design of this strategy is inspired by the lecture 2 of Algorithmic Trading, original source obtained from  [Algorithmic Trading Society Lectures Imperial College London](https://github.com/algotradingsoc/Lectures2022/blob/main/AlgoTradeSocLectures.ipynb)

"""
module EMA
using AirBorne.Utils: sortedStructInsert!
using AirBorne.Structures: ContextTypeA, TimeEvent, nextDay!
using AirBorne.Markets.StaticMarket: Order, place_order!
using Dates: Day
using DataFrames: DataFrame, groupby, combine, mean
using DotMaps: DotMap
using DirectSearch

"""
    initialize!

    Template for the initialization procedure, before being passed onto an engine like DEDS a preloaded
    function must be defined so that the initialization function meets the engine requirements.
    
    ```julia
    # Specify custom arguments to tune the behaviour of SMA
    my_initialize!(context,data) = SMA.initialize!(context;...)
    # Or just run with the default parameters
    my_initialize!(context,data) = SMA.trading_logic!(context)
    ```
"""
function interday_initialize!(
    context::ContextTypeA;
    longHorizon::Real=100,
    shortHorizon::Real=10,
    initialCapital::Real=10^5,
    nextEventFun::Union{Function,Nothing}=nothing,
)
    context.extra.long_horizon = longHorizon
    context.extra.short_horizon = shortHorizon

    ###################################
    ####  Specify Account Balance  ####
    ###################################
    context.accounts.usd = DotMap(Dict())
    context.accounts.usd.balance = initialCapital
    context.accounts.usd.currency = "USD"

    #########################################
    ####  Define first simulation event  ####
    #########################################
    if !(isnothing(nextEventFun))
        nextEventFun(context)
    end
    return nothing
end

"""
    interday_trading_logic!(context::ContextTypeA, data::DataFrame)

    Template for the trading logic algorithm, before being passed onto an engine like DEDS a preloaded
    function must be defined so that the trading logic function meets the engine requirements.

    ```julia
    # Specify custom arguments to tune the behaviour of SMA
    my_trading_logic!(context,data) = SMA.trading_logic!(context,data;...)
    # Or just run with the default parameters
    my_trading_logic!(context,data) = SMA.trading_logic!(context,data)
    ```
"""
function interday_trading_logic!(
    context::ContextTypeA, data::DataFrame; nextEventFun::Union{Function,Nothing}=nothing
)
    # 1. Specify next event (precalculations can be specified here) 
    if !(isnothing(nextEventFun))
        nextEventFun(context)
    end

    # 2. Generate orders and  place orders
    if size(data, 1) < context.extra.long_horizon # Skip if not enough data
        return nothing
    end

    # SMA Calculations: This assumes the data of the subdataframe comes pre-sorted with newest results last.

    function ema(prices, window)
        multiplier::Float64 = (2/(window+1))
        prev = prices[1]
        emas = [prev]
        if window > 1
            for i in range(2,length(prices))
                val = multiplier * (prices[i-1] + prev)
                push!(emas,val)
                prev = val
            end
            push!(emas, multiplier * (prices[end] + prev))
        end
        return emas
    end

    shortEMA(sdf_col) = ema(sdf_col, context.extra.short_horizon)[end]
    
    longEMA(sdf_col) = ema(sdf_col, context.extra.long_horizon)[end]

    ema_df = combine(
        groupby(data, ["symbol", "exchangeName"]),
        :close => shortEMA => :EMA_S,
        :close => longEMA => :EMA_L,
    )
    ema_df[!, :position] = ((ema_df.EMA_S .>= ema_df.EMA_L) .- 0.5) .* 2

    
        # 1. Specify next event (precalculations can be specified here) 
        if !(isnothing(nextEventFun))
            nextEventFun(context)
        end

        # 2. Generate orders and  place orders
        if size(data, 1) < context.extra.long_horizon # Skip if not enough data
            return nothing
        end

        # Order Generation
        for r in eachrow(ema_df)
            assetID = r.exchangeName * "/" * r.symbol
            if r.position > 0 # Set Portfolio to 100 Shares on ticker under a bullish signal
                amount = 100 - get(context.portfolio, assetID, 0)
            elseif r.position < 10
                amount = get(context.portfolio, assetID, 0) * -1
            end
            if amount === 0
                continue
            end
            order_specs = DotMap(Dict())
            order_specs.ticker = r.symbol
            order_specs.shares = amount # Can be replaced by r.amount
            order_specs.type = "MarketOrder"
            order_specs.account = context.accounts.usd
            order =

    # Order Generation
    for r in eachrow(ema_df)
        assetID = r.exchangeName * "/" * r.symbol
        if r.position > 0 # Set Portfolio to 100 Shares on ticker under a bullish signal
            amount = 100 - get(context.portfolio, assetID, 0)
        elseif r.position < 10
            amount = get(context.portfolio, assetID, 0) * -1
        end
        if amount === 0
            continue
        end
        order_specs = DotMap(Dict())
        order_specs.ticker = r.symbol
        order_specs.shares = amount # Can be replaced by r.amount
        order_specs.type = "MarketOrder"
        order_specs.account = context.accounts.usd
        order = Order(r.exchangeName, order_specs)
        place_order!(context, order)
    end
    return nothing
end

end
end