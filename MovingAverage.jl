"""
This module is a Wrapper for moving average trading strategies
Will enable the user to specify the following parameters:
    - short horizon
    - long horizon
    - weighting type (simple MA, Exponential, specified by user)
    - position type (long, short, both)
"""

module MovingAverage

using AirBorne.Utils: sortedStructInsert!
using AirBorne.Structures: ContextTypeA, TimeEvent, nextDay!
using AirBorne.Markets.StaticMarket: Order, place_order!
using Dates: Day
using DataFrames: DataFrame, groupby, combine, mean
using DotMaps: DotMap

"""
initialize!

Template for the initialization procedure, before being passed onto an engine like DEDS a preloaded
function must be defined so that the initialization function meets the engine requirements.

averagingType can be one of the following:
    - :simple
    - :exponential
    - :custom
averagingFun must be specified if averagingType is set to :custom. 
It must be a function that takes in a column of data and a window size and returns a single value.

```julia
# Specify custom arguments to tune the behaviour of SMA
my_initialize!(context,data) = SMA.initialize!(context;...)
# Or just run with the default parameters
my_initialize!(context,data) = SMA.trading_logic!(context)
```
"""
function interday_initialize!(
    context::ContextTypeA;
    longHorizon::Real=100,  #union  with Nothing,use custom weights length
    shortHorizon::Real=10,
    initialCapital::Real=10^5,
    nextEventFun::Union{Function,Nothing}=nothing,
    averagingType::Union{Symbol,Nothing}=:simple,
    customWeights::Union{Vector{Vector{Float64}},Nothing}=nothing,
)
    context.extra.long_horizon = longHorizon
    context.extra.short_horizon = shortHorizon
    context.extra.averaging_type = averagingType
    if averagingType == :custom && isnothing(customWeights)
        error("averagingType is set to :custom but customWeights is not defined")
    end
    if averagingType == :custom
        if length(customWeights[1]) != shortHorizon
            error("customWeights[1] must be of length shortHorizon")
        end
        if length(customWeights[2]) != longHorizon
            error("customWeights[2] must be of length longHorizon")
        end
        context.extra.short_weights = customWeights[1]
        context.extra.long_weights = customWeights[2]
    end

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

    function customWeighting(prices, weights)
        tmp = sum(prices .* weights)/ sum(weights)
        println(tmp)
        println(weights)
        return tmp
    end

    # Average Calculations: This assumes the data of the subdataframe comes pre-sorted with newest results last.
    shortAVG(sdf_col) = 
        if context.extra.averaging_type == :simple
            return mean(last(sdf_col, Int(context.extra.short_horizon)))
        elseif context.extra.averaging_type == :exponential
            return ema(sdf_col, context.extra.short_horizon)[end]
        elseif context.extra.averaging_type == :custom
            return customWeighting(last(sdf_col, context.extra.short_horizon), context.extra.short_weights)
        end

    longAVG(sdf_col) = 
        if context.extra.averaging_type == :simple
            return mean(last(sdf_col, Int(context.extra.long_horizon)))
        elseif context.extra.averaging_type == :exponential
            return ema(sdf_col, context.extra.long_horizon)[end]
        elseif context.extra.averaging_type == :custom
            return customWeighting(last(sdf_col, context.extra.long_horizon), context.extra.long_weights)
        end
    avg_df = combine(
        groupby(data, ["symbol", "exchangeName"]),
        :close => shortAVG => :SHORT_AVG,
        :close => longAVG => :LONG_AVG,
    )
    avg_df[!, :position] = ((avg_df.SHORT_AVG .>= avg_df.LONG_AVG) .- 0.5) .* 2

    # Order Generation
    for r in eachrow(avg_df)
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
