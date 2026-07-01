using CSV
using DataFrames
using JuMP
using LinearAlgebra

const EPS_M = 1.0e-7

function inflate_big_m(values)
    nonnegative = max.(0.0, Float64.(values))
    return nonnegative .* (1.0 + EPS_M) .+ EPS_M
end

as_string_list(values) = [string(x) for x in skipmissing(values)]
year_columns(df::DataFrame) = [name for name in names(df) if occursin("Year", String(name))]

function pad_columns(A::AbstractMatrix, width::Integer)
    rows, cols = size(A)
    cols >= width && return Matrix{Float64}(A)
    return hcat(Matrix{Float64}(A), zeros(rows, width - cols))
end

function pad_to_length(v::AbstractVector, len::Integer)
    out = Float64.(v)
    length(out) >= len && return out
    return vcat(out, zeros(len - length(out)))
end

function read_scenario_data(data_dir::AbstractString, scenario)
    initial_receipts_df = CSV.read(joinpath(data_dir, "InitialItemReceipts_Scen_$(scenario).csv"), DataFrame)
    demand_df = CSV.read(joinpath(data_dir, "Demand.csv"), DataFrame)
    order_receipts_df = CSV.read(joinpath(data_dir, "OrderItemReceipts_Scen_$(scenario).csv"), DataFrame)
    order_values_df = CSV.read(joinpath(data_dir, "OrderItemValues_Scen_$(scenario).csv"), DataFrame)
    initial_values_df = coalesce.(CSV.read(joinpath(data_dir, "InitialItemValues_Scen_$(scenario).csv"), DataFrame), 0)

    item = as_string_list(initial_receipts_df[!, :Item])
    initial_receipts = Matrix{Float64}(initial_receipts_df[:, year_columns(initial_receipts_df)])
    demand = Float64.(demand_df[!, :Demand])
    order_receipts = Matrix{Float64}(order_receipts_df[:, year_columns(order_receipts_df)])
    initial_values = Matrix{Float64}(initial_values_df[:, year_columns(initial_values_df)])
    order_values = Matrix{Float64}(order_values_df[:, year_columns(order_values_df)])

    max_length = maximum((size(initial_receipts, 2), size(order_receipts, 2),
                          size(initial_values, 2), size(order_values, 2), length(demand)))

    return (
        initial_receipts = pad_columns(initial_receipts, max_length),
        order_receipts = pad_columns(order_receipts, max_length),
        initial_values = pad_columns(initial_values, max_length),
        order_values = pad_columns(order_values, max_length),
        demand = pad_to_length(demand, max_length),
        item = item,
    )
end

function compute_big_m_bound(initial_receipts, order_receipts, order_values, demand;
                             big_m_safety_factor=6.0)
    n_items = size(initial_receipts, 1)
    n_periods = size(order_receipts, 1)
    order_prices = diag(order_values)[1:n_periods]

    M_initial_use = fill(Float64(n_items), n_periods)

    positive_prices = order_prices[order_prices .> 0.0]
    min_price = isempty(positive_prices) ? 1.0 : minimum(positive_prices)
    period_inflow = [sum(max.(initial_receipts[:, t], 0.0)) for t in 1:n_periods]
    ceiling = Float64(big_m_safety_factor) * maximum(period_inflow) / max(min_price, 1.0e-12)
    if !isfinite(ceiling) || ceiling <= 0.0
        ceiling = Float64(n_items)
    end

    M_order_raw = zeros(Float64, n_periods)
    for t in 1:n_periods
        initial_receipt_bound = sum(max.(initial_receipts[:, t], 0.0))
        order_receipt_bound = t == 1 ? 0.0 :
                              sum(max(Float64(order_receipts[i, t]), 0.0) * M_order_raw[i]
                                  for i in 1:(t - 1))
        numerator = initial_receipt_bound + order_receipt_bound - Float64(demand[t])
        denom = max(Float64(order_prices[t]), 1.0e-12)
        M_order_raw[t] = min(max(0.0, numerator / denom), ceiling)
    end

    M_total_use_raw = [
        M_initial_use[t] + min(t == 1 ? 0.0 : sum(@view M_order_raw[1:(t - 1)]), ceiling)
        for t in 1:n_periods
    ]
    M_order_stock_raw = [min(sum(@view M_order_raw[1:t]), ceiling) for t in 1:n_periods]

    return (
        M_order = inflate_big_m(M_order_raw),
        M_total_use = inflate_big_m(M_total_use_raw),
        M_order_stock = inflate_big_m(M_order_stock_raw),
        M_initial_use = fill(Float64(n_items), n_periods),
        M_ceiling = Float64(ceiling),
    )
end

function load_scenario_inputs(data_dir::AbstractString, scenarios)
    inputs = []
    reference_items = nothing

    for scenario in scenarios
        data = read_scenario_data(data_dir, scenario)

        if reference_items === nothing
            reference_items = data.item
        elseif data.item != reference_items
            error("Scenario $(scenario) has a different item order from the first scenario.")
        end

        push!(inputs, (
            scenario = Int(scenario),
            initial_receipts = data.initial_receipts,
            order_receipts = data.order_receipts,
            initial_values = data.initial_values,
            order_values = data.order_values,
            demand = data.demand,
        ))
    end

    return inputs, reference_items
end

function build_extensive_model(data_dir::AbstractString, scenarios;
                               optimizer_factory=nothing,
                               big_m_safety_factor=6.0)
    scenario_inputs, item = load_scenario_inputs(data_dir, scenarios)
    isempty(scenario_inputs) && error("No scenarios were provided.")

    n_items = size(scenario_inputs[1].initial_receipts, 1)
    model = optimizer_factory === nothing ? Model() : Model(optimizer_factory)

    # Shared first-stage inventory choice. Everything else is scenario-specific.
    @variable(model, 0 <= initial_qty[1:n_items] <= 1)
    @variable(model, theta >= 0)

    scenario_vars = Dict{Int, NamedTuple}()
    big_m_by_scenario = Dict{Int, NamedTuple}()

    for data in scenario_inputs
        scenario = data.scenario
        initial_receipts = data.initial_receipts
        order_receipts = data.order_receipts
        initial_values = data.initial_values
        order_values = data.order_values
        demand = data.demand

        size(initial_receipts, 1) == n_items ||
            error("Scenario $(scenario) has a different number of items.")

        n_periods = size(order_receipts, 1)
        order_prices = diag(order_values)[1:n_periods]
        big_m = compute_big_m_bound(
            initial_receipts,
            order_receipts,
            order_values,
            demand;
            big_m_safety_factor = big_m_safety_factor,
        )
        big_m_by_scenario[scenario] = big_m

        order_qty = @variable(model, [t = 1:n_periods], lower_bound = 0,
                              base_name = "order_qty_s$(scenario)")
        use_initial = @variable(model, [i = 1:n_items, t = 1:n_periods],
                                lower_bound = 0, upper_bound = 1,
                                base_name = "use_initial_s$(scenario)")
        use_order = @variable(model, [i = 1:n_periods, t = 1:n_periods],
                              lower_bound = 0,
                              base_name = "use_order_s$(scenario)")
        initial_stock = @variable(model, [i = 1:n_items, t = 1:n_periods],
                                  lower_bound = 0,
                                  base_name = "initial_stock_s$(scenario)")
        order_stock = @variable(model, [i = 1:n_periods, t = i:n_periods],
                                lower_bound = 0,
                                base_name = "order_stock_s$(scenario)")
        order_period = @variable(model, [t = 1:n_periods], Bin,
                                 base_name = "order_period_s$(scenario)")
        initial_use_period = @variable(model, [t = 1:n_periods], Bin,
                                       base_name = "initial_use_period_s$(scenario)")

        @constraint(model, [i = 1:n_items],
                    initial_stock[i, 1] + use_initial[i, 1] == initial_qty[i])
        @constraint(model, [i = 1:n_items, t = 2:n_periods],
                    initial_stock[i, t] + use_initial[i, t] == initial_stock[i, t - 1])

        @constraint(model, [i = 1:n_periods], order_stock[i, i] == order_qty[i])
        for i in 1:n_periods, t in (i + 1):n_periods
            @constraint(model, order_stock[i, t] + use_order[i, t] == order_stock[i, t - 1])
        end

        initial_receipt = [
            @expression(model, sum(initial_receipts[i, t] * initial_stock[i, t]
                                   for i in 1:n_items))
            for t in 1:n_periods
        ]
        order_receipt = [
            @expression(model, sum(order_receipts[i, t] * order_stock[i, t]
                                   for i in 1:(t - 1)))
            for t in 1:n_periods
        ]
        initial_value_used = [
            @expression(model, sum(initial_values[i, t] * use_initial[i, t]
                                   for i in 1:n_items))
            for t in 1:n_periods
        ]
        order_value_used = [
            @expression(model, sum(order_values[i, t] * use_order[i, t]
                                   for i in 1:(t - 1)))
            for t in 1:n_periods
        ]
        order_cost = [
            @expression(model, order_prices[t] * order_qty[t])
            for t in 1:n_periods
        ]

        @constraint(model, [t = 1:n_periods],
            initial_receipt[t] + order_receipt[t] +
            initial_value_used[t] + order_value_used[t] -
            order_cost[t] - demand[t] == 0
        )

        order_stock_end = [
            @expression(model, sum(order_stock[i, t] for i in 1:t))
            for t in 1:n_periods
        ]
        initial_use_total = [
            @expression(model, sum(use_initial[i, t] for i in 1:n_items))
            for t in 1:n_periods
        ]
        total_use = [
            @expression(model, initial_use_total[t] + sum(use_order[i, t] for i in 1:t))
            for t in 1:n_periods
        ]

        @constraint(model, [t = 1:n_periods], order_qty[t] <= big_m.M_order[t] * order_period[t])
        @constraint(model, [t = 1:n_periods], total_use[t] <= big_m.M_total_use[t] * (1 - order_period[t]))
        @constraint(model, [t = 1:n_periods], order_stock_end[t] <= big_m.M_order_stock[t] * (1 - initial_use_period[t]))
        @constraint(model, [t = 1:n_periods], initial_use_total[t] <= big_m.M_initial_use[t] * initial_use_period[t])

        for t in 1:n_periods, i in t:n_periods
            fix(use_order[i, t], 0.0; force = true)
        end

        @constraint(model, theta >= sum(initial_values[i, 1] * initial_qty[i]
                                        for i in 1:n_items))

        scenario_vars[scenario] = (
            order_qty = order_qty,
            use_initial = use_initial,
            use_order = use_order,
            initial_stock = initial_stock,
            order_stock = order_stock,
            order_period = order_period,
            initial_use_period = initial_use_period,
        )
    end

    @objective(model, Min, theta)

    return (
        model = model,
        initial_qty = initial_qty,
        theta = theta,
        scenario_vars = scenario_vars,
        big_m_by_scenario = big_m_by_scenario,
        item = item,
        scenarios = collect(Int.(scenarios)),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    data_dir = joinpath(@__DIR__, "inventory_example_data")
    # Small default instance for reproducibility.
    scenarios = [0, 1]
    # Larger/full instance for decomposition experiments:
    # scenarios = collect(0:8)

    built = build_extensive_model(data_dir, scenarios)

    println("Inventory extensive model built.")
    println("Scenarios: ", built.scenarios)
    println("Variables: ", num_variables(built.model))
    println("Constraints: ", num_constraints(built.model; count_variable_in_set_constraints = false))
end
