using CSV
using DataFrames
using JuMP
using LinearAlgebra
using Printf
using Xpress
const MOI = JuMP.MOI

# -----------------------------------------------------------------------------
# Benders via elastic LP-relaxation feasibility cuts.
#
# This file implements an outer-loop Benders method for the LP relaxation of the
# scenario recourse problems. The master contains the shared first-stage
# variables `initial_qty`, `theta`, and one nonnegative violation estimator
# `eta[s]` per scenario.
#
# The cut-generating subproblems keep the scenario-specific flow and timing
# variables in the subproblem, but relax the timing binaries to the interval
# [0, 1]. Rather than relying directly on an infeasibility certificate from an
# infeasible LP, each relaxed subproblem is made elastic with nonnegative
# artificial slacks and minimizes total scaled violation. The duals of the
# constraints fixing the local copy of `initial_qty` give subgradients of this
# LP-relaxation violation function.
#
# The final binary-recourse check is diagnostic: LP-relaxation cuts can prove
# infeasibility of some first-stage points, but they do not by themselves enforce
# feasibility of the original binary-recourse problem.
# -----------------------------------------------------------------------------

const EPS_M = 1.0e-7

# Tolerance on the scaled elastic violation. Each elasticized row is divided by
# a natural row scale, so the slacks are approximately relative violations. The
# cut-generating LPs are solved to tighter tolerances than the acceptance
# tolerance to reduce numerical noise in the dual-based cut coefficients.
const DEFAULT_FEASIBILITY_TOL = 1.0e-4
const ELASTIC_LP_TOL = 1.0e-9

# Very small apparent violations can be dominated by numerical noise in the
# LP duals. Do not add new cuts below this floor.
const CUT_NOISE_FLOOR = 2.0e-6
# Smallest eta budget the restoration pass will tighten to before giving up.
const ETA_BUDGET_FLOOR = 1.0e-9

const SCENARIO_FILE_TEMPLATES = (
    "InitialItemReceipts_Scen_%s.csv",
    "InitialItemValues_Scen_%s.csv",
    "OrderItemReceipts_Scen_%s.csv",
    "OrderItemValues_Scen_%s.csv",
)

# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------

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

function scenario_file_names(scenario)
    return [replace(template, "%s" => string(scenario)) for template in SCENARIO_FILE_TEMPLATES]
end

function validate_scenario_files(data_dir::AbstractString, scenarios)
    isdir(data_dir) || error("Data directory does not exist: $(data_dir)")
    missing_files = String[]

    demand_file = joinpath(data_dir, "Demand.csv")
    isfile(demand_file) || push!(missing_files, demand_file)

    for scenario in scenarios
        for file_name in scenario_file_names(scenario)
            path = joinpath(data_dir, file_name)
            isfile(path) || push!(missing_files, path)
        end
    end

    if !isempty(missing_files)
        error("Missing required data files:\n  " * join(missing_files, "\n  "))
    end
    return
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

    max_length = maximum((
        size(initial_receipts, 2),
        size(order_receipts, 2),
        size(initial_values, 2),
        size(order_values, 2),
        length(demand),
    ))

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
                             big_m_safety_factor::Float64 = 6.0)
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
            sum(max(Float64(order_receipts[i, t]), 0.0) * M_order_raw[i] for i in 1:(t - 1))
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
    validate_scenario_files(data_dir, scenarios)

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

# -----------------------------------------------------------------------------
# Model containers
# -----------------------------------------------------------------------------

mutable struct MasterProblem
    model::Model
    initial_qty::Vector{VariableRef}
    theta::VariableRef
    eta::Dict{Int,VariableRef}
    eta_budget::Union{Nothing,ConstraintRef}
end

struct ScenarioSubproblem
    scenario::Int
    model::Model
    x_copy::Vector{VariableRef}
    x_link::Vector{ConstraintRef}
    elastic_feasibility::Bool
    relax_binaries::Bool
end

# -----------------------------------------------------------------------------
# Xpress helpers
# -----------------------------------------------------------------------------

function maybe_set_attribute!(model::Model, name::AbstractString, value)
    try
        set_attribute(model, name, value)
    catch err
        @warn "Could not set solver attribute $(name)=$(value). Continuing." exception=(err, catch_backtrace())
    end
    return
end

function configure_xpress!(model::Model;
                           silent::Bool = true,
                           presolve = nothing,
                           time_limit = nothing,
                           mip_rel_gap = nothing,
                           threads = nothing)
    silent && set_silent(model)
    presolve === nothing || maybe_set_attribute!(model, "PRESOLVE", presolve ? 1 : 0)
    time_limit === nothing || maybe_set_attribute!(model, "MAXTIME", Float64(time_limit))
    mip_rel_gap === nothing || maybe_set_attribute!(model, "MIPRELSTOP", Float64(mip_rel_gap))
    threads === nothing || maybe_set_attribute!(model, "THREADS", Int(threads))
    return
end

# -----------------------------------------------------------------------------
# Master problem
# -----------------------------------------------------------------------------

function scenario_ids(scenario_inputs)
    return sort([Int(data.scenario) for data in scenario_inputs])
end

function build_master(scenario_inputs;
                      optimizer_factory = Xpress.Optimizer,
                      silent::Bool = true,
                      xpress_presolve = true,
                      time_limit = nothing,
                      threads = nothing,
                      phase1_theta_weight::Float64 = 0.0)
    isempty(scenario_inputs) && error("No scenarios were provided.")
    n_items = size(scenario_inputs[1].initial_receipts, 1)
    scenarios = scenario_ids(scenario_inputs)

    model = Model(optimizer_factory)
    configure_xpress!(model; silent = silent, presolve = xpress_presolve, time_limit = time_limit, threads = threads)

    @variable(model, 0 <= initial_qty[1:n_items] <= 1)
    @variable(model, theta >= 0)
    @variable(model, eta_var[s in scenarios] >= 0)

    # Scenario epigraph constraints. These involve only x and theta.
    for data in scenario_inputs
        @constraint(model, theta >= sum(data.initial_values[i, 1] * initial_qty[i] for i in 1:n_items))
    end

    eta = Dict(Int(s) => eta_var[s] for s in scenarios)
    master = MasterProblem(model, collect(initial_qty), theta, eta, nothing)
    set_master_objective!(master, :phase1; theta_weight = phase1_theta_weight)
    return master
end

function set_master_objective!(master::MasterProblem, phase::Symbol; theta_weight::Float64 = 0.0)
    scenarios = sort(collect(keys(master.eta)))
    if phase == :phase1
        @objective(master.model, Min, sum(master.eta[s] for s in scenarios) + theta_weight * master.theta)
    elseif phase == :phase2
        @objective(master.model, Min, master.theta)
    else
        error("Unknown master objective phase: $(phase). Use :phase1 or :phase2.")
    end
    return
end

function add_eta_budget!(master::MasterProblem; feasibility_tol::Float64 = DEFAULT_FEASIBILITY_TOL)
    scenarios = sort(collect(keys(master.eta)))
    if master.eta_budget === nothing
        master.eta_budget = @constraint(master.model, sum(master.eta[s] for s in scenarios) <= feasibility_tol)
    else
        set_normalized_rhs(master.eta_budget, feasibility_tol)
    end
    return master.eta_budget
end

function solve_master!(master::MasterProblem; label::AbstractString = "master")
    optimize!(master.model)
    ts = termination_status(master.model)
    ps = primal_status(master.model)
    if ts != MOI.OPTIMAL || ps != MOI.FEASIBLE_POINT
        error("$(label) was not solved to optimality. termination_status=$(ts), primal_status=$(ps).")
    end
    return
end

# -----------------------------------------------------------------------------
# Scenario subproblem.
#
# elastic_feasibility=true, relax_binaries=true:
#   Cut-generating LP. Timing binaries are relaxed to [0, 1], artificial slacks
#   are added to selected rows, and the objective minimizes total scaled
#   violation. A zero optimum means the LP relaxation is feasible for the fixed
#   first-stage decision. The duals of `x_link` give a subgradient cut.
#
# elastic_feasibility=false, relax_binaries=true:
#   Plain LP relaxation. Used for final consistency checking.
#
# elastic_feasibility=true, relax_binaries=false:
#   Elastic MIP verification model. Timing variables are binary and slacks are
#   retained to measure the minimum scaled violation of the original recourse
#   logic at a candidate first-stage decision.
# -----------------------------------------------------------------------------

function build_scenario_subproblem(data;
                                   optimizer_factory = Xpress.Optimizer,
                                   relax_binaries::Bool = true,
                                   elastic_feasibility::Bool = true,
                                   silent::Bool = true,
                                   big_m_safety_factor::Float64 = 6.0,
                                   xpress_presolve = true,
                                   time_limit = nothing,
                                   mip_rel_gap = nothing,
                                   threads = nothing)
    scenario = Int(data.scenario)
    initial_receipts = data.initial_receipts
    order_receipts = data.order_receipts
    initial_values = data.initial_values
    order_values = data.order_values
    demand = data.demand

    n_items = size(initial_receipts, 1)
    n_periods = size(order_receipts, 1)
    order_prices = diag(order_values)[1:n_periods]
    big_m = compute_big_m_bound(
        initial_receipts,
        order_receipts,
        order_values,
        demand;
        big_m_safety_factor = big_m_safety_factor,
    )

    model = Model(optimizer_factory)
    configure_xpress!(
        model;
        silent = silent,
        presolve = xpress_presolve,
        time_limit = time_limit,
        mip_rel_gap = mip_rel_gap,
        threads = threads,
    )
    if elastic_feasibility && relax_binaries
        # The x_link duals become Benders cut gradients. Use tighter LP
        # tolerances for the cut-generating subproblems so that numerical noise
        # in the dual coefficients stays below the feasibility tolerance.
        maybe_set_attribute!(model, "FEASTOL", ELASTIC_LP_TOL)
        maybe_set_attribute!(model, "OPTIMALITYTOL", ELASTIC_LP_TOL)
    end

    @variable(model, x_copy[1:n_items])
    x_link = @constraint(model, [i = 1:n_items], x_copy[i] == 0.0)

    elastic_penalty = AffExpr(0.0)

    # The rows of this model have different magnitudes: initial-item flow is
    # O(1), order-flow rows scale with the order bounds, and balance rows scale
    # with demand. Each elasticized row is divided by a natural row scale so that
    # slack variables measure comparable relative violations and the x_link
    # duals are better conditioned.
    inv_balance_scale = [1.0 / (1.0 + abs(Float64(demand[t]))) for t in 1:n_periods]
    inv_order_scale = [1.0 / (1.0 + big_m.M_order[i]) for i in 1:n_periods]
    inv_total_use_scale = [1.0 / (1.0 + big_m.M_total_use[t]) for t in 1:n_periods]
    inv_order_stock_scale = [1.0 / (1.0 + big_m.M_order_stock[t]) for t in 1:n_periods]
    inv_initial_use_scale = [1.0 / (1.0 + big_m.M_initial_use[t]) for t in 1:n_periods]

    order_qty = @variable(model, [t = 1:n_periods], lower_bound = 0,
                          base_name = "order_qty_s$(scenario)")
    use_initial = @variable(model, [i = 1:n_items, t = 1:n_periods],
                            lower_bound = 0,
                            base_name = "use_initial_s$(scenario)")
    @constraint(model, [i = 1:n_items, t = 1:n_periods], use_initial[i, t] <= 1.0)
    use_order = @variable(model, [i = 1:n_periods, t = 1:n_periods],
                          lower_bound = 0,
                          base_name = "use_order_s$(scenario)")
    initial_stock = @variable(model, [i = 1:n_items, t = 1:n_periods],
                              lower_bound = 0,
                              base_name = "initial_stock_s$(scenario)")
    order_stock = @variable(model, [i = 1:n_periods, t = i:n_periods],
                            lower_bound = 0,
                            base_name = "order_stock_s$(scenario)")

    if relax_binaries
        order_period = @variable(model, [t = 1:n_periods], lower_bound = 0,
                                 base_name = "order_period_relaxed_s$(scenario)")
        initial_use_period = @variable(model, [t = 1:n_periods], lower_bound = 0,
                                       base_name = "initial_use_period_relaxed_s$(scenario)")
        @constraint(model, [t = 1:n_periods], order_period[t] <= 1.0)
        @constraint(model, [t = 1:n_periods], initial_use_period[t] <= 1.0)
    else
        order_period = @variable(model, [t = 1:n_periods], Bin,
                                 base_name = "order_period_s$(scenario)")
        initial_use_period = @variable(model, [t = 1:n_periods], Bin,
                                       base_name = "initial_use_period_s$(scenario)")
    end

    if elastic_feasibility
        initial_flow_pos = @variable(model, [i = 1:n_items, t = 1:n_periods], lower_bound = 0,
                                     base_name = "initial_flow_pos_s$(scenario)")
        initial_flow_neg = @variable(model, [i = 1:n_items, t = 1:n_periods], lower_bound = 0,
                                     base_name = "initial_flow_neg_s$(scenario)")
        elastic_penalty += sum(initial_flow_pos) + sum(initial_flow_neg)

        @constraint(model, [i = 1:n_items],
            initial_stock[i, 1] + use_initial[i, 1] +
            initial_flow_pos[i, 1] - initial_flow_neg[i, 1] == x_copy[i]
        )
        @constraint(model, [i = 1:n_items, t = 2:n_periods],
            initial_stock[i, t] + use_initial[i, t] +
            initial_flow_pos[i, t] - initial_flow_neg[i, t] == initial_stock[i, t - 1]
        )
    else
        @constraint(model, [i = 1:n_items],
            initial_stock[i, 1] + use_initial[i, 1] == x_copy[i]
        )
        @constraint(model, [i = 1:n_items, t = 2:n_periods],
            initial_stock[i, t] + use_initial[i, t] == initial_stock[i, t - 1]
        )
    end

    if elastic_feasibility
        order_flow_pos = @variable(model, [i = 1:n_periods, t = 1:n_periods], lower_bound = 0,
                                   base_name = "order_flow_pos_s$(scenario)")
        order_flow_neg = @variable(model, [i = 1:n_periods, t = 1:n_periods], lower_bound = 0,
                                   base_name = "order_flow_neg_s$(scenario)")
        for i in 1:n_periods, t in i:n_periods
            elastic_penalty += order_flow_pos[i, t] + order_flow_neg[i, t]
        end

        @constraint(model, [i = 1:n_periods],
            (order_stock[i, i] - order_qty[i]) * inv_order_scale[i] +
            order_flow_pos[i, i] - order_flow_neg[i, i] == 0
        )
        for i in 1:n_periods, t in (i + 1):n_periods
            @constraint(model,
                (order_stock[i, t] + use_order[i, t] - order_stock[i, t - 1]) * inv_order_scale[i] +
                order_flow_pos[i, t] - order_flow_neg[i, t] == 0
            )
        end
    else
        @constraint(model, [i = 1:n_periods], order_stock[i, i] == order_qty[i])
        for i in 1:n_periods, t in (i + 1):n_periods
            @constraint(model, order_stock[i, t] + use_order[i, t] == order_stock[i, t - 1])
        end
    end

    initial_receipt = [
        @expression(model, sum(initial_receipts[i, t] * initial_stock[i, t] for i in 1:n_items))
        for t in 1:n_periods
    ]
    order_receipt = [
        @expression(model, sum(order_receipts[i, t] * order_stock[i, t] for i in 1:(t - 1)))
        for t in 1:n_periods
    ]
    initial_value_used = [
        @expression(model, sum(initial_values[i, t] * use_initial[i, t] for i in 1:n_items))
        for t in 1:n_periods
    ]
    order_value_used = [
        @expression(model, sum(order_values[i, t] * use_order[i, t] for i in 1:(t - 1)))
        for t in 1:n_periods
    ]
    order_cost = [
        @expression(model, order_prices[t] * order_qty[t])
        for t in 1:n_periods
    ]

    if elastic_feasibility
        balance_pos = @variable(model, [t = 1:n_periods], lower_bound = 0,
                                base_name = "balance_pos_s$(scenario)")
        balance_neg = @variable(model, [t = 1:n_periods], lower_bound = 0,
                                base_name = "balance_neg_s$(scenario)")
        elastic_penalty += sum(balance_pos) + sum(balance_neg)

        @constraint(model, [t = 1:n_periods],
            (initial_receipt[t] + order_receipt[t] +
             initial_value_used[t] + order_value_used[t] -
             order_cost[t] - demand[t]) * inv_balance_scale[t] +
            balance_pos[t] - balance_neg[t] == 0
        )
    else
        @constraint(model, [t = 1:n_periods],
            initial_receipt[t] + order_receipt[t] +
            initial_value_used[t] + order_value_used[t] -
            order_cost[t] - demand[t] == 0
        )
    end

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

    if elastic_feasibility
        order_qty_slack = @variable(model, [t = 1:n_periods], lower_bound = 0,
                                    base_name = "order_qty_slack_s$(scenario)")
        total_use_slack = @variable(model, [t = 1:n_periods], lower_bound = 0,
                                    base_name = "total_use_slack_s$(scenario)")
        order_stock_slack = @variable(model, [t = 1:n_periods], lower_bound = 0,
                                      base_name = "order_stock_slack_s$(scenario)")
        initial_use_slack = @variable(model, [t = 1:n_periods], lower_bound = 0,
                                      base_name = "initial_use_slack_s$(scenario)")
        elastic_penalty += sum(order_qty_slack) + sum(total_use_slack) +
                           sum(order_stock_slack) + sum(initial_use_slack)

        @constraint(model, [t = 1:n_periods],
            (order_qty[t] - big_m.M_order[t] * order_period[t]) * inv_order_scale[t] <=
            order_qty_slack[t]
        )
        @constraint(model, [t = 1:n_periods],
            (total_use[t] - big_m.M_total_use[t] * (1 - order_period[t])) * inv_total_use_scale[t] <=
            total_use_slack[t]
        )
        @constraint(model, [t = 1:n_periods],
            (order_stock_end[t] - big_m.M_order_stock[t] * (1 - initial_use_period[t])) * inv_order_stock_scale[t] <=
            order_stock_slack[t]
        )
        @constraint(model, [t = 1:n_periods],
            (initial_use_total[t] - big_m.M_initial_use[t] * initial_use_period[t]) * inv_initial_use_scale[t] <=
            initial_use_slack[t]
        )
    else
        @constraint(model, [t = 1:n_periods], order_qty[t] <= big_m.M_order[t] * order_period[t])
        @constraint(model, [t = 1:n_periods], total_use[t] <= big_m.M_total_use[t] * (1 - order_period[t]))
        @constraint(model, [t = 1:n_periods], order_stock_end[t] <= big_m.M_order_stock[t] * (1 - initial_use_period[t]))
        @constraint(model, [t = 1:n_periods], initial_use_total[t] <= big_m.M_initial_use[t] * initial_use_period[t])
    end

    # Same-period ordered usage is not allowed in the extensive-form model.
    for t in 1:n_periods, i in t:n_periods
        fix(use_order[i, t], 0.0; force = true)
    end

    if elastic_feasibility
        @objective(model, Min, elastic_penalty)
    else
        @objective(model, Min, 0.0)
    end

    return ScenarioSubproblem(
        scenario,
        model,
        collect(x_copy),
        collect(x_link),
        elastic_feasibility,
        relax_binaries,
    )
end

function fix_x_copy!(subproblem::ScenarioSubproblem, x_value::AbstractVector{<:Real})
    length(x_value) == length(subproblem.x_copy) || error(
        "x has length $(length(x_value)), but scenario $(subproblem.scenario) expects $(length(subproblem.x_copy))."
    )
    for i in eachindex(subproblem.x_copy)
        set_normalized_rhs(subproblem.x_link[i], Float64(x_value[i]))
    end
    return
end

# -----------------------------------------------------------------------------
# Subproblem solves and cut generation
# -----------------------------------------------------------------------------

function solve_elastic_relaxation!(subproblem::ScenarioSubproblem,
                                   x_value::AbstractVector{<:Real};
                                   feasibility_tol::Float64 = DEFAULT_FEASIBILITY_TOL,
                                   dual_sign::Float64 = 1.0)
    subproblem.elastic_feasibility || error("Expected an elastic subproblem.")
    subproblem.relax_binaries || error("Elastic cuts should be generated from the LP relaxation.")

    fix_x_copy!(subproblem, x_value)
    optimize!(subproblem.model)

    ts = termination_status(subproblem.model)
    ps = primal_status(subproblem.model)
    if ts != MOI.OPTIMAL || ps != MOI.FEASIBLE_POINT
        error(
            "Elastic LP subproblem for scenario $(subproblem.scenario) was not solved to optimality. " *
            "termination_status=$(ts), primal_status=$(ps)."
        )
    end

    violation = Float64(objective_value(subproblem.model))
    duals = Float64.(dual.(subproblem.x_link))

    # For a minimization problem with a constraint written as `x_copy == rhs`,
    # the JuMP/MOI dual convention gives the sensitivity of the elastic value
    # with respect to `rhs`. The `dual_sign` keyword is kept configurable for
    # checking sign conventions across solver/bridge combinations.
    g = dual_sign .* duals

    return (
        scenario = subproblem.scenario,
        is_feasible = violation <= feasibility_tol,
        violation = violation,
        g = collect(g),
        termination_status = ts,
        primal_status = ps,
        dual_status = dual_status(subproblem.model),
    )
end

function solve_plain_lp_relaxation!(subproblem::ScenarioSubproblem,
                                    x_value::AbstractVector{<:Real})
    !subproblem.elastic_feasibility || error("Expected a non-elastic subproblem.")
    subproblem.relax_binaries || error("Expected an LP-relaxation subproblem.")

    fix_x_copy!(subproblem, x_value)
    optimize!(subproblem.model)

    return (
        scenario = subproblem.scenario,
        is_feasible = primal_status(subproblem.model) == MOI.FEASIBLE_POINT,
        termination_status = termination_status(subproblem.model),
        primal_status = primal_status(subproblem.model),
        dual_status = dual_status(subproblem.model),
    )
end

# Integer recourse is verified with an elastic MIP: timing variables are binary,
# but the scaled slacks are retained. The candidate is accepted when the minimum
# scaled violation is within the same tolerance used for the LP relaxation. This
# gives a more informative diagnostic than a hard feasibility-only MIP when the
# candidate lies close to the recourse-feasibility boundary.
function solve_integer_recourse!(subproblem::ScenarioSubproblem,
                                 x_value::AbstractVector{<:Real};
                                 feasibility_tol::Float64 = DEFAULT_FEASIBILITY_TOL)
    subproblem.elastic_feasibility || error("Expected an elastic subproblem.")
    !subproblem.relax_binaries || error("Expected an integer-recourse subproblem.")

    fix_x_copy!(subproblem, x_value)
    optimize!(subproblem.model)

    ps = primal_status(subproblem.model)
    violation = ps == MOI.FEASIBLE_POINT ? Float64(objective_value(subproblem.model)) : Inf

    return (
        scenario = subproblem.scenario,
        is_feasible = violation <= feasibility_tol,
        violation = violation,
        termination_status = termination_status(subproblem.model),
        primal_status = ps,
    )
end

function add_elastic_value_cut!(master::MasterProblem,
                                scenario::Int,
                                violation::Float64,
                                g::Vector{Float64},
                                x_value::Vector{Float64})
    haskey(master.eta, scenario) || error("Master has no eta variable for scenario $(scenario).")
    length(g) == length(master.initial_qty) || error("Cut gradient length does not match x length.")

    cut = @constraint(master.model,
        violation + sum(g[i] * (master.initial_qty[i] - x_value[i]) for i in eachindex(x_value)) <= master.eta[scenario]
    )
    return cut
end

# -----------------------------------------------------------------------------
# Main two-phase Benders algorithm
# -----------------------------------------------------------------------------

function run_benders_elastic_lp_relaxation(data_dir::AbstractString;
                                           scenarios = [0, 1],
                                           phase1_iterations::Int = 100,
                                           phase2_iterations::Int = 300,
                                           optimizer_factory = Xpress.Optimizer,
                                           silent::Bool = true,
                                           big_m_safety_factor::Float64 = 6.0,
                                           feasibility_tol::Float64 = DEFAULT_FEASIBILITY_TOL,
                                           phase1_theta_weight::Float64 = 0.0,
                                           verify_plain_lp::Bool = true,
                                           verify_integer_recourse::Bool = true,
                                           dual_sign::Float64 = 1.0,
                                           xpress_presolve = true,
                                           xpress_time_limit = nothing,
                                           xpress_mip_rel_gap = nothing,
                                           xpress_threads = nothing,
                                           print_log::Bool = true)
    phase1_iterations >= 1 || error("phase1_iterations must be at least 1.")
    phase2_iterations >= 1 || error("phase2_iterations must be at least 1.")

    scenario_inputs, item = load_scenario_inputs(data_dir, scenarios)
    n_items = size(scenario_inputs[1].initial_receipts, 1)
    scenario_list = scenario_ids(scenario_inputs)

    master = build_master(
        scenario_inputs;
        optimizer_factory = optimizer_factory,
        silent = silent,
        xpress_presolve = xpress_presolve,
        time_limit = xpress_time_limit,
        threads = xpress_threads,
        phase1_theta_weight = phase1_theta_weight,
    )

    elastic_subproblems = Dict(
        data.scenario => build_scenario_subproblem(
            data;
            optimizer_factory = optimizer_factory,
            relax_binaries = true,
            elastic_feasibility = true,
            silent = silent,
            big_m_safety_factor = big_m_safety_factor,
            xpress_presolve = xpress_presolve,
            time_limit = xpress_time_limit,
            threads = xpress_threads,
        )
        for data in scenario_inputs
    )

    # The plain-LP and integer-recourse models are only needed at the final
    # verification step, so they are built lazily. This keeps the initial model
    # build smaller and avoids unnecessary memory use when many scenarios are
    # present.
    data_by_scenario = Dict(Int(data.scenario) => data for data in scenario_inputs)
    plain_lp_subproblems = Dict{Int,ScenarioSubproblem}()
    mip_subproblems = Dict{Int,ScenarioSubproblem}()

    function get_plain_lp_subproblem(s::Int)
        return get!(plain_lp_subproblems, s) do
            build_scenario_subproblem(
                data_by_scenario[s];
                optimizer_factory = optimizer_factory,
                relax_binaries = true,
                elastic_feasibility = false,
                silent = silent,
                big_m_safety_factor = big_m_safety_factor,
                xpress_presolve = xpress_presolve,
                time_limit = xpress_time_limit,
                threads = xpress_threads,
            )
        end
    end

    function get_mip_subproblem(s::Int)
        return get!(mip_subproblems, s) do
            build_scenario_subproblem(
                data_by_scenario[s];
                optimizer_factory = optimizer_factory,
                relax_binaries = false,
                elastic_feasibility = true,
                silent = silent,
                big_m_safety_factor = big_m_safety_factor,
                xpress_presolve = xpress_presolve,
                time_limit = xpress_time_limit === nothing ? 600.0 : xpress_time_limit,
                mip_rel_gap = xpress_mip_rel_gap,
                threads = xpress_threads,
            )
        end
    end

    history = NamedTuple[]
    total_cuts = 0
    last_x = zeros(Float64, n_items)
    last_theta = NaN
    last_master_objective = NaN

    if print_log
        println("Two-phase Benders with elastic LP-relaxation cuts")
        println("Scenarios: ", scenario_list)
        println("Items: ", n_items)
        println("Xpress presolve for subproblems: ", xpress_presolve)
        println()
        println("Phase | Iter | master obj | theta | sum eta | max viol | cuts | total cuts")
        println("------+------|------------+-------+---------+----------+------+-----------")
    end

    function iteration_step!(phase::Symbol, iteration::Int;
                             cut_add_tol::Float64 = feasibility_tol)
        solve_master!(master; label = "$(phase) master")
        x_k = Float64.(value.(master.initial_qty))
        theta_k = Float64(value(master.theta))
        eta_sum = sum(Float64(value(master.eta[s])) for s in scenario_list)
        master_objective = Float64(objective_value(master.model))

        max_violation = 0.0
        sum_violation = 0.0
        cuts_this_iteration = 0
        scenario_violations = Dict{Int,Float64}()

        for data in scenario_inputs
            ret = solve_elastic_relaxation!(
                elastic_subproblems[data.scenario],
                x_k;
                feasibility_tol = feasibility_tol,
                dual_sign = dual_sign,
            )
            scenario_violations[data.scenario] = ret.violation
            max_violation = max(max_violation, ret.violation)
            sum_violation += ret.violation

            # Always add the cut if the true violation is above tolerance. This
            # is useful even if the current eta already approximately covers the
            # old cuts, because this is a new supporting hyperplane at x_k.
            if ret.violation > cut_add_tol
                add_elastic_value_cut!(
                    master,
                    ret.scenario,
                    ret.violation,
                    Float64.(ret.g),
                    x_k,
                )
                cuts_this_iteration += 1
            end
        end

        return (
            phase = phase,
            iteration = iteration,
            x = x_k,
            theta = theta_k,
            eta_sum = eta_sum,
            master_objective = master_objective,
            max_violation = max_violation,
            sum_violation = sum_violation,
            cuts_added = cuts_this_iteration,
            scenario_violations = scenario_violations,
        )
    end

    # Phase I: find an x for which the LP-relaxation recourse is feasible.
    set_master_objective!(master, :phase1; theta_weight = phase1_theta_weight)
    phase1_feasible = false
    phase1_result = nothing
    phase1_status = :phase1_iteration_limit
    stall_count = 0
    prev_x = nothing
    prev_max_violation = Inf

    for iteration in 1:phase1_iterations
        step = iteration_step!(:phase1, iteration)
        total_cuts += step.cuts_added
        last_x = copy(step.x)
        last_theta = step.theta
        last_master_objective = step.master_objective

        push!(history, (
            phase = step.phase,
            iteration = step.iteration,
            master_objective = step.master_objective,
            theta = step.theta,
            eta_sum = step.eta_sum,
            max_violation = step.max_violation,
            sum_violation = step.sum_violation,
            cuts_added = step.cuts_added,
            total_cuts = total_cuts,
        ))

        if print_log
            @printf("%5s | %4d | %10.4e | %5.4e | %7.2e | %8.2e | %4d | %9d\n",
                String(step.phase), iteration, step.master_objective, step.theta,
                step.eta_sum, step.max_violation, step.cuts_added, total_cuts)
        end

        if step.max_violation <= feasibility_tol
            phase1_feasible = true
            phase1_result = step
            break
        end

        # Stall guard: if the master returns the same x and the violation does
        # not improve, the new cuts are not informative. This can indicate a
        # wrong cut-gradient sign or tolerances that are too tight.
        x_unchanged = prev_x !== nothing && maximum(abs.(step.x .- prev_x)) <= 1.0e-12
        if x_unchanged && step.max_violation >= prev_max_violation - 1.0e-12
            stall_count += 1
        else
            stall_count = 0
        end
        prev_x = copy(step.x)
        prev_max_violation = step.max_violation

        if stall_count >= 3
            phase1_status = :phase1_stalled
            print_log && println(
                "Phase I stalled: x and the violation stopped changing while cuts keep ",
                "being added. Check the cut gradient sign (dual_sign) and tolerances."
            )
            break
        end
    end

    if !phase1_feasible && phase1_status == :phase1_iteration_limit
        print_log && println("Phase I iteration limit reached before LP-relaxation feasibility was found.")
    end

    if !phase1_feasible
        return (
            status = phase1_status,
            x = last_x,
            theta = last_theta,
            master_objective = last_master_objective,
            total_cuts = total_cuts,
            history = history,
            item = item,
            master = master,
            elastic_subproblems = elastic_subproblems,
            plain_lp_subproblems = plain_lp_subproblems,
            mip_subproblems = mip_subproblems,
        )
    end

    # Phase II: minimize theta while enforcing eta close to zero. New cuts are
    # still added whenever the candidate x violates the elastic LP relaxation.
    #
    # Use an eta budget below the termination tolerance. If the budget is too
    # loose, the master may prefer a slightly infeasible point because the small
    # allowed violation can reduce theta.
    eta_budget_value = 0.5 * feasibility_tol
    add_eta_budget!(master; feasibility_tol = eta_budget_value)
    set_master_objective!(master, :phase2)
    stall_count = 0
    prev_x = nothing
    prev_max_violation = Inf

    for iteration in 1:phase2_iterations
        local step
        try
            step = iteration_step!(:phase2, iteration;
                cut_add_tol = max(CUT_NOISE_FLOOR, 0.5 * eta_budget_value))
        catch err
            # Restoration can tighten the budget below the cut noise floor and
            # make the master infeasible; back off once instead of dying.
            if eta_budget_value < 0.5 * feasibility_tol
                eta_budget_value = min(10.0 * eta_budget_value, 0.5 * feasibility_tol)
                set_normalized_rhs(master.eta_budget, eta_budget_value)
                print_log && @printf(
                    "Master solve failed at tightened eta budget; relaxing budget to %.1e.\n",
                    eta_budget_value)
                continue
            end
            rethrow()
        end
        total_cuts += step.cuts_added
        last_x = copy(step.x)
        last_theta = step.theta
        last_master_objective = step.master_objective

        push!(history, (
            phase = step.phase,
            iteration = step.iteration,
            master_objective = step.master_objective,
            theta = step.theta,
            eta_sum = step.eta_sum,
            max_violation = step.max_violation,
            sum_violation = step.sum_violation,
            cuts_added = step.cuts_added,
            total_cuts = total_cuts,
        ))

        if print_log
            @printf("%5s | %4d | %10.4e | %5.4e | %7.2e | %8.2e | %4d | %9d\n",
                String(step.phase), iteration, step.master_objective, step.theta,
                step.eta_sum, step.max_violation, step.cuts_added, total_cuts)
        end

        if step.max_violation > feasibility_tol
            # Stall guard, same idea as in Phase I: identical x and
            # non-improving violation means new cuts carry no information.
            x_unchanged = prev_x !== nothing && maximum(abs.(step.x .- prev_x)) <= 1.0e-12
            if x_unchanged && step.max_violation >= prev_max_violation - 1.0e-12
                stall_count += 1
            else
                stall_count = 0
            end
            prev_x = copy(step.x)
            prev_max_violation = step.max_violation

            if stall_count >= 3
                print_log && println(
                    "Phase II stalled: x and the violation stopped changing. The eta ",
                    "budget or feasibility_tol is likely too tight for the data scale."
                )
                return (
                    status = :phase2_stalled,
                    x = last_x,
                    theta = last_theta,
                    master_objective = last_master_objective,
                    total_cuts = total_cuts,
                    history = history,
                    item = item,
                    master = master,
                    elastic_subproblems = elastic_subproblems,
                    plain_lp_subproblems = plain_lp_subproblems,
                    mip_subproblems = mip_subproblems,
                )
            end
            continue
        end
        stall_count = 0
        prev_x = nothing
        prev_max_violation = Inf

        if verify_plain_lp
            plain_lp_results = [
                solve_plain_lp_relaxation!(get_plain_lp_subproblem(data.scenario), step.x)
                for data in scenario_inputs
            ]
            plain_lp_feasible = all(r.is_feasible for r in plain_lp_results)
            if !plain_lp_feasible
                bad = [r.scenario for r in plain_lp_results if !r.is_feasible]

                # Restoration: Phase II can return a point very close to the
                # LP-feasibility boundary. Tighten the eta budget and continue
                # cutting before declaring that the plain LP check has failed.
                if eta_budget_value > ETA_BUDGET_FLOOR
                    eta_budget_value = max(0.1 * eta_budget_value, ETA_BUDGET_FLOOR)
                    set_normalized_rhs(master.eta_budget, eta_budget_value)
                    print_log && @printf(
                        "Plain LP failed for scenarios %s; tightening eta budget to %.1e and continuing.\n",
                        string(bad), eta_budget_value)
                    continue
                end

                print_log && println("Elastic objective is small, but plain LP relaxation failed for scenarios ", bad)
                return (
                    status = :elastic_zero_but_plain_lp_failed,
                    x = step.x,
                    theta = step.theta,
                    master_objective = step.master_objective,
                    total_cuts = total_cuts,
                    failed_lp_scenarios = bad,
                    plain_lp_results = plain_lp_results,
                    history = history,
                    item = item,
                    master = master,
                    elastic_subproblems = elastic_subproblems,
                    plain_lp_subproblems = plain_lp_subproblems,
                    mip_subproblems = mip_subproblems,
                )
            end
        end

        if verify_integer_recourse
            integer_results = [
                solve_integer_recourse!(get_mip_subproblem(data.scenario), step.x;
                                        feasibility_tol = feasibility_tol)
                for data in scenario_inputs
            ]
            integer_feasible = all(r.is_feasible for r in integer_results)
            if print_log
                println("Integer recourse scaled violations: ",
                    join(["s$(r.scenario)=$(@sprintf("%.2e", r.violation))" for r in integer_results], ", "))
            end

            if integer_feasible
                print_log && println("All LP relaxations and integer recourse subproblems are feasible. Terminating.")
                return (
                    status = :integer_feasible,
                    x = step.x,
                    theta = step.theta,
                    master_objective = step.master_objective,
                    total_cuts = total_cuts,
                    history = history,
                    item = item,
                    master = master,
                    elastic_subproblems = elastic_subproblems,
                    plain_lp_subproblems = plain_lp_subproblems,
                    mip_subproblems = mip_subproblems,
                )
            end

            bad = [r.scenario for r in integer_results if !r.is_feasible]
            print_log && println(
                "LP relaxation is feasible, but integer recourse failed for scenarios ",
                bad,
                ". No LP-relaxation cut is available at this x."
            )
            return (
                status = :lp_feasible_mip_infeasible,
                x = step.x,
                theta = step.theta,
                master_objective = step.master_objective,
                total_cuts = total_cuts,
                failed_integer_scenarios = bad,
                integer_results = integer_results,
                history = history,
                item = item,
                master = master,
                elastic_subproblems = elastic_subproblems,
                plain_lp_subproblems = plain_lp_subproblems,
                mip_subproblems = mip_subproblems,
            )
        end

        print_log && println("All LP relaxations are feasible. Integer recourse was not checked.")
        return (
            status = :lp_relaxation_feasible,
            x = step.x,
            theta = step.theta,
            master_objective = step.master_objective,
            total_cuts = total_cuts,
            history = history,
            item = item,
            master = master,
            elastic_subproblems = elastic_subproblems,
            plain_lp_subproblems = plain_lp_subproblems,
        )
    end

    print_log && println("Phase II iteration limit reached before convergence.")
    return (
        status = :phase2_iteration_limit,
        x = last_x,
        theta = last_theta,
        master_objective = last_master_objective,
        total_cuts = total_cuts,
        history = history,
        item = item,
        master = master,
        elastic_subproblems = elastic_subproblems,
        plain_lp_subproblems = plain_lp_subproblems,
        mip_subproblems = mip_subproblems,
    )
end


# Backwards-compatible wrapper for the earlier file name/API. This routes to the
# elastic two-phase implementation, which avoids solver-dependent Farkas-ray
# handling and is usually faster when early master candidates make the LP
# relaxation infeasible.
function run_benders_lp_relaxation(data_dir::AbstractString;
                                   scenarios = [0, 1],
                                   max_iterations::Int = 400,
                                   optimizer_factory = Xpress.Optimizer,
                                   silent::Bool = true,
                                   big_m_safety_factor::Float64 = 6.0,
                                   verify_integer_recourse::Bool = true,
                                   print_log::Bool = true,
                                   kwargs...)
    phase1_iterations = max(1, min(max_iterations, max(50, max_iterations ÷ 3)))
    phase2_iterations = max(1, max_iterations - phase1_iterations)
    return run_benders_elastic_lp_relaxation(
        data_dir;
        scenarios = scenarios,
        phase1_iterations = phase1_iterations,
        phase2_iterations = phase2_iterations,
        optimizer_factory = optimizer_factory,
        silent = silent,
        big_m_safety_factor = big_m_safety_factor,
        verify_integer_recourse = verify_integer_recourse,
        print_log = print_log,
        kwargs...,
    )
end

# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------

function parse_scenarios(arg::AbstractString)
    stripped = strip(arg)
    if occursin(":", stripped)
        parts = split(stripped, ":")
        length(parts) == 2 || error("Scenario range must look like 0:8")
        return collect(parse(Int, parts[1]):parse(Int, parts[2]))
    end
    return [parse(Int, strip(x)) for x in split(stripped, ",") if !isempty(strip(x))]
end

function parse_bool_arg(arg::AbstractString)
    value = lowercase(strip(arg))
    value in ("true", "t", "yes", "y", "1") && return true
    value in ("false", "f", "no", "n", "0") && return false
    error("Boolean arguments must be true/false, yes/no, or 1/0. Got: $(arg)")
end

function default_data_dir()
    local_data_dir = joinpath(@__DIR__, "inventory_example_data")
    isdir(local_data_dir) && return local_data_dir

    sibling_data_dir = normpath(joinpath(@__DIR__, "..", "inventory_example_data"))
    isdir(sibling_data_dir) && return sibling_data_dir

    return local_data_dir
end

function print_solution_summary(result; max_items_to_print::Int = 20)
    println()
    println("Status: ", result.status)
    println("theta: ", result.theta)
    println("master objective: ", result.master_objective)
    println("total cuts: ", result.total_cuts)
    x = result.x
    println("x length: ", length(x))
    println("selected x count > 1e-6: ", count(>(1.0e-6), x))
    println("selected x count > 0.5: ", count(>(0.5), x))
    println("first x entries:")
    for i in 1:min(length(x), max_items_to_print)
        @printf("  x[%d] = %.8f\n", i, x[i])
    end
    if length(x) > max_items_to_print
        println("  ...")
    end
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Usage:
    #   julia benders_elastic_lp.jl [scenarios] [phase1_iter] [phase2_iter] [data_dir] [verify_mip] [dual_sign]
    # Examples:
    #   julia benders_elastic_lp.jl
    #   julia benders_elastic_lp.jl 0:8 100 300 ./inventory_example_data true 1.0
    scenarios = length(ARGS) >= 1 ? parse_scenarios(ARGS[1]) : [0, 1]
    phase1_iterations = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100
    phase2_iterations = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 300
    data_dir = length(ARGS) >= 4 ? ARGS[4] : default_data_dir()
    verify_integer_recourse = length(ARGS) >= 5 ? parse_bool_arg(ARGS[5]) : true
    dual_sign = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 1.0

    result = run_benders_elastic_lp_relaxation(
        data_dir;
        scenarios = scenarios,
        phase1_iterations = phase1_iterations,
        phase2_iterations = phase2_iterations,
        optimizer_factory = Xpress.Optimizer,
        silent = true,
        verify_plain_lp = true,
        verify_integer_recourse = verify_integer_recourse,
        dual_sign = dual_sign,
        xpress_presolve = true,
        print_log = true,
    )

    print_solution_summary(result)

    x_path = joinpath(@__DIR__, "benders_x.csv")
    open(x_path, "w") do io
        println(io, "item,x")
        for (name, value) in zip(result.item, result.x)
            println(io, name, ",", value)
        end
    end
    println("Final x written to ", x_path)
end
