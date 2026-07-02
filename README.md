# Multi-scenario inventory optimisation example

This repository contains a small Julia/JuMP example for a multi-scenario inventory optimisation problem.

The first-stage decision is a shared initial allocation vector `x`. Each scenario then has its own recourse decisions for using initial inventory and placing/using orders. The objective is to minimise the maximum initial value required across the scenarios.

## Files

* `OptBMExtensive.jl`  
  Extensive-form reference model.

* `benders_elastic_lp.jl`  
  Benders implementation using elastic LP-relaxation scenario subproblems.

* `inventory_example_data/`  
  Small example data set.

## Requirements

The code is written for Julia/JuMP and was tested with FICO Xpress.

Install the Julia dependencies from the project environment:

```julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Extensive form

To build the extensive form on the small two-scenario case, use:

```bash
julia OptBMExtensive.jl
```

The extensive form is mainly included as a reference implementation. It becomes difficult to solve directly as the number of scenarios increases.

## Benders LP-relaxation approach

The Benders implementation is in:

```text
benders_elastic_lp.jl
```

The method is a two-phase LP-relaxation Benders approach:

1. The master problem contains the shared first-stage allocation `x`, the objective variable `theta`, and one elastic feasibility estimator `eta_s` per scenario.
2. Phase 1 minimises total elastic LP violation to find an `x` that is feasible for the LP-relaxed scenario recourse problems.
3. Phase 2 minimises `theta` while keeping the elastic violation budget close to zero.
4. After convergence, the code verifies:
   * plain LP recourse feasibility; and
   * binary recourse feasibility using an elastic MIP check for each scenario.

The Benders cuts are generated from the LP relaxation of each scenario subproblem. The binary recourse problem is not used to generate cuts; it is checked after the LP-relaxation Benders loop has converged.

The elastic MIP check keeps the timing variables binary but retains scaled slack variables. The reported violations are therefore minimum scaled recourse violations at the candidate `x`, not raw constraint residuals.

## Example commands

Two scenarios:

```bash
julia benders_elastic_lp.jl 0:1 100 300 ./inventory_example_data true 1.0
```

All nine scenarios:

```bash
julia benders_elastic_lp.jl 0:8 100 300 ./inventory_example_data true 1.0
```

Arguments are:

```text
julia benders_elastic_lp.jl [scenarios] [phase1_iterations] [phase2_iterations] [data_dir] [verify_binary_recourse] [dual_sign]
```

For example, `0:8` runs scenarios 0 to 8.

## Current behaviour

For the nine-scenario case, the LP-relaxation Benders loop converges to an LP-recourse-feasible solution, and the plain LP recourse check passes for all scenarios.

However, the final elastic MIP recourse check reports positive scaled violations for all scenarios at the current tolerance. This means the method currently gives an LP-relaxation lower bound, but it has not recovered a feasible first-stage solution for the original binary-recourse problem.

Summary from one run:

| Metric                 |                                                     Result |
| ---------------------- | ---------------------------------------------------------: |
| Status                 | LP-recourse feasible, binary recourse not verified feasible |
| LP lower bound `theta` |                                                `4.71093e8` |
| Cuts                   |                                                      `135` |
| Plain LP verification  |                                 Passed for all 9 scenarios |
| Binary recourse check  | Positive scaled violations for all scenarios at `1e-4` tol |

Minimum scaled violations from the elastic MIP checks:

```text
s0=9.3e-3, s1=2.2e-3, s2=2.4e-3, s3=5.8e-2, s4=6.4e-3,
s5=9.2e-4, s6=5.3e-3, s7=1.4e-3, s8=1.5e-3
```

The main open modelling/decomposition question is whether LP-relaxation Benders is only useful as a lower-bound method here, or whether there is a practical way to strengthen or extend it so that the final shared `x` is feasible for the original binary-recourse problem.

## Notes

This is an experimental implementation intended to make the decomposition issue reproducible. It is not intended to be a polished solver package.
