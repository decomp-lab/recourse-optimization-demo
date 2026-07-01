# Recourse Optimization Demo

Small JuMP example for a multi-scenario planning problem with shared first-stage decisions and scenario-specific recourse decisions.

## Files

- `OptBMExtensive.jl` — builds the extensive-form JuMP model
- `inventory_example_data/` — CSV input data for scenarios 0 to 8

## Running

From the repository folder, start Julia and run:

```julia
include("OptBMExtensive.jl")
```

Or run from a terminal:

```bash
julia OptBMExtensive.jl
```

To instantiate the project environment first:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```
By default, the script builds a small two-scenario instance using scenarios `[0, 1]`. 
The data directory contains additional scenarios; edit the `scenarios` line in `OptBMExtensive.jl` to build a larger instance.
