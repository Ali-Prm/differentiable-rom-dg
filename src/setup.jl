
#Run once to install all required Julia packages:
#julia --project=. fom/setup.jl

using Pkg
Pkg.add([
    "Trixi",
    "OrdinaryDiffEq",
    "OrdinaryDiffEqSSPRK",
    "HDF5",
    "StaticArrays",
    "Printf",
    "ProgressMeter",
])
Pkg.resolve()
println("Setup complete.")
