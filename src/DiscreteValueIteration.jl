"""
This module implements a value iteration solver that uses the interface defined in POMDPs.jl
"""
module DiscreteValueIteration

using Printf
using POMDPs
using POMDPModelTools
using POMDPPolicies
using SparseArrays

import POMDPs: Solver, solve, Policy, action, value 

export
    ValueIterationPolicy,
    ValueIterationSolver,
    SparseVIPolicy,
    SparseValueIterationSolver,
    solve,
    action,
    value,
    locals

include("common.jl")
include("vanilla.jl")
include("sparse.jl")
include("docs.jl")

end # module
