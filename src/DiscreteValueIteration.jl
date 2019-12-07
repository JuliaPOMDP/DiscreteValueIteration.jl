"""
This module implements a value iteration solver that uses the interface defined in POMDPs.jl
"""
module DiscreteValueIteration

using Printf
using POMDPs
using POMDPModelTools
using POMDPPolicies
using SparseArrays
using Distributed
using SharedArrays

import POMDPs: Solver, solve, Policy, action, value 

export
    ValueIterationPolicy,
    ValueIterationSolver,
    SparseValueIterationSolver,
    ParallelValueIterationSolver,
    ind2state
    solve,
    action,
    value

include("common.jl")
include("vanilla.jl")
include("sparse.jl")
include("parallel.jl")

end # module
