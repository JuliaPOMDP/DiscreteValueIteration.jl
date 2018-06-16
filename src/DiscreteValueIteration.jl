"""
This module implements a value iteration solver that uses the interface defined in POMDPs.jl
"""
module DiscreteValueIteration

using POMDPs
using POMDPToolbox
using Parameters

import POMDPs: Solver, solve, Policy, action, value 

export
    ValueIterationPolicy,
    ValueIterationSolver,
    ParallelValueIterationSolver,
    ParallelValueIterationPolicy,
    create_policy,
    solve,
    action,
    value,
    locals

include("vanilla.jl")
include("parallel.jl")
include("docs.jl")

end # module
