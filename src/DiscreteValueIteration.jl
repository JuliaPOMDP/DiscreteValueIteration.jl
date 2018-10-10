"""
This module implements a value iteration solver that uses the interface defined in POMDPs.jl
"""
module DiscreteValueIteration

using Printf
using POMDPs
using POMDPModelTools
using POMDPPolicies
using Distributed
using SharedArrays

import POMDPs: Solver, solve, Policy, action, value 

export
    ValueIterationPolicy,
    ValueIterationSolver,
    ParallelValueIterationSolver,
    solve,
    action,
    value,
    locals

include("vanilla.jl")
include("parallel.jl")
include("docs.jl")

end # module
