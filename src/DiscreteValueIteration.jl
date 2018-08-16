"""
This module implements a value iteration solver that uses the interface defined in POMDPs.jl
"""
module DiscreteValueIteration

using Printf
using POMDPs
using POMDPModelTools
using POMDPPolicies

import POMDPs: Solver, solve, Policy, action, value 

export
    ValueIterationPolicy,
    ValueIterationSolver,
    solve,
    action,
    value,
    locals

include("vanilla.jl")
include("docs.jl")

end # module
