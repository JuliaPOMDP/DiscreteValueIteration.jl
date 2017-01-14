"""
This module implements a value iteration solver that uses the interface defined in POMDPs.jl
"""
module DiscreteValueIteration

using POMDPs
using POMDPToolbox

import POMDPs: Solver, solve, Policy, action, value 

export
    ValueIterationPolicy,
    ValueIterationSolver,
    create_policy,
    solve,
    action,
    value,
    locals

include("vanilla.jl")
include("docs.jl")

end # module
