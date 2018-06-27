"""
This module implements a value iteration solver that uses the interface defined in POMDPs.jl
"""
module DiscreteValueIteration

using POMDPs
using POMDPToolbox

import POMDPs: Solver, solve, Policy, action, value 
import Base: localindexes

export
    ValueIterationPolicy,
    ValueIterationSolver,
    ParallelValueIterationSolver,
    create_policy,
    solve,
    action,
    value,
    locals, 
    ind2state

include("vanilla.jl")
include("parallel.jl")
include("docs.jl")

end # module
