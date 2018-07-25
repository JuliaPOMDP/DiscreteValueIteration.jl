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
    ParallelValueIterationNotSharedSolver,
    ParallelValueIterationMacroSolver,
    solve,
    action,
    value,
    locals

include("vanilla.jl")
include("parallel.jl")
include("parallel_not_shared.jl")
include("parallel_@parallel.jl")
include("docs.jl")

end # module
