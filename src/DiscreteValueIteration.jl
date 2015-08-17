# implements value iteration solver for MDPs

module DiscreteValueIteration

using POMDPs

import POMDPs: Solver, solve!, Policy, action, value 

export 
    ValueIterationPolicy,
    ValueIterationSolver,
    solve!

typealias Action Any

include("vanilla.jl")

end # module
