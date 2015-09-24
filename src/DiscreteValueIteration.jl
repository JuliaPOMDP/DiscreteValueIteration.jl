# implements value iteration solver for MDPs

module DiscreteValueIteration

using POMDPs

import POMDPs: Solver, solve, Policy, create_policy, action, value 

export 
    ValueIterationPolicy,
    ValueIterationSolver,
    create_policy,
    solve,
    action,
    value

include("vanilla.jl")

end # module
