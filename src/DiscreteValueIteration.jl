#####################################################################
# This file implements the value iteration algorithm for solving MDPs
# 
# The following functions are required to use this solver:
# n_states(mdp::POMDP)
# n_actions(mdp::POMDP)
# discount(mdp::POMDP)
# states(mdp::POMDP)
# actions(mdp::POMDP)
# transition(mdp::POMDP, s::State, a::Action)
# reward(mdp::POMDP, s::State, a::Action)
# index(mdp::POMDP, s::State)
# domain(space::AbstractSpace) # state and action spaces
# create_transition_distribution(mdp::POMDP)
#####################################################################

module DiscreteValueIteration

using POMDPs

import POMDPs: Solver, solve, Policy, create_policy, action, value 

export 
    ValueIterationPolicy,
    ValueIterationSolver,
    create_policy,
    solve,
    action,
    value,
    locals

include("vanilla.jl")

end # module
