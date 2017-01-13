#####################################################################
# This file implements the value iteration algorithm for solving MDPs
# 
# The following functions are required to use this solver:
# n_states(mdp::POMDP)
# n_actions(mdp::POMDP)
# discount(mdp::POMDP)
# states(mdp::POMDP)
# actions(mdp::POMDP, s::State)
# transition(mdp::POMDP, s::State, a::Action)
# reward(mdp::POMDP, s::State, a::Action)
# state_index(mdp, s)
# index(mdp::POMDP, s::State)
# domain(space::AbstractSpace) # state and action spaces
# create_transition_distribution(mdp::POMDP)
#####################################################################
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
