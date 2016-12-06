#####################################################################
# This file implements the value iteration algorithm for solving MDPs
# 
# The following functions are required to use this solver:
# n_states(mdp::POMDP)
# n_actions(mdp::POMDP)
# discount(mdp::POMDP)
# states(mdp::POMDP)
# actions(mdp::POMDP, s::State, actionSpace::AbstractSpace)
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
using POMDPModels
using GridInterpolations

import POMDPs: Solver, solve, Policy, create_policy, action, value 

export
    ValueIterationPolicy,
    ValueIterationSolver,
    create_policy,
    solve,
    action,
    value,
    locals

include("policy.jl")
include("vanilla.jl")

include("multilinear_interpolation.jl")
export
	MultilinearInterpolationValueIterationSolver

include("docs.jl")

end # module
