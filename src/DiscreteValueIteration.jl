#####################################################################
# This file implements the value iteration algorithm for solving MDPs
# 
# The following functions are required to use this solver:
# discount(mdp::POMDP)
# states(mdp::POMDP)
# actions(mdp::POMDP)
# transition(mdp::POMDP, s::State, a::Action)
# reward(mdp::POMDP, s::State, a::Action)
# length(d::AbstractDistribution)
# index(d::AbstractDistribution, i::Int64)
# weight(d::AbstractDistribution, i::Int64)
# domain(space::AbstractSpace) # for action and observation spaces
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
