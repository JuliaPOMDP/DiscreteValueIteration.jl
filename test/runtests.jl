using DiscreteValueIteration
using POMDPModels
using POMDPs
using Base.Test



# Test basic value iteration functionality
include("runtests_value_iteration_policy.jl") # test the policy object

include("runtests_basic_value_iteration.jl")  # then the creation of a policy
#include("runtests_basic_value_iteration_disallowing_actions.jl") # then a complex form where states determine actions



# Read & and test the supporting example models for 
# locally approximate value iteration
include("TinyGridWorldMDP.jl")
include("runtests_tiny_gridworld.jl")


# Test locally approximate value iteration
# includes multilinear, simplex interpolation
include("runtests_locally_weighted_value_iteration.jl")




println("Finished tests")



