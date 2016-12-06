using DiscreteValueIteration
using POMDPModels
using POMDPs
using Base.Test



# Test basic value iteration functionality
include("runtests_basic_value_iteration.jl")
#include("runtests_basic_value_iteration_disallowing_actions.jl")



# Read & and test the supporting example models for 
# locally approximate value iteration
include("TinyGridWorldMDP.jl")
include("runtests_tiny_gridworld.jl")


# Test locally approximate value iteration
include("runtests_multilinear_interpolation_value_iteration.jl")




println("Finished tests")



