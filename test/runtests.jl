using DiscreteValueIteration
using POMDPModels
using POMDPs
using Base.Test

# Test basic value iteration functionality
include("runtests_value_iteration_policy.jl") # test the policy object
include("runtests_basic_value_iteration.jl")  # then the creation of a policy
include("runtests_basic_value_iteration_disallowing_actions.jl") # then a complex form where states determine actions
include("runtests_parallel.jl") # test against the serial VI on GridWorld from POMDPModels

println("Testing Requirements")
@requirements_info ValueIterationSolver() GridWorld()
@requirements_info ParallelValueIterationSolver() GridWorld()
