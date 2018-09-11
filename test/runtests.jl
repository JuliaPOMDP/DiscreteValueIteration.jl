using DiscreteValueIteration
using POMDPModels
using POMDPModelTools
using POMDPs
using Test

# Test basic value iteration functionality
@testset "all" begin
@testset "policy" begin
    include("test_value_iteration_policy.jl") # test the policy object
end
@testset "basic" begin
    include("test_basic_value_iteration.jl")  # then the creation of a policy
end
@testset "basic disallowing actions" begin
    include("test_basic_value_iteration_disallowing_actions.jl") # then a complex form where states determine actions
end

println("Testing Requirements")
@requirements_info ValueIterationSolver() GridWorld()
end
