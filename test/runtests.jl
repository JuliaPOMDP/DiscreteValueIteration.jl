using DiscreteValueIteration
using POMDPModels
using POMDPTools
using POMDPs
using Test
import POMDPLinter: @show_requirements

# special grid world for some tests:
#  - reward(m, s, a, sp)
#  - constrained actions
mutable struct SpecialGridWorld <: MDP{GridWorldState, GridWorldAction}
    gw::LegacyGridWorld
end

POMDPs.discount(g::SpecialGridWorld) = discount(g.gw)
POMDPs.transition(g::SpecialGridWorld, s::GridWorldState, a::GridWorldAction) = transition(g.gw, s, a)
POMDPs.reward(g::SpecialGridWorld, s::GridWorldState, a::GridWorldAction, sp::GridWorldState) = reward(g.gw, s, a, sp)
POMDPs.stateindex(g::SpecialGridWorld, s::GridWorldState) = stateindex(g.gw, s)
POMDPs.actionindex(g::SpecialGridWorld, a::GridWorldAction) = actionindex(g.gw, a)
POMDPs.actions(g::SpecialGridWorld, s::GridWorldState) = actions(g.gw, s)
POMDPs.states(g::SpecialGridWorld) = states(g.gw)
POMDPs.actions(g::SpecialGridWorld) = actions(g.gw)

SpecialGridWorld() = SpecialGridWorld(LegacyGridWorld(sx=2, sy=3, rs = [GridWorldState(2,3)], rv = [10.0]))

# Let's extend actions to hard-code & limit to the
# particular feasible actions from each state....
function POMDPs.actions(mdp::SpecialGridWorld, s::GridWorldState)
    # up: 1, down: 2, left: 3, right: 4
    sidx = stateindex(mdp, s)
    if sidx == 1
        acts = [GridWorldAction(:left), GridWorldAction(:right)]
    elseif sidx == 2
        acts = [GridWorldAction(:up), GridWorldAction(:right)]
    elseif sidx == 3
        acts = [GridWorldAction(:left), GridWorldAction(:up)]
    elseif sidx == 4
        acts = [GridWorldAction(:left), GridWorldAction(:right)]
    elseif sidx == 5
        acts = [GridWorldAction(:left), GridWorldAction(:right)]
    elseif sidx == 6
        acts = [GridWorldAction(:up), GridWorldAction(:down), GridWorldAction(:left), GridWorldAction(:right)]
    elseif sidx == 7
        acts = [GridWorldAction(:up), GridWorldAction(:down), GridWorldAction(:left), GridWorldAction(:right)]
    end
    return acts
end

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

    @testset "sparse" begin
        include("test_sparse.jl")
    end

    println("Testing Requirements")
    @show_requirements solve(ValueIterationSolver(), LegacyGridWorld())
end
