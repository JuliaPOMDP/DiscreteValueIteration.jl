mutable struct SpecialGridWorld <: MDP{GridWorldState, GridWorldAction}
    gw::GridWorld
end

POMDPs.discount(g::SpecialGridWorld) = discount(g.gw)
POMDPs.n_states(g::SpecialGridWorld) = n_states(g.gw)
POMDPs.n_actions(g::SpecialGridWorld) = n_actions(g.gw)
POMDPs.transition(g::SpecialGridWorld, s::GridWorldState, a::GridWorldAction) = transition(g.gw, s, a)
POMDPs.reward(g::SpecialGridWorld, s::GridWorldState, a::GridWorldAction, sp::GridWorldState) = reward(g.gw, s, a, sp)
POMDPs.state_index(g::SpecialGridWorld, s::GridWorldState) = state_index(g.gw, s)
POMDPs.action_index(g::SpecialGridWorld, a::GridWorldAction) = action_index(g.gw, a)
POMDPs.actions(g::SpecialGridWorld, s::GridWorldState) = actions(g.gw, s)
POMDPs.states(g::SpecialGridWorld) = states(g.gw)
POMDPs.actions(g::SpecialGridWorld) = actions(g.gw)

# Let's extend actions to hard-code & limit to the
# particular feasible actions from each state....
function POMDPs.actions(mdp::SpecialGridWorld, s::GridWorldState)
    # up: 1, down: 2, left: 3, right: 4
    sidx = state_index(mdp, s)
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


function test_conditioning_actions_on_state()
    # Condition available actions on next state....
    # GridWorld(sx=2,sy=3) w reward at (2,3):
    #
    # Here's our grid, with some actions missing:
    #
    # |state (x,y)____available actions__|
    # |5 (1,3)__l,r__|6 (2,3)__u,d,l,r+REWARD__|
    # |3 (1,2)__l,u__|4 (2,2)_______l,r________|
    # |1 (1,1)__l,r__|2 (2,1)_______u,r________|
    # 7 (0,0) is absorbing state
    mdp = SpecialGridWorld(GridWorld(sx=2, sy=3, rs = [GridWorldState(2,3)], rv = [10.0]))

    solver = ValueIterationSolver(verbose=true)
    policy = solve(solver, mdp)

    println(policy.policy)

    # up: 1, down: 2, left: 3, right: 4
    correct_policy = [4,1,1,3,4,1,1] # alternative policies
    # for state 6 are possible, but since they are ordered
    # such that 1 comes first, 1 will always be the policy
    return policy.policy == correct_policy
end

@test test_conditioning_actions_on_state() == true

println("Finished tests")
