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
    mdp = SpecialGridWorld(LegacyGridWorld(sx=2, sy=3, rs = [GridWorldState(2,3)], rv = [10.0]))

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
