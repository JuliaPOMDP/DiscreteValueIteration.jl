using DiscreteValueIteration
using POMDPModels
using POMDPs
using Base.Test

# Although there must be a way to overwrite the actions method of a 
# GridWorld instance, it is still unclear how to do that:
# -- Can't include the POMDPs.actions function inside the test 
#    inside the runtests.jl file.  If it runs first, then the simple
#    tests fail because they are using the wrong actions() method.  
#    This failure occurs even if we re-overwrite the POMDPs.actions function
#    for those tests to include all 4 possible actions.  If it runs
#    second, then the POMDPs.actions is not successfully overwritten
#    for the limited-actions case, and the verification of new functionality 
#    test fails.  This occurs regardless of whether I include 
#    `import POMDPs.actions` at the top of the runtests.jl file as per
#    http://stackoverflow.com/questions/28188800/can-i-write-i-julia-method-that-works-whenever-possible-like-a-c-template-fu/28204432
# -- Can't include it as a separate test file.  Julia wants a single test
#    file named runtests.jl, rather than running all test files in the
#    tests directory.  Including a separate file like this one with an
#    `include` statement in the runtests.jl file also does not succeed
#    in overwriting the POMDPs.actions method, whether the POMDPs.actions 
#    method is defined inside or outside of the test.
# -- Creating a separate SpecialGridWorld Experiment is possible, but seems
#    to require a substantial amount of duplicate code.  Initial attempts 
#    to overwrite particular instances of GridWorld functionality in a 
#    separate SpecialGridWorld Experiment have been unsuccessful. Since 
#    code duplication is inherently bad, I also don't use this approach.
#
# This test illustrates that the DiscreteValueIteration Solver will note 
# and behave correctly for states where the number of possible next actions 
# is limited programmatically. This functionality was not previously available.
#
# Currently the test can be run with:
#   julia runtests_disallowing_actions.jl
#
	
	

# Let's extend actions to hard-code & limit to the 
# particular feasible actions from each state....
function POMDPs.actions(mdp::GridWorld, s::GridWorldState, as::GridWorldActionSpace)
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
	return GridWorldActionSpace(acts)
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
	mdp = GridWorld(sx=2, sy=3, rs = [GridWorldState(2,3)], rv = [10.0])
	
	solver = ValueIterationSolver()
    policy = create_policy(solver, mdp)
    policy = solve(solver, mdp, policy, verbose=true)

	println(policy.policy)
	
	# up: 1, down: 2, left: 3, right: 4
	correct_policy = [4,1,1,3,4,1,1] # alternative policies
	# for state 6 are possible, but since they are ordered
	# such that 1 comes first, 1 will always be the policy
	return policy.policy == correct_policy
end

@test test_conditioning_actions_on_state() == true

println("Finished tests")