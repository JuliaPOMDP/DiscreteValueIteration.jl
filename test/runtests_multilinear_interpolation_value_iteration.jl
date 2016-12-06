# Tests of multilinear interpolation, the MultilinearInterpolationValueIterationSolver.

function test_locally_weighted()
	# Simple test....
	# We start with a 6x6 gridworld, then discretize it into 2x2
	# The reward states are in the bottom right hand side

	bmdp = GridWorld(sx=6, sy=6, rs=[GridWorldState(6,2)], rv = [10.0])

	
	tiny = TinyGridWorldMDP(bmdp, [2,5], [2,5])
	solver = MultilinearInterpolationValueIterationSolver(tiny)
	policy = create_policy(solver, bmdp) 
	pp = solve(solver, bmdp, policy, verbose=true)

	# up: 1, down: 2, left: 3, right: 4
	correct_policy = [4,1,2,2,1]
	correct_utility = [9.11869,10.0,8.66276,9.11889,0.0]
	# test the policy, utility, value at a gridpoint, value requiring interpolation
	return (pp.subspace_policy.policy == correct_policy) &&
			(isapprox(pp.subspace_policy.util, correct_utility, rtol=1e-5)) &&
			(isapprox(value(policy, GridWorldState(2,1)), 9.118695, rtol=1e-5 )) &&
			(isapprox(value(policy, GridWorldState(2,4)), 8.814738, rtol=1e-5 ))
end

@test test_locally_weighted() == true