# Tests of locally weighted value iteration

function test_multilinear()
	# Simple test....
	# We start with a 6x6 gridworld, then discretize it into 2x2
	# The reward states are in the bottom right hand side

	bmdp = GridWorld(sx=6, sy=6, rs=[GridWorldState(6,2)], rv = [10.0])
	
	solver = LocallyWeightedValueIterationSolver( TinyGridWorldMDP(bmdp, RectangleGrid([2,5], [2,5])))
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

function test_simplex()
	# Simple test....
	# We start with a 6x6 gridworld, then discretize it into 2x2
	# The reward states are in the bottom right hand side

	bmdp = GridWorld(sx=6, sy=6, rs=[GridWorldState(6,2)], rv = [10.0])
	
	solver = LocallyWeightedValueIterationSolver( TinyGridWorldMDP(bmdp, SimplexGrid([2,5], [2,5])))
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

function test_knn_1()
	# Simple test....
	# We start with a 6x6 gridworld, then discretize it into 2x2
	# The reward states are in the bottom right hand side
	# This tests the single nearest neighbor
	bmdp = GridWorld(sx=6, sy=6, rs=[GridWorldState(6,2)], rv = [10.0])
	
	solver = LocallyWeightedValueIterationSolver( TinyGridWorldMDP(bmdp, KnnGrid(1, [2,5], [2,5])))
	policy = create_policy(solver, bmdp) 
	pp = solve(solver, bmdp, policy, verbose=true)
	
	# up: 1, down: 2, left: 3, right: 4
	correct_policy = [4,1,2,2,1]
	correct_utility = [9.11869,10.0,8.66276,9.11889,0.0]
	# test the policy, utility, value at a gridpoint, value requiring interpolation
	return (pp.subspace_policy.policy == correct_policy) &&
			(isapprox(pp.subspace_policy.util, correct_utility, rtol=1e-5)) &&
			(isapprox(value(policy, GridWorldState(2,1)), 9.118695, rtol=1e-5 )) &&
			(isapprox(value(policy, GridWorldState(2,4)), 8.66276, rtol=1e-5 )) &&
			(isapprox(value(policy, GridWorldState(6,6)), 9.11889, rtol=1e-5 )) &&
			(isapprox(value(policy, GridWorldState(6,0)), 10.0, rtol=1e-5 ))
end


function test_knn_2()
	# Simple test....
	# We start with a 6x6 gridworld, then discretize it into 2x2
	# The reward states are in the bottom right hand side
	# This tests the 2-nearest neighbors
	# -------------------
	# 6`
	# 5  8.67      9.19
	# 4 
	# 3
	# 2  9.19      10.0
	# 1
	#   1  2  3  4  5  6
	# -------------------
	
	bmdp = GridWorld(sx=6, sy=6, rs=[GridWorldState(6,2)], rv = [10.0])
	
	solver = LocallyWeightedValueIterationSolver( TinyGridWorldMDP(bmdp, KnnGrid(2, [2,5], [2,5])))
	policy = create_policy(solver, bmdp) 
	pp = solve(solver, bmdp, policy, verbose=true)
	
	# up: 1, down: 2, left: 3, right: 4
	correct_policy = [4,1,2,2,1]
	correct_utility = [9.11869,10.0,8.66276,9.11889,0.0]
	# test the policy, utility, value at a gridpoint, value requiring interpolation
	return (pp.subspace_policy.policy == correct_policy) &&
			(isapprox(pp.subspace_policy.util, correct_utility, rtol=1e-5)) &&
			(isapprox(value(policy, GridWorldState(2,1)), 9.559345, rtol=1e-5 )) &&
			(isapprox(value(policy, GridWorldState(2,4)), 8.890727, rtol=1e-5 )) &&
			(isapprox(value(policy, GridWorldState(5,0)), 9.559345, rtol=1e-5 ))
end


function test_knnfast_1()
	# Simple test....
	# We start with a 6x6 gridworld, then discretize it into 2x2
	# The reward states are in the bottom right hand side
	# This tests the single nearest neighbor
	bmdp = GridWorld(sx=6, sy=6, rs=[GridWorldState(6,2)], rv = [10.0])
	
	solver = LocallyWeightedValueIterationSolver( TinyGridWorldMDP(bmdp, KnnFastGrid(1, [2,5], [2,5])))
	policy = create_policy(solver, bmdp) 
	pp = solve(solver, bmdp, policy, verbose=true)
	
	# up: 1, down: 2, left: 3, right: 4
	correct_policy = [4,1,2,2,1]
	correct_utility = [9.11869,10.0,8.66276,9.11889,0.0]
	# test the policy, utility, value at a gridpoint, value requiring interpolation
	return (pp.subspace_policy.policy == correct_policy) &&
			(isapprox(pp.subspace_policy.util, correct_utility, rtol=1e-5)) &&
			(isapprox(value(policy, GridWorldState(2,1)), 9.118695, rtol=1e-5 )) &&
			(isapprox(value(policy, GridWorldState(2,4)), 8.66276, rtol=1e-5 )) &&
			(isapprox(value(policy, GridWorldState(6,6)), 9.11889, rtol=1e-5 )) &&
			(isapprox(value(policy, GridWorldState(6,0)), 10.0, rtol=1e-5 ))
end


function test_knnfast_2()
	# Simple test....
	# We start with a 6x6 gridworld, then discretize it into 2x2
	# The reward states are in the bottom right hand side
	# This tests the 2-nearest neighbors
	# -------------------
	# 6`
	# 5  8.67      9.19
	# 4 
	# 3
	# 2  9.19      10.0
	# 1
	#   1  2  3  4  5  6
	# -------------------
	
	bmdp = GridWorld(sx=6, sy=6, rs=[GridWorldState(6,2)], rv = [10.0])
	
	solver = LocallyWeightedValueIterationSolver( TinyGridWorldMDP(bmdp, KnnFastGrid(2, [2,5], [2,5])))
	policy = create_policy(solver, bmdp) 
	pp = solve(solver, bmdp, policy, verbose=true)
	
	# up: 1, down: 2, left: 3, right: 4
	correct_policy = [4,1,2,2,1]
	correct_utility = [9.11869,10.0,8.66276,9.11889,0.0]
	
	# test the policy, utility, value at a gridpoint, value requiring interpolation
	return (pp.subspace_policy.policy == correct_policy) &&
			(isapprox(pp.subspace_policy.util, correct_utility, rtol=1e-5)) &&
			(isapprox(value(policy, GridWorldState(2,1)), 9.559345, rtol=1e-5 )) &&
			(isapprox(value(policy, GridWorldState(2,4)), 8.890727, rtol=1e-5 )) &&
			(isapprox(value(policy, GridWorldState(5,0)), 9.559345, rtol=1e-5 ))
end

@test test_multilinear() == true
@test test_simplex() == true
@test test_knn_1() == true
@test test_knn_2() == true
@test test_knnfast_1() == true
@test test_knnfast_2() == true