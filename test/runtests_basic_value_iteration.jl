
function support_serial_qtest(mdp::Union{MDP,POMDP}, file::AbstractString; niter::Int64=100, res::Float64=1e-3)
    qt = readdlm(file)
    solver = ValueIterationSolver(max_iterations=niter, belres=res)
    policy = create_policy(solver, mdp) 
    policy = solve(solver, mdp, policy, verbose=true)
    (q, u, p, am) = locals(policy)
    npolicy = ValueIterationPolicy(mdp, deepcopy(q))
    nnpolicy = ValueIterationPolicy(mdp, deepcopy(q), deepcopy(u), deepcopy(p))
    s = GridWorldState(1,1)
    a1 = action(policy, s)
    v1 = value(policy, s)
    a2 = action(npolicy, s)
    v2 = value(npolicy, s)
    return (isapprox(qt, q, rtol=1e-5)) && (policy.policy == nnpolicy.policy)
end


function test_complex_gridworld()
	# Load correct policy from file and verify we can reconstruct it
	rstates = [GridWorldState(4,3), GridWorldState(4,6), GridWorldState(9,3), GridWorldState(8,8)]
	rvals = [-10.0, -5.0, 10.0, 3.0]
	xs = 10
	ys = 10
	mdp = GridWorld(sx=xs, sy=ys, rs = rstates, rv = rvals)
	file = "grid-world-10x10-Q-matrix.txt"
	niter = 100
	res = 1e-3

	return support_serial_qtest(mdp, file, niter=niter, res=res)
end

function test_creation_of_policy_given_utilities()
	# If we create a ValueIterationPolicy by providing a utility,
	# does that utility successfully get saved?
	#
	# GridWorld:
	# |_________|_________|_________| plus a fake state for absorbing
	mdp = GridWorld(sx=1, sy=3, rs = [GridWorldState(1,3)], rv = [10.0])
	# Create a starter set of utilities 
	# 0.0 is associated with the fake absorbing state
	correct_utility = [5.45632, 8.20505, 10.0, 0.0] 

	# Derive a policy & check that they match	
	policy = ValueIterationPolicy(mdp, utility=correct_utility, include_Q=true)
	return policy.mdp == mdp && policy.util == correct_utility
end


function test_creation_of_policy_given_q_util_policy()
	# If we create a ValueIterationPolicy by providing a utility, q, and policy,
	# does those values successfully get saved?
	#
	# GridWorld:
	# |_________|_________|_________| plus a fake state for absorbing
	mdp = GridWorld(sx=1, sy=3, rs = [GridWorldState(1,3)], rv = [10.0])
	correct_utility = [5.45632, 8.20505, 10.0, 0.0] 
	correct_qmat = [5.45636 5.1835 5.1835 5.1835; 8.20506 6.47848 7.7948 7.7948; 10.0 10.0 10.0 10.0; 0.0 0.0 0.0 0.0]
	correct_policy = [1,1,1,1]
	
	# Derive a policy & check that they match	
	policy = ValueIterationPolicy(mdp, correct_qmat, correct_utility, correct_policy)
	return (policy.mdp == mdp && policy.qmat == correct_qmat && policy.util == correct_utility && policy.policy == correct_policy && policy.include_Q == true)
end

function test_creation_of_policy_given_q()
	# If we create a ValueIterationPolicy by providing a default q,
	# does its value successfully get saved?
	#
	# GridWorld:
	# |_________|_________|_________| plus a fake state for absorbing
	mdp = GridWorld(sx=1, sy=3, rs = [GridWorldState(1,3)], rv = [10.0])
	correct_qmat = [5.45636 5.1835 5.1835 5.1835; 8.20506 6.47848 7.7948 7.7948; 10.0 10.0 10.0 10.0; 0.0 0.0 0.0 0.0]
	
	# Derive a policy & check that they match	
	policy = ValueIterationPolicy(mdp, correct_qmat)
	return (policy.mdp == mdp && policy.qmat == correct_qmat)
end

function test_simple_grid()
	# Simple test....
	# GridWorld(sx=2,sy=3) w reward at (2,3):
	# Here's our grid:
	# |state (x,y)____available actions__|
	# ----------------------------------------------
	# |5 (1,3)__u,d,l,r__|6 (2,3)__u,d,l,r+REWARD__|
	# |3 (1,2)__u,d,l,r__|4 (2,2)______u,d,l,r_____|
	# |1 (1,1)__u,d,l,r__|2 (2,1)______u,d,l,r_____|
	# ----------------------------------------------
	# 7 (0,0) is absorbing state
	mdp = GridWorld(sx=2, sy=3, rs = [GridWorldState(2,3)], rv = [10.0])
	
	solver = ValueIterationSolver()
    policy = create_policy(solver, mdp) 
    policy = solve(solver, mdp, policy, verbose=true)

	# up: 1, down: 2, left: 3, right: 4
	correct_policy = [1,1,1,1,4,1,1] # alternative policies
	# are possible, but since they are tied & the first 
	# action is always 1, we will always return 1 for tied
	# actions
	return policy.policy == correct_policy
end




@test test_complex_gridworld() == true
@test test_creation_of_policy_given_utilities() == true
@test test_creation_of_policy_given_q_util_policy() == true
@test test_creation_of_policy_given_q() == true
@test test_simple_grid() == true