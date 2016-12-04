using DiscreteValueIteration
using POMDPModels
using POMDPs
using Base.Test


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
	
	
    solver = ValueIterationSolver(max_iterations=niter, belres=res)
    policy = solve(solver, mdp, policy, verbose=true)
	
	println( policy.util)
	println( policy.qmat)
	println( policy.policy)
	println( policy.include_Q)
	
	return policy.util == correct_utility
end


function test_creation_of_policy_given_q_util_policy()
	# If we create a ValueIterationPolicy by providing a utility, q, and policy,
	# does those values successfully get saved?
	#
	# GridWorld:
	# |_________|_________|_________| plus a fake state for absorbing
	mdp = GridWorld(sx=1, sy=3, rs = [GridWorldState(1,3)], rv = [10.0])
	# Create a starter set of utilities 
	# 0.0 is associated with the fake absorbing state
	correct_utility = [5.45632, 8.20505, 10.0, 0.0] 

	# Derive a policy & check that they match	
	policy = ValueIterationPolicy(mdp, utility=utility, include_Q=true)
	return policy.util == correct_utility
end

#=
    function ValueIterationPolicy(mdp::Union{MDP,POMDP}, q::Matrix{Float64}, util::Vector{Float64}, policy::Vector{Int64})
        self = new()
        self.qmat = q
        self.util = util
        self.policy = policy
        am = Any[]
        for a in iterator(actions(mdp))
            aidx = action_index(mdp, a)
            push!(am, aidx)
        end
        self.action_map = am
        self.include_Q = true
        self.mdp = mdp
        return self
    end
=#	

@test test_complex_gridworld() == true
@test test_creation_of_policy_given_utilities() == true
#@test test_creation_of_policy_given_q_util_policy() == true

println("Finished tests")



