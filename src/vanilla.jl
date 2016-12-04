# The vanilla discrete value iteration solver solves for 
# each and every discrete state.

# The solver type 
type ValueIterationSolver <: Solver
    max_iterations::Int64 # max number of iterations 
    belres::Float64 # the Bellman Residual
end
# Default constructor
function ValueIterationSolver(;max_iterations::Int64=100, belres::Float64=1e-3)
    return ValueIterationSolver(max_iterations, belres)
end

# returns a default value iteration policy
function create_policy(solver::ValueIterationSolver, mdp::Union{MDP,POMDP})
    return ValueIterationPolicy(mdp)
end

#####################################################################
# Solve runs the value iteration algorithm.
# The policy input argument is either provided by the user or 
# initialized during the function call.
# Verbose is a flag that triggers text output to the command line
# Example code for running the function:
# mdp = GridWorld(10, 10) # initialize a 10x10 grid world MDP (user written code)
# solver = ValueIterationSolver(max_iterations=40, belres=1e-3)
# policy = ValueIterationPolicy(mdp)
# solve(solver, mdp, policy, verbose=true) 
#####################################################################
function solve(solver::ValueIterationSolver, mdp::Union{MDP,POMDP}, policy=create_policy(solver, mdp); verbose::Bool=false)

    # pre-allocate the transition distribution and the interpolants
    dist = create_transition_distribution(mdp)

    total_time = 0.0
    iter_time = 0.0

    # main loop
    for i = 1:solver.max_iterations
        residual = 0.0
        tic()
        # state loop
        for s in iterator(states(mdp))
            sidx = state_index(mdp, s)
            old_util = policy.util[sidx] # for residual 
            max_util = -Inf
            # action loop
            # util(s) = max_a( R(s,a) + discount_factor * sum(T(s'|s,a)util(s') )
			# Checks only the subset of actions available  
			# from each state (conditions actions on state)
            for a in iterator(actions(mdp, s, actions(mdp)))
                aidx = action_index(mdp, a)
                dist = transition(mdp, s, a, dist) # fills distribution over neighbors
                u = 0.0
                for sp in iterator(dist)
                    p = pdf(dist, sp)
                    p == 0.0 ? continue : nothing # skip if zero prob
                    r = reward(mdp, s, a, sp)
                    spidx = state_index(mdp, sp)
                    u += p * (r + discount(mdp) * policy.util[spidx]) 
                end
                if u > max_util
                    max_util = u
                    policy.policy[sidx] = aidx
                end
                policy.include_Q ? (policy.qmat[sidx, aidx] = u) : nothing
            end # action
            # update the value array
            policy.util[sidx] = max_util 
            diff = abs(max_util - old_util)
            diff > residual ? (residual = diff) : nothing
        end # state
        iter_time = toq()
        total_time += iter_time
        verbose ? println("Iteration : $i, residual: $residual, iteration run-time: $iter_time, total run-time: $total_time") : nothing
        residual < solver.belres ? break : nothing
    end # main
    policy
end

function action{S,A}(policy::ValueIterationPolicy, s::S, a::A=nothing)
    sidx = state_index(policy.mdp, s)
    aidx = policy.policy[sidx]
    return policy.action_map[aidx]
end
function value{S}(policy::ValueIterationPolicy, s::S)
    sidx = state_index(policy.mdp, s)
    policy.util[sidx]
end

