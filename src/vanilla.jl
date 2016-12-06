# The solver type 
type ValueIterationSolver <: Solver
    max_iterations::Int64 # max number of iterations 
    belres::Float64 # the Bellman Residual
end
# Default constructor
function ValueIterationSolver(;max_iterations::Int64=100, belres::Float64=1e-3)
    return ValueIterationSolver(max_iterations, belres)
end

# The policy type
type ValueIterationPolicy <: Policy
    qmat::Matrix{Float64} # Q matrix storing Q(s,a) values
    util::Vector{Float64} # The value function V(s)
    policy::Vector{Int64} # Policy array, maps state index to action index
    action_map::Vector # Maps the action index to the concrete action type
    include_Q::Bool # Flag for including the Q-matrix
    mdp::Union{MDP,POMDP} # uses the model for indexing in the action function
    # constructor with an optinal initial value function argument
    function ValueIterationPolicy(mdp::Union{MDP,POMDP}; 
                                  utility::Vector{Float64}=Array(Float64,0),
                                  include_Q::Bool=true)
        ns = n_states(mdp)
        na = n_actions(mdp)
        self = new()
        if !isempty(utility)
            @assert size(utilty) == ns "Input utility dimension mismatch"
            self.util = utility
        else
            self.util = zeros(ns)
        end
        self.action_map = ordered_actions(mdp)
        self.policy = zeros(Int64,ns)
        include_Q ? self.qmat = zeros(ns,na) : self.qmat = zeros(0,0)
        self.include_Q = include_Q
        self.mdp = mdp
        return self
    end
    # constructor for solved q, util and policy
    function ValueIterationPolicy(mdp::Union{MDP,POMDP}, q::Matrix{Float64}, util::Vector{Float64}, policy::Vector{Int64})
        self = new()
        self.qmat = q
        self.util = util
        self.policy = policy
        self.action_map = ordered_actions(mdp)
        self.include_Q = true
        self.mdp = mdp
        return self
    end
    # constructor for defualt Q-matrix
    function ValueIterationPolicy(mdp::Union{MDP,POMDP}, q::Matrix{Float64})
        (ns, na) = size(q)
        p = zeros(ns)
        u = zeros(ns)
        for i = 1:ns
            p[i] = indmax(q[i,:])
            u[i] = maximum(q[i,:])
        end
        self = new()
        self.qmat = q
        self.util = u
        self.policy = p
        self.action_map = ordered_actions(mdp)
        self.include_Q = true
        self.mdp = mdp
        return self
    end
end

# returns a default value iteration policy
function create_policy(solver::ValueIterationSolver, mdp::Union{MDP,POMDP})
    return ValueIterationPolicy(mdp)
end

# returns the fields of the policy type
function locals(p::ValueIterationPolicy)
    return (p.qmat,p.util,p.policy,p.action_map)
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
function solve{S,A}(solver::ValueIterationSolver, mdp::Union{MDP{S,A},POMDP{S,A}}, policy=create_policy(solver, mdp); verbose::Bool=false)

    # solver parameters
    max_iterations = solver.max_iterations
    belres = solver.belres
    discount_factor = discount(mdp)

    # intialize the utility and Q-matrix
    util = policy.util
    qmat = policy.qmat
    include_Q = policy.include_Q
    pol = policy.policy 

    # pre-allocate the transtion distirbution and the interpolants
    dist = create_transition_distribution(mdp)

    total_time = 0.0
    iter_time = 0.0

    # create an ordered list of states for fast iteration
    states = ordered_states(mdp)

    # main loop
    for i = 1:max_iterations
        residual = 0.0
        tic()
        # state loop
        for (istate,s) in enumerate(states)
            old_util = util[istate] # for residual 
            max_util = -Inf
            # action loop
            # util(s) = max_a( R(s,a) + discount_factor * sum(T(s'|s,a)util(s') )
            for (iaction, a) in enumerate(policy.action_map)
                dist = transition(mdp, s, a, dist) # fills distribution over neighbors
                u = 0.0
                for sp in iterator(dist)
                    p = pdf(dist, sp)
                    p == 0.0 ? continue : nothing # skip if zero prob
                    r = reward(mdp, s, a, sp)
                    sidx = state_index(mdp, sp)
                    u += p * (r + discount_factor * util[sidx]) 
                end
                new_util = u 
                if new_util > max_util
                    max_util = new_util
                    pol[istate] = iaction
                end
                include_Q ? (qmat[istate, iaction] = new_util) : nothing
            end # action
            # update the value array
            util[istate] = max_util 
            diff = abs(max_util - old_util)
            diff > residual ? (residual = diff) : nothing
        end # state
        iter_time = toq()
        total_time += iter_time
        verbose ? println("Iteration : $i, residual: $residual, iteration run-time: $iter_time, total run-time: $total_time") : nothing
        residual < belres ? break : nothing
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
