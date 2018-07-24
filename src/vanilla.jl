# The solver type
mutable struct ValueIterationSolver <: Solver
    max_iterations::Int64 # max number of iterations
    belres::Float64 # the Bellman Residual
    verbose::Bool 
    include_Q::Bool
    init_util::Vector{Float64}
end
# Default constructor
function ValueIterationSolver(;max_iterations::Int64 = 100, 
                               belres::Float64 = 1e-3,
                               verbose::Bool = false,
                               include_Q::Bool = true,
                               init_util::Vector{Float64}=Vector{Float64}(0))    
    return ValueIterationSolver(max_iterations, belres, verbose, include_Q, init_util)
end

# The policy type
mutable struct ValueIterationPolicy <: Policy
    qmat::Matrix{Float64} # Q matrix storing Q(s,a) values
    util::Vector{Float64} # The value function V(s)
    policy::Vector{Int64} # Policy array, maps state index to action index
    action_map::Vector # Maps the action index to the concrete action type
    include_Q::Bool # Flag for including the Q-matrix
    mdp::Union{MDP,POMDP} # uses the model for indexing in the action function
end

# constructor with an optinal initial value function argument
function ValueIterationPolicy(mdp::Union{MDP,POMDP};
                              utility::Vector{Float64}=zeros(n_states(mdp)),
                              policy::Vector{Int64}=zeros(Int64, n_states(mdp)),
                              include_Q::Bool=true)
    ns = n_states(mdp)
    na = n_actions(mdp)
    @assert length(utility) == ns "Input utility dimension mismatch"
    @assert length(policy) == ns "Input policy dimension mismatch"
    action_map = ordered_actions(mdp)
    include_Q ? qmat = zeros(ns,na) : qmat = zeros(0,0)
    return ValueIterationPolicy(qmat, utility, policy, action_map, include_Q, mdp)
end

# constructor for solved q, util and policy
function ValueIterationPolicy(mdp::Union{MDP,POMDP}, q::Matrix{Float64}, util::Vector{Float64}, policy::Vector{Int64})
    action_map = ordered_actions(mdp)
    include_Q = true
    return ValueIterationPolicy(q, util, policy, action_map, include_Q, mdp)
end

# constructor for default Q-matrix
function ValueIterationPolicy(mdp::Union{MDP,POMDP}, q::Matrix{Float64})
    (ns, na) = size(q)
    p = zeros(ns)
    u = zeros(ns)
    for i = 1:ns
        p[i] = indmax(q[i,:])
        u[i] = maximum(q[i,:])
    end
    action_map = ordered_actions(mdp)
    include_Q = true
    return ValueIterationPolicy(q, u, p, action_map, include_Q, mdp)
end

# returns the fields of the policy type
function locals(p::ValueIterationPolicy)
    return (p.qmat,p.util,p.policy,p.action_map)
end

@POMDP_require solve(solver::ValueIterationSolver, mdp::Union{MDP,POMDP}) begin
    P = typeof(mdp)
    S = state_type(P)
    A = action_type(P)
    @req discount(::P)
    @req n_states(::P)
    @req n_actions(::P)
    @subreq ordered_states(mdp)
    @subreq ordered_actions(mdp)
    @req transition(::P,::S,::A)
    @req reward(::P,::S,::A,::S)
    @req state_index(::P,::S)
    @req action_index(::P, ::A)
    @req actions(::P, ::S)
    as = actions(mdp)
    ss = states(mdp)
    @req iterator(::typeof(as))
    @req iterator(::typeof(ss))
    s = first(iterator(ss))
    a = first(iterator(as))
    dist = transition(mdp, s, a)
    D = typeof(dist)
    @req iterator(::D)
    @req pdf(::D,::S)
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
function solve(solver::ValueIterationSolver, mdp::Union{MDP,POMDP}; kwargs...)
    
    # deprecation warning - can be removed when Julia 1.0 is adopted
    if !isempty(kwargs)
        warn("Keyword args for solve(::ValueIterationSolver, ::MDP) are no longer supported. For verbose output, use the verbose option in the ValueIterationSolver")
    end
    
    @warn_requirements solve(solver, mdp)

    # solver parameters
    max_iterations = solver.max_iterations
    belres = solver.belres
    discount_factor = discount(mdp)
    ns = n_states(mdp)
    na = n_actions(mdp)

    # intialize the utility and Q-matrix
    if !isempty(solver.init_util)
        @assert length(solver.init_util) == ns "Input utility dimension mismatch"
        util = solver.init_util
    else
        util = zeros(ns)
    end
    include_Q = solver.include_Q
    if include_Q
        qmat = zeros(ns, na)
    end
    pol = zeros(Int64, ns)

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
            sub_aspace = actions(mdp, s)
            if isterminal(mdp, s)
                util[istate] = 0.0
                pol[istate] = 1
            else
                old_util = util[istate] # for residual
                max_util = -Inf
                # action loop
                # util(s) = max_a( R(s,a) + discount_factor * sum(T(s'|s,a)util(s') )
                for a in iterator(sub_aspace)
                    iaction = action_index(mdp, a)
                    dist = transition(mdp, s, a) # creates distribution over neighbors
                    u = 0.0
                    for (sp, p) in weighted_iterator(dist)
                        p == 0.0 ? continue : nothing # skip if zero prob
                        r = reward(mdp, s, a, sp)
                        isp = state_index(mdp, sp)
                        u += p * (r + discount_factor * util[isp])
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
            end
        end # state
        iter_time = toq()
        total_time += iter_time
        solver.verbose ? @printf("[Iteration %-4d] residual: %10.3G | iteration runtime: %10.3f ms, (%10.3G s total)\n", i, residual, iter_time*1000.0, total_time) : nothing
        residual < belres ? break : nothing
    end # main
    if include_Q
        return ValueIterationPolicy(mdp, qmat, util, pol)
    else
        return ValueIterationPolicy(mdp, utility=util, policy=pol, include_Q=false)
    end
end

function action(policy::ValueIterationPolicy, s::S) where S
    sidx = state_index(policy.mdp, s)
    aidx = policy.policy[sidx]
    return policy.action_map[aidx]
end

function value(policy::ValueIterationPolicy, s::S) where S
    sidx = state_index(policy.mdp, s)
    policy.util[sidx]
end