# The policy type
"""
    ValueIterationPolicy <: Policy 

The policy type. Contains the Q-Matrix, the Utility function and an array of indices corresponding to optimal actions.
There are three ways to initialize the policy type:

    `policy = ValueIterationPolicy(mdp)` 
    `policy = ValueIterationPolicy(mdp, utility_array)`
    `policy = ValueIterationPolicy(mdp, qmatrix)`

The Q-matrix is nxm, where n is the number of states and m is the number of actions.

# Fields 
- `qmat`  Q matrix storing Q(s,a) values
- `util` The value function V(s)
- `policy` Policy array, maps state index to action index
- `action_map` Maps the action index to the concrete action type
- `include_Q` Flag for including the Q-matrix
- `mdp`  uses the model for indexing in the action function
"""
struct ValueIterationPolicy{Q<:AbstractMatrix, U<:AbstractVector, P<:AbstractVector, A, M<:MDP} <: Policy
    qmat::Q
    util::U 
    policy::P 
    action_map::Vector{A}
    include_Q::Bool 
    mdp::M
end

# constructor with an optinal initial value function argument
function ValueIterationPolicy(mdp::Union{MDP,POMDP};
                              utility::AbstractVector{Float64}=zeros(n_states(mdp)),
                              policy::AbstractVector{Int64}=zeros(Int64, n_states(mdp)),
                              include_Q::Bool=true)
    ns = length(states(mdp))
    na = length(actions(mdp))
    @assert length(utility) == ns "Input utility dimension mismatch"
    @assert length(policy) == ns "Input policy dimension mismatch"
    action_map = ordered_actions(mdp)
    include_Q ? qmat = zeros(ns,na) : qmat = zeros(0,0)
    return ValueIterationPolicy(qmat, utility, policy, action_map, include_Q, mdp)
end

# constructor for solved q, util and policy
function ValueIterationPolicy(mdp::Union{MDP,POMDP}, q::AbstractMatrix{F}, util::AbstractVector{F}, policy::Vector{Int64}) where {F}
    action_map = ordered_actions(mdp)
    include_Q = true
    return ValueIterationPolicy(q, util, policy, action_map, include_Q, mdp)
end

# constructor for default Q-matrix
function ValueIterationPolicy(mdp::Union{MDP,POMDP}, q::AbstractMatrix{F}) where {F}
    (ns, na) = size(q)
    p = zeros(Int64, ns)
    u = zeros(ns)
    for i = 1:ns
        p[i] = argmax(q[i,:])
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

function action(policy::ValueIterationPolicy, s::S) where S
    sidx = stateindex(policy.mdp, s)
    aidx = policy.policy[sidx]
    return policy.action_map[aidx]
end

function value(policy::ValueIterationPolicy, s::S) where S
    sidx = stateindex(policy.mdp, s)
    policy.util[sidx]
end

value(policy::ValueIterationPolicy, s, a) = actionvalues(policy, s)[actionindex(policy.mdp, a)]

function POMDPPolicies.actionvalues(policy::ValueIterationPolicy, s::S) where S
    if !policy.include_Q
        error("ValueIterationPolicy does not contain the Q matrix. Use the include_Q=true keyword argument in the solver.")
    else
        sidx = stateindex(policy.mdp, s)
        return policy.qmat[sidx,:]
    end
end

function Base.show(io::IO, mime::MIME"text/plain", p::ValueIterationPolicy)
    println(io, "ValueIterationPolicy:")
    ds = get(io, :displaysize, displaysize(io))
    ioc = IOContext(io, :displaysize=>(first(ds)-1, last(ds)))
    showpolicy(ioc, mime, p.mdp, p)
end
