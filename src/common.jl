# The policy type
mutable struct ValueIterationPolicy{F} <: Policy
    qmat::AbstractMatrix{F} # Q matrix storing Q(s,a) values
    util::AbstractVector{F} # The value function V(s)
    policy::AbstractVector{Int64} # Policy array, maps state index to action index
    action_map::Vector # Maps the action index to the concrete action type
    include_Q::Bool # Flag for including the Q-matrix
    mdp::Union{MDP,POMDP} # uses the model for indexing in the action function
end

# constructor with an optinal initial value function argument
function ValueIterationPolicy(mdp::Union{MDP,POMDP};
                              utility::AbstractVector{Float64}=zeros(n_states(mdp)),
                              policy::AbstractVector{Int64}=zeros(Int64, n_states(mdp)),
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

function POMDPPolicies.actionvalues(policy::ValueIterationPolicy, s::S) where S
    if !policy.include_Q
        error("ValueIterationPolicyError: the policy does not contain the Q function!")
    else
        sidx = stateindex(policy.mdp, s)
        return policy.qmat[sidx,:]
    end
end
