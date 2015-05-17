
type DiscretePolicy{VT, QT, PT} <: Policy

    V::DenseArray{VT} # utility function
    Q::DenseArray{QT} # Q-matrix
    policy::DenseArray{PT} # policy computed from the utility
end

function DiscretePolicy(;V::DenseArray=[], Q::DenseArray=[], P::DenseArray=[])

    vt = None
    qt = None
    pt = None

    !isempty(V) ? (vt = typeof(V[1])) : (nothing)
    !isempty(Q) ? (qt = typeof(Q[1])) : (nothing)
    !isempty(P) ? (pt = typeof(P[1])) : (nothing)

    return DiscretePolicy{vt, qt, pt}(V, Q, P)
end


function action(p::DiscretePolicy, s::Int64)

    policy = p.policy
    if isempty(policy) 
        error("$(typeof(p)) has an empty policy array")
    end

    return policy[s]
end


function value(p::DiscretePolicy, s::Int64, a::Int64)

    Q = p.Q
    if isempty(Q)
        error("$(typeof(p)) has an empty Q-matrix")
    end

    return Q[a,s]
end


function value(p::DiscretePolicy, s::Int64)
    
    V = p.V
    if isempty(V) 
        error("$(typeof(p)) has an empty optimal value array")
    end

    return V[s]
end


# computes policy given value function (can be done in parallel)
function computePolicy(mdp::DiscreteMDP, V::DenseArray)
    nStates  = numStates(mdp)
    nActions = numActions(mdp)

    @assert nStates == length(V)

    policy = zeros(Int64, nStates)

    for si = 1:nStates
        qHi = 0.0
        for ai = 1:nActions
            states, probs = nextStates(mdp, si, ai)
            qNow = reward(mdp, si, ai)
            for sp = 1:length(states)
                spi = states[sp]
                qNow += probs[sp] * V[spi]
            end
            if ai == 1 || qNow > qHi
                qHi = qNow
                policy[si] = ai
            end
        end
    end
    return policy
end
