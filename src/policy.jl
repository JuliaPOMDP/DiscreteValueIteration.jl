
type DiscretePolicy{VT, QT, PT} <: Policy

    V::DenseArray{VT} # utility function
    Q::DenseArray{QT} # Q-matrix
    policy::DenseArray{PT} # policy computed from the utility
end

function DiscretePolicy(;V::DenseArray=[], Q::DenseArray=[], P::DenseArray=[])

    vt = Any
    qt = Any
    pt = Any

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
