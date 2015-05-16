
type DiscretePolicy{T} <: Policy

    V::DenseArray{T} # utility function
    Q::DenseArray{T} # Q-matrix
    policy::Vector{Int64} 

    includeV::Bool # flag for including the utility
    includeQ::Bool # flag for including the Q-matrix
    includeA::Bool # flag for including the policy

    function DiscretePolicy(;valType::Float64)

end


function action(p::DiscretePolicy, s::Int64)

    if !p.includeA
        error("$(typeof(p)) has an empty policy array")
    end

    return p.policy[s]
end


function value(p::DiscretePolicy, s::Int64, a::Int64)

    if !p.includeQ
        error("$(typeof(p)) has an empty Q-matrix")
    end

    return p.Q[s,a]
end


function value(p::DiscretePolicy, s::Int64)
    
    if !p.includeV
        error("$(typeof(p)) has an empty optimal value array")
    end

    return p.V[s]
end
