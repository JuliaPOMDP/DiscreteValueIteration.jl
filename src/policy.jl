# The policy type
type ValueIterationPolicy <: Policy
    qmat::Matrix{Float64} # Q matrix storing Q(s,a) values
    util::Vector{Float64} # The value function V(s)
    policy::Vector{Int64} # Policy array, maps state index to action index
    action_map::Vector{Any} # Maps the action index to the concrete action type
    include_Q::Bool # Flag for including the Q-matrix
    mdp::Union{MDP,POMDP} # uses the model for indexing in the action function
    # constructor with an optimal initial value function argument
    function ValueIterationPolicy(mdp::Union{MDP,POMDP}; 
                                  utility::Vector{Float64}=Array(Float64,0),
                                  include_Q::Bool=true)
        ns = n_states(mdp)
        na = n_actions(mdp)
        self = new()
        if !isempty(utility)
            @assert size(utility)[1] == ns "Input utility dimension mismatch"
            self.util = utility
        else
            self.util = zeros(ns)
        end
        am = Any[]
        for a in iterator(actions(mdp))
            aidx = action_index(mdp, a)
            push!(am, aidx)
        end
        self.action_map = am
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
    # constructor for default Q-matrix
    function ValueIterationPolicy(mdp::Union{MDP,POMDP}, q::Matrix{Float64})
        (ns, na) = size(q)
        p = zeros(ns)
        u = zeros(ns)
        for i = 1:ns
            p[i] = indmax(q[i,:])
            u[i] = maximum(q[i,:])
        end
        am = Any[]
        for a in iterator(actions(mdp))
            aidx = action_index(mdp, a)
            push!(am, aidx)
        end
        self = new()
        self.qmat = q
        self.util = u
        self.policy = p
        self.action_map = am
        self.include_Q = true
        self.mdp = mdp
        return self
    end
end


# returns the fields of the policy type
function locals(p::ValueIterationPolicy)
    return (p.qmat,p.util,p.policy,p.action_map)
end




type LocallyWeightedValueIterationPolicy <: Policy
	subspace_policy::ValueIterationPolicy
	grid::RectangleGrid
	
    function LocallyWeightedValueIterationPolicy(mdp::Union{MDP,POMDP}, grid::RectangleGrid)
        self = new()
		self.subspace_policy = ValueIterationPolicy(mdp)
		self.grid = grid
        return self
    end
end