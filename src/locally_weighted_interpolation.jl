# The locally weighted discrete value iteration solver 
# solves for states according to a local weighting function
# that uses multilinear or simplex interpolation.
# The user needs to do the following:
# - Write the big_mdp object. This might be a continuous MDP or a high-
#   dimensional MDP. If we continue & use the RectangularValueIterationSolver, 
#   we will *not* actually solve this MDP -- we'll solve a smaller, discrete one,
#   and provide the ability to query the solved MDP for locally approximately
#   optimal values and actions.
# - Create a grid of type RectangleGrid that discretizes the space
#   as the user desires with whatever level of granularity the user desires.
#   The grid defines the basis functions. It can have as many dimensions
#   with whatever labels the user prefers, but it recommended to have each 
#   dimension reflect one dimension in the state space and the values
#   included in the RectangleGrid within each dimension reflecting
#   some discretization of that dimension.
#   The tiny_mdp might contain ContinuousProblemState(0,-10,-10) -- or it might
#   instead be a new state space where var1 ranges from 0->10 by 1, var2 ranges from 
#   -20->20 by 1, and var3 ranges from -1->10 by 1. Alternatively, the tiny_mdp 
#   might be in the original space, and the big_to_small transform might be
#   the identity.  Alternatively, the tiny_mdp might be in the gridworld.
# - Implement the type TinyMDP that is an MDP containing your big MDP, the 
#   smaller MDP you construct from the big MDP, and the grid.  TinyMDP uses
#   whatever state and action spaces the user implements.
# - Use the LocallyWeightedValueIterationSolver on the user-implemented TinyMDP type,
#   with no further changes.


# Requires a grid::AbstractGrid with `interpolate` and `interpolants` defined according
# to the interface in GridInterpolations.jl. 


# Any Tiny*MDP will require these properties to work with LocallyWeightedValueIterationSolver:
# - big::Union{MDP,POMDP}
# - small::Union{MDP,POMDP}
# - grid::AbstractGrid with `interpolate` and `interpolants` defined according
#   to the interface in GridInterpolations.jl. 
# See TinyGridWorldMDP.jl for an example implementation.


# Example implementations of the expected class (Tiny*MDP) are in the test folder.

# The solver type 
type LocallyWeightedValueIterationSolver <: Solver
	tiny_mdp::Union{MDP,POMDP} # a solvable MDP (smaller, discrete)
    max_iterations::Int64 # max number of iterations 
    belres::Float64 # the Bellman Residual
	
	# Default constructor
	function LocallyWeightedValueIterationSolver(tiny_mdp::Union{MDP,POMDP}, max_iterations::Int64=100, belres::Float64=1e-3)
		self = new()
		self.tiny_mdp = tiny_mdp
		self.max_iterations = max_iterations
		self.belres = belres
		return self
	end
end

# returns a default value iteration policy
function create_policy(solver::LocallyWeightedValueIterationSolver, mdp::Union{MDP,POMDP})
    return LocallyWeightedValueIterationPolicy(mdp, solver.tiny_mdp.grid)
end

# runs the value iteration algorithm
function solve(solver::LocallyWeightedValueIterationSolver, big_mdp::Union{MDP,POMDP}, policy::LocallyWeightedValueIterationPolicy; verbose::Bool=false)
	limitedSolver = ValueIterationSolver()
    p = create_policy(limitedSolver, solver.tiny_mdp.small) 
    p = solve(limitedSolver, solver.tiny_mdp.small, p, verbose=true)
	policy.subspace_policy = p
	policy
end

function action{S,A}(policy::LocallyWeightedValueIterationPolicy, s::S, a::A=nothing)
	action_quality = Base.Collections.PriorityQueue() # lowest is at top
    locs, weights = interpolants(policy.grid, [s.x, s.y]) ########################################################
	for i in 1:length(locs)
		basis_state = locs[i] # reflects the order of the data provided (util)
		act = policy.subspace_policy.policy[basis_state]
		if !haskey(action_quality, act)
			action_quality[act] = 0
		end
		action_quality[act] -= weights[i]
	end
	best_action, total_weight = Base.Collections.peek(action_quality)
    return policy.subspace_policy.action_map[best_action]
end

function value{S}(policy::LocallyWeightedValueIterationPolicy, s::S)
	interpolate(policy.grid, policy.subspace_policy.util, [s.x, s.y]) ########################################################
end


