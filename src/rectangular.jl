# The multilinear (rectangular) weighted discrete value iteration solver 
# solves for states according to a local weighting function
# that uses multilinear interpolation.
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
# - Implement big_to_small, which converts from instances of the big_mdp state
#   objects into instances of the tiny_mdp state objects.  For instance, the
#   initial state might be ContinuousProblemState that has three variables,
#   where variable 1 ranges from 0 to 1000, variable 2 ranges from -10 to 10,
#   and variable 3 ranges from -10 to 100.  The RectangleGrid might look like:
#   RectangleGrid(collect(0:100:1000), collect(-10:0.5:10), collect(-10:10:100))
#   The tiny_mdp might contain ContinuousProblemState(0,-10,-10) -- or it might
#   instead be a new state space where var1 ranges from 0->10 by 1, var2 ranges from 
#   -20->20 by 1, and var3 ranges from -1->10 by 1. Alternatively, the tiny_mdp 
#   might be in the original space, and the big_to_small transform might be
#   the identity.  Alternatively, the tiny_mdp might be in the gridworld.
# - Implement the type TinyMDP that is an MDP containing your big MDP, the 
#   smaller MDP you construct from the big MDP, and the grid.  TinyMDP uses
#   whatever state and action spaces the user implements.
# - Use the RectangularValueIterationSolver on the user-implemented TinyMDP type,
#   with no further changes.

# The TinyMDP type, which builds a much smaller MDP from a larger MDP
# at intervals specified by the user
function big_to_small(s::GridWorldState, grid::RectangleGrid)
    # Converts from big_mdp-state instances to small-size GridWorldState instances
	# that appear on the provided grid
	# For the GridWorld problem, the tiny_mdp is simply a smaller discretized
	# MDP -- this is the transform function from a state in the large m-by-n space to
	# state in the smaller gridworld space indexed by (1 to m/discretization) and
	# (1 to n/discretization) cells.
    nrows = size(grid.cutPoints)[1]
    ncols = size(grid.cutPoints[1])[1]
    locs, weights = interpolants(grid, [s.x, s.y])
    mxweight, mxindx = findmax(weights,1)
    x,y = ind2sub((nrows, ncols), locs[mxindx...])
    return GridWorldState(x,y)
end
@assert big_to_small(GridWorldState(1,1), RectangleGrid([2,5], [2,5])) == GridWorldState(1,1)
@assert big_to_small(GridWorldState(1,5), RectangleGrid([2,5], [2,5])) == GridWorldState(1,2)
@assert big_to_small(GridWorldState(10,-2), RectangleGrid([2,5], [2,5])) == GridWorldState(2,1)
@assert big_to_small(GridWorldState(5,6), RectangleGrid([2,5], [2,5])) == GridWorldState(2,2)

@assert big_to_small(GridWorldState(1,15), RectangleGrid([2,5], [2,5,10])) == GridWorldState(1,3)
@assert big_to_small(GridWorldState(7,15), RectangleGrid([2,5], [2,5,10])) == GridWorldState(2,3)
@assert big_to_small(GridWorldState(6,0), RectangleGrid([2,5], [2,5,10])) == GridWorldState(2,1)

type TinyGridWorldMDP <: MDP{GridWorldState, GridWorldAction}
    big::GridWorld
    small::GridWorld
    grid

    function TinyGridWorldMDP(big_mdp::GridWorld, x_indices::Array{Int}, y_indices::Array{Int})
        self = new()
        
        self.big = big_mdp
        self.grid = RectangleGrid(x_indices, y_indices)

        sx = length(x_indices)
        sy = length(y_indices)

        reward_states_to_values = Dict()
        for i in 1:length(big_mdp.reward_states)
            s = big_mdp.reward_states[i]
            new_reward_state = big_to_small(s, self.grid)
            if !haskey(reward_states_to_values, new_reward_state)
                reward_states_to_values[new_reward_state] = big_mdp.reward_values[i] 
            else
                reward_states_to_values[new_reward_state] += big_mdp.reward_values[i]            
            end
        end
        terminals = Set{GridWorldState}()
        for s in big_mdp.terminals
            new_terminal_state = big_to_small(s, self.grid)
            terminals = push!(terminals, new_terminal_state)
        end

        self.small = GridWorld(sx, sy, 
            collect(keys(reward_states_to_values)), 
            collect(values(reward_states_to_values)), 
            big_mdp.bounds_penalty, big_mdp.tprob, terminals, 
            big_mdp.discount_factor, zeros(2))

        return self
    end
end

#############################################################

# The solver type 
type RectangularValueIterationSolver <: Solver
	tiny_mdp::Union{MDP,POMDP} # builds a grid world per user request
    max_iterations::Int64 # max number of iterations 
    belres::Float64 # the Bellman Residual
	
	# Default constructor
	function RectangularValueIterationSolver(tiny_mdp::Union{MDP,POMDP}, max_iterations::Int64=100, belres::Float64=1e-3)
		self = new()
		self.tiny_mdp = tiny_mdp
		self.max_iterations = max_iterations
		self.belres = belres
		return self
	end
end

# returns a default value iteration policy
function create_policy(solver::RectangularValueIterationSolver, mdp::Union{MDP,POMDP})
    return LocallyWeightedValueIterationPolicy(mdp, solver.tiny_mdp.grid)
end

# runs the value iteration algorithm
function solve(solver::RectangularValueIterationSolver, big_mdp::Union{MDP,POMDP}, policy::LocallyWeightedValueIterationPolicy; verbose::Bool=false)
	limitedSolver = ValueIterationSolver()
    p = create_policy(limitedSolver, solver.tiny_mdp.small) 
    p = solve(limitedSolver, solver.tiny_mdp.small, p, verbose=true)
	policy.subspace_policy = p
	policy
end

function action{S,A}(policy::LocallyWeightedValueIterationPolicy, s::S, a::A=nothing)
	action_quality = Base.Collections.PriorityQueue() # lowest is at top
    locs, weights = interpolants(policy.grid, [s.x, s.y])
	for i in 1:length(locs)
		basis_state = locs[i]
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
	interpolate(policy.grid, policy.subspace_policy.util, [s.x, s.y])
end

