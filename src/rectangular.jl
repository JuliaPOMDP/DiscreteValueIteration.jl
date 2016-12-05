# The rectangular weighted discrete value iteration solver 
# solves for states according to a local weighting function
# that uses rectangular interpolation.

# The TinyMDP type, which builds a much smaller MDP from a larger MDP
# at intervals specified by the user
function big_to_small(s::GridWorldState, grid)
    # Converts from big-size GridWorldState instances to small-size GridWorldState instances
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

type TinyMDP <: MDP{GridWorldState, GridWorldAction}
    big::GridWorld
    small::GridWorld
    grid

    function TinyMDP(big_mdp::GridWorld, x_indices::Array{Int}, y_indices::Array{Int})
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
	function RectangularValueIterationSolver(tiny_mdp::TinyMDP, max_iterations::Int64=100, belres::Float64=1e-3)
		self = new()
		self.tiny_mdp = tiny_mdp
		self.max_iterations = max_iterations
		self.belres = belres
		return self
	end
	function RectangularValueIterationSolver(big_mdp::Union{MDP,POMDP}, x_indices::Array{Int}, y_indices::Array{Int}; max_iterations::Int64=100, belres::Float64=1e-3)
		tiny = TinyMDP(big_mdp, x_indices, y_indices)
		return RectangularValueIterationSolver(tiny, max_iterations, belres)
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
	println(action_quality)
	best_action, total_weight = Base.Collections.peek(action_quality)
    return policy.subspace_policy.action_map[best_action]
end

function value{S}(policy::LocallyWeightedValueIterationPolicy, s::S)
	interpolate(policy.grid, policy.subspace_policy.util, [s.x, s.y])
end

