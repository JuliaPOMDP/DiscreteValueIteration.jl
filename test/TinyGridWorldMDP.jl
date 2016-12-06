# The Tiny*MDP type builds a much smaller MDP from a larger MDP
# at intervals specified by the user in a grid or simplex.  
# It stores the original MPP, the discretized smaller MDP, and 
# the grid or simplex. 

using GridInterpolations

function big_to_small(s::GridWorldState, grid::AbstractGrid)
	# Helper function.
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

type TinyGridWorldMDP <: MDP{GridWorldState, GridWorldAction}
	# TinyGridWorldMDP builds a tiny discretized gridworld based on a 
	# large gridworld big_mdp and a grid provided by the user.

    big::GridWorld
    small::GridWorld
    grid::AbstractGrid

    function TinyGridWorldMDP(big_mdp::GridWorld, grid::AbstractGrid)
        self = new()
        
        self.big = big_mdp
        self.grid = grid

		@assert length(self.grid.cutPoints) == 2 # as per POMDPModels.GridWorld
        sx = length(self.grid.cutPoints[1])
        sy = length(self.grid.cutPoints[2])

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