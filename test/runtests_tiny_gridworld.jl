

# A series of tests checking that the conversion between the original coordinate
# system and the limited coordinate system is correct appears inside rectangular.jl
# This test relies on TinyGridWorldMDP

@assert big_to_small(GridWorldState(1,1), RectangleGrid([2,5], [2,5])) == GridWorldState(1,1)
@assert big_to_small(GridWorldState(1,5), RectangleGrid([2,5], [2,5])) == GridWorldState(1,2)
@assert big_to_small(GridWorldState(10,-2), RectangleGrid([2,5], [2,5])) == GridWorldState(2,1)
@assert big_to_small(GridWorldState(5,6), RectangleGrid([2,5], [2,5])) == GridWorldState(2,2)

@assert big_to_small(GridWorldState(1,15), RectangleGrid([2,5], [2,5,10])) == GridWorldState(1,3)
@assert big_to_small(GridWorldState(7,15), RectangleGrid([2,5], [2,5,10])) == GridWorldState(2,3)
@assert big_to_small(GridWorldState(6,0), RectangleGrid([2,5], [2,5,10])) == GridWorldState(2,1)


function test_correct_conversion_to_gridworld()
	big_mdp = GridWorld(sx=6, sy=6, rs=[GridWorldState(6,2), GridWorldState(5,1)], rv = [10.0, 5.0])
	mdp_indices = [2,5]
	tiny_mdp = TinyGridWorldMDP(big_mdp, mdp_indices, mdp_indices)

	return (length(tiny_mdp.small.reward_states) == 1) && 
		(tiny_mdp.small.reward_states[1] == GridWorldState(2,1) ) &&
		(tiny_mdp.small.reward_values[1] == 15)
end
@test test_correct_conversion_to_gridworld() == true


