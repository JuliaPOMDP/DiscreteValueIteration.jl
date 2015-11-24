"""
The solver type. Contains two parameters that are keyworded in the constructor:

    - max_iterations::Int64, the maximum number of iterations value iteration runs for (default 100)
    - belres::Float64, the Bellman residual (default 1e-3)

The solver can be initialized by running:
    `solver = ValueIterationSolver(max_iterations=1000, belres=1e-6)`

"""
ValueIterationSolver


"""
The policy type. Contains the Q-Matrix, the Utility function and an array of indices corresponding to optimal actions.
There are three ways to initialize the policy type:

    `policy = ValueIterationPolicy(mdp)` 
    `policy = ValueIterationPolicy(mdp, utility_array)`
    `policy = ValueIterationPolicy(mdp, qmatrix)`

The Q-matrix is nxm, where n is the number of states and m is the number of actions.
"""
ValueIterationPolicy


"""
Returns an empty policy
"""
create_policy


"""
Computes the optimal policy for an MDP. The function takes a verbose flag which can dump text output onto the screen.
You can run the function:
    `policy = solve(solver, mdp, verbose=true)`
"""
solve


"""
Returns the optimal action associated with state s `action(policy, s)`
"""
action


"""
Returns the optimal value associated with state s `value(policy, s)`
"""
value
