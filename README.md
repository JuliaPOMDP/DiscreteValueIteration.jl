# DiscreteValueIteration

[![Build Status](https://travis-ci.org/JuliaPOMDP/DiscreteValueIteration.jl.svg?branch=master)](https://travis-ci.org/JuliaPOMDP/DiscreteValueIteration.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaPOMDP/DiscreteValueIteration.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaPOMDP/DiscreteValueIteration.jl?branch=master)

This package implements the discrete value iteration algorithm in Julia for solving Markov decision processes (MDPs).
The user should define the problem according to the API in [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl). Examples of
problem definitions can be found in [POMDPModels.jl](https://github.com/JuliaPOMDP/POMDPModels.jl). For an extensive tutorial, see the [this](http://nbviewer.ipython.org/github/JuliaPOMDP/POMDPs.jl/blob/master/examples/GridWorld.ipynb) notebook.

## Installation

Start Julia and run the following command:

```julia
using POMDPs
POMDPs.add("DiscreteValueIteration")
```


## Usage

Use

```julia
using POMDPs
using DiscreteValueIteration
@requirements_info ValueIterationSolver() YourMDP()
```

to get a list of POMDPs.jl functions necessary to use the solver. This should return a list of the following functions to be implemented for your MDP:

```julia
discount(::MDP)
n_states(::MDP)
n_actions(::MDP)
transition(::MDP, ::State, ::Action)
reward(::MDP, ::State, ::Action, ::State)
state_index(::MDP, ::State)
action_index(::MDP, ::Action)
actions(::MDP, ::State)
iterator(::ActionSpace)
iterator(::StateSpace)
iterator(::StateDistribution)
pdf(::StateDistribution, ::State)
states(::MDP)
actions(::MDP)
```

Once the above functions are defined, the solver can be called with the following syntax:

```julia
using DiscreteValueIteration

mdp = MyMDP() # initializes the MDP
solver = ValueIterationSolver(max_iterations=100, belres=1e-6) # initializes the Solver type
solve(solver, mdp) # runs value iterations
```

To extract the policy for a given state, simply call the action function:

```julia
a = action(polciy, s) # returns the optimal action for state s
```
