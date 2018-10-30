# DiscreteValueIteration

[![Build Status](https://travis-ci.org/JuliaPOMDP/DiscreteValueIteration.jl.svg?branch=master)](https://travis-ci.org/JuliaPOMDP/DiscreteValueIteration.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaPOMDP/DiscreteValueIteration.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaPOMDP/DiscreteValueIteration.jl?branch=master)

This package implements the discrete value iteration algorithm in Julia for solving Markov decision processes (MDPs).
The user should define the problem according to the API in [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl). Examples of
problem definitions can be found in [POMDPModels.jl](https://github.com/JuliaPOMDP/POMDPModels.jl). For an extensive tutorial, see [these](https://github.com/JuliaPOMDP/POMDPExamples.jl) notebooks.

There are two solvers in the package. The "vanilla" [`ValueIterationSolver`](src/vanilla.jl) calls functions from the POMDPs.jl interface in every iteration, while the [`SparseValueIterationSolver`](src/sparse.jl) first creates sparse transition and reward matrices and then performs value iteration with the new matrix representation. While both solvers take advantage of sparsity, the `SparseValueIterationSolver` is generally faster because of low-level optimizations, while the `ValueIterationSolver` has the advantage that it does not require allocation of transition matrices (which could potentially be too large to fit in memory).

## Installation

Start Julia and make sure you have the JuliaPOMDP registry:

```julia
import POMDPs
POMDPs.add_registry()
```

Then install using the standard package manager:

```julia
using Pkg; Pkg.add("DiscreteValueIteration")
```

## Usage

Use

```julia
using POMDPs
using DiscreteValueIteration
@requirements_info ValueIterationSolver() YourMDP()
@requirements_info SparseValueIterationSolver() YourMDP()
```

to get a list of POMDPs.jl functions necessary to use the solver. This should return a list of the following functions to be implemented for your MDP:

```julia
discount(::MDP)
n_states(::MDP)
n_actions(::MDP)
transition(::MDP, ::State, ::Action)
reward(::MDP, ::State, ::Action, ::State)
stateindex(::MDP, ::State)
actionindex(::MDP, ::Action)
actions(::MDP, ::State)
support(::StateDistribution)
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
a = action(policy, s) # returns the optimal action for state s
```
