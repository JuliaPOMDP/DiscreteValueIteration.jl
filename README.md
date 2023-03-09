# DiscreteValueIteration

[![CI](https://github.com/JuliaPOMDP/DiscreteValueIteration.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaPOMDP/DiscreteValueIteration.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/JuliaPOMDP/DiscreteValueIteration.jl/branch/master/graph/badge.svg?token=Qcmkye6fB0)](https://app.codecov.io/github/JuliaPOMDP/DiscreteValueIteration.jl)

This package implements the discrete value iteration algorithm in Julia for solving Markov decision processes (MDPs).
The user should define the problem with [QuickPOMDPs.jl](https://github.com/JuliaPOMDP/QuickPOMDPs.jl) or according to the API in [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl). Examples of
problem definitions can be found in [POMDPModels.jl](https://github.com/JuliaPOMDP/POMDPModels.jl). For an extensive tutorial, see [these](https://github.com/JuliaPOMDP/POMDPExamples.jl) notebooks.

There are two solvers in the package. The "vanilla" [`ValueIterationSolver`](src/vanilla.jl) calls functions from the POMDPs.jl interface in every iteration, while the [`SparseValueIterationSolver`](src/sparse.jl) first creates sparse transition and reward matrices and then performs value iteration with the new matrix representation. While both solvers take advantage of sparsity, the `SparseValueIterationSolver` is generally faster because of low-level optimizations, while the `ValueIterationSolver` has the advantage that it does not require allocation of transition matrices (which could potentially be too large to fit in memory).

## Installation

```julia
using Pkg; Pkg.add("DiscreteValueIteration")
```

## Usage

Given an MDP `mdp` defined with [QuickPOMDPs.jl](https://github.com/JuliaPOMDP/QuickPOMDPs.jl) or the POMDPs.jl interface, use 

```julia
using DiscreteValueIteration

solver = ValueIterationSolver(max_iterations=100, belres=1e-6, verbose=true) # creates the solver
policy = solve(solver, mdp) # runs value iterations
```
To extract the policy for a given state, simply call the action function:

```julia
a = action(policy, s) # returns the optimal action for state s
```

Or to extract the value, use
```julia
value(policy, s) # returns the optimal value at state s
```

### Requirements for problems defined using the POMDPs.jl interface

If you are using the POMDPs.jl interface instead of QuickPOMDPs.jl, you can see the requirements for using these solvers with

```julia
using POMDPs
using DiscreteValueIteration
@requirements_info ValueIterationSolver() YourMDP()
@requirements_info SparseValueIterationSolver() YourMDP()
```

This should return a list of the following functions to be implemented for your MDP:

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
