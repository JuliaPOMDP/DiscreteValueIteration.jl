[![Build Status](https://travis-ci.org/sisl/DiscreteValueIteration.jl.svg?branch=master)](https://travis-ci.org/sisl/DiscreteValueIteration.jl)
[![Coverage Status](https://coveralls.io/repos/sisl/DiscreteValueIteration.jl/badge.svg)](https://coveralls.io/r/sisl/DiscreteValueIteration.jl)

# DiscreteValueIteration

This package implements a discrete value iteration solver in Julia for Markov decision processes (MDPs). A multi-core parallel is availiable as well. At the moment, Gauss-Siedel and
vanilla value iteration solvers are provided.

## Installation

Start Julia and run the following command:

```julia
Pkg.clone("https://github.com/sisl/DiscreteValueIteration.jl")
```

Note that DiscreteMDPs.jl is required to use this module. 

## Usage

To use the DiscreteValueIteration module, begin your code by adding the maximum number of processors you would like to
use, and export the module

```julia
addprocs(10) # this is the maximum number of processors you would like to use
using ParallelValueIteration
```

Note: if you plan on using only the serial solver, you can ignore the addprocs command

To use the solver with your MDP, follow the API defined in DiscreteMDPs.jl. Define the following functions in your MDP module (see GridWolrd_.jl for a detailed example):

```julia
states(mdp::YourMDP) # returns an iterator over MDP states
actions(mdp::YourMDP) # returns an iterator over MDP actions
numStates(mdp::YourMDP) # returns the number of states
numActions(mdp::YourMDP) # returns the number of actions
nextStates(mdp::YourMDP, state, action) # returns arrays neighboring states and their probabilities e.g. (states, probs)
reward(mdp::YourMDP, state, action)
```

## Serial Solver

The module defines a SerialSolver type that can be initialized in the following way:

```julia
maxIterations = 50 # maximum number of iterations in the DP loop
tolerance = 1e-2   # Bellman residual
gs = true          # Gauss-Siedel falg
includeV = true    # return utility flag
includeQ = true    # return Q-matrix flag
includeA = false   # return policy flag
solver = SerialSolver(maxIterations=maxIterations, tolerance=tolerance, gaussSiedel=gs,
                      includeV=includeV, includeQ=includeQ, includeA=includeA)
solver = SerialSolver(maxIterations=50) # this also works
```

To solve the MDP:

```julia
mdp = AwesomeMDPType(arguments) # your MDP 
policy = solve(solver, mdp) # solve using value iteration
```

## Parallel Solver

The module defines a ParallelSolver type that has a single required input argument and a number of optional arguments.
The following two arguments are availiable in addition to the ones defined for the SerivalSolver: 

```julia
# required input
numProcs      = 8 # numbers of processors used by the solver
# optional input
stateOrder = {1:250,251:500} # default ordering is {1:numStates}
solver = ParallelSolver(numProcs, stateOrder=stateOrder, maxIterations=maxIterations,
                        tolerance=tolerance, gaussSiedel=gs,
                        includeV=includeV, includeQ=includeQ, includeA=includeA)
policy = solve(psolver, mdp) # the solve function returns the utility function and the Q-matrix
```

The state ordering is required for backwards induction value iteration, where the value function must be updated in a
specific order. For MDPs that do not require a state ordering, the stateOrder variable is set to a default value.

## Policy Solution

To access the utility function, Q-matrix, or the policy, the following API is provided:

```julia
s = 1
a = 1
u  = value(policy, s) # expected optimal value for state s
q  = value(policy, s, a) # expected value for state-action pair
ap = action(policy, s) # action that maximizes the expected utility
```


## Tutorial

An IJulia notebook tutorial is availiable with more details:

[Tutorial](http://nbviewer.ipython.org/github/sisl/DiscreteValueIteration.jl/blob/master/test/Discrete-Value-Iteration.ipynb)

## Improving Performance

- The MDP type should be small (in memory size), to avoid unnecessary data copying to each processor
