# ParallelValueIteration

This package performs parallel value iteration on a multi-core machine. At the moment, it provides Gauss-Siedel and
vanilla value iteration solvers.

## Installation

Start Julia and run the following command:

```julia
Pkg.clone("https://github.com/sisl/ParallelValueIteration.jl")
```

## Usage

To use the ParallelValueIteration module, begin your code by adding the maximum number of processors you would like to
use, and export the module

```julia
addprocs(10)
using ParallelValueIteration
```

To use the solver with your MDP, follow the API defined in DiscreteMDPs.jl. Define the following functions in your MDP module (see GridWolrd_.jl for a detailed example):

```julia
states(mdp::YourMDP) # returns an iterator over MDP states
actions(mdp::YourMDP) # returns an iterator over MDP actions
numStates(mdp::YourMDP) # returns the number of states
numActions(mdp::YourMDP) # returns the number of actions
nextStates(mdp::YourMDP, state, action) # returns arrays neighboring states and their probabilities e.g. (states, probs)
reward(mdp::YourMDP, state, action)
```

## Solver

The module defines a ParallelSolver type that requires a number of input arguments

```julia
numProcs      = 8 # numbers of processors used by the solver
stateOrder    = {[1,1000], [1001,2000]} # ordering in which to process states
numIterations = 10 # number of iterations in the DP loop
tolerance     = 1e-3 # Bellman tolerance
gsFlag        = true # flag for Gauss-Siedel iteration

myMDP = AwesomeMDPType(arguments) # your MDP 

pvi = ParallelSolver(numProcs, stateOrder, numIterations, tolerance, gsFlag)
(util, qMatrix) = solve(pvi, myMDP) # the solve function returns the utility function and the Q-matrix
```

The state ordering is required for backwards induction value iteration, where the value function must be updated in a
specific order. For MDPs that do not require a state ordering, the stateOrder variable can be defined in the following
way:

```julia
stateOrder    = {[1,nStates]} # nStates is the total number of states in your MDP
```


## Improving Performance

- The MDP type should be small (in memory size), to avoid unnecessary data copying to each processor
