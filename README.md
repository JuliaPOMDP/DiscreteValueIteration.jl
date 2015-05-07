# ParallelValueIteration

This package performs parallel value iteration on a multi-core machine. At the moment, it provides Gauss-Siedel and
vanilla value iteration solvers.

## Installation

Start Julia and run the following command:

```julia
Pkg.clone
```

## Usage

To use this module


Define the following functions in your MDP module:

```julia
states(mdp::YourMDP) # returns an iterator over MDP states
actions(mdp::YourMDP) # returns an iterator over MDP actions
numStates(mdp::YourMDP) # returns the number of states
numActions(mdp::YourMDP) # returns the number of actions
nextStates(mdp::YourMDP, state, action) # returns arrays neighboring states and their probabilities e.g. (states, probs)
reward(mdp::YourMDP, state, action)
```


## Improving Performance

- The MDP type should be small (in memory size), to avoid unnecessary data copying to each processor
