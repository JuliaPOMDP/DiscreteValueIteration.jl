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


## Improving Performance

- The MDP type should be small, to avoid unnecessary data copying to each processor
