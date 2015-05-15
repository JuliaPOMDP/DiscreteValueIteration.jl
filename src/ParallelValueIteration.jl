# multi-core, single machine parallel value iteration solver

module ParallelValueIteration

using DiscreteMDPs

import DiscreteMDPs.Solver
import DiscreteMDPs.solve
import DiscreteMDPs.Policy

include("policy.jl")
include("serial.jl")
include("parallel.jl")

export ParallelSolver
export SerialSolver
export solve
export ValueIterationPolicy

end # module
