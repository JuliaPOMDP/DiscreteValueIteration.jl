"""
This module implements a value iteration solver that uses the interface defined in POMDPs.jl
"""
module DiscreteValueIteration

using Printf
using POMDPs
using POMDPTools
using SparseArrays
using LinearAlgebra
import POMDPLinter: @POMDP_require, @req, @subreq, @warn_requirements

import POMDPs: Solver, solve, Policy, action, value

export
    ValueIterationPolicy,
    ValueIterationSolver,
    SparseValueIterationSolver,
    solve,
    action,
    value,
    locals

include("common.jl")
include("vanilla.jl")
include("sparse.jl")

end # module
