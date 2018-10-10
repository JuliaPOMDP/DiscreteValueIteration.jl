using Distributed
using LinearAlgebra
using Test
addprocs(3)
@everywhere begin
    using SharedArrays
    using POMDPs
    using POMDPModels
    using DiscreteValueIteration
end

mdp = SimpleGridWorld(size=(500, 500))

parallel_solver = ParallelValueIterationSolver(n_procs=2, verbose=true, max_iterations=15)

@time parallel_policy = solve(parallel_solver, mdp)
serial_solver = ValueIterationSolver(verbose=true, max_iterations=15)
@time serial_policy = solve(serial_solver, mdp)

@test isapprox(norm(parallel_policy.util - serial_policy.util), 0., atol=serial_solver.belres)

synchronous_solver = ParallelValueIterationSolver(asynchronous=false)
@test_throws String solve(synchronous_solver, mdp)