addprocs(2)
using DiscreteValueIteration

@everywhere using POMDPs, POMDPModels

mdp = GridWorld(sx=1000, sy=1000)

solver = ParallelValueIterationSolver(n_procs=3)

parallel_policy = solve(solver, mdp)

serial_policy = solve(ValueIterationSolver(), mdp)

@test isapprox(norm(parallel_policy.util - serial_policy.util), 0.)