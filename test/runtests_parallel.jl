# addprocs(2)
@everywhere using DiscreteValueIteration, POMDPs, POMDPModels

mdp = GridWorld(sx=1000, sy=1000)

solver = ParallelValueIterationSolver(n_procs=3)

parallel_policy = solve(solver, mdp, verbose=true)

serial_policy = solve(ValueIterationSolver(), mdp, verbose=true)

@test isapprox(norm(parallel_policy.util - serial_policy.util), 0., atol=solver.belres)