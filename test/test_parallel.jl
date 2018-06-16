addprocs(2)
@everywhere using POMDPs, POMDPModels
@everywhere using DiscreteValueIteration
mdp = GridWorld(sx=500, sy=500)

solver = ParallelValueIterationSolver(n_procs=3)

policy = solve(solver, mdp, verbose=true)

solver = ValueIterationSolver()

solve(solver, mdp, verbose=true)
