N_PROCS = 56
addprocs(N_PROCS)

using POMDPs
using POMDPModels
using SubHunt
using DiscreteValueIteration

dpomdp = DSubHuntPOMDP(SubHuntPOMDP(), 1.0)
println("Starting benchmark on Discrete SubHunt POMDP with $(n_states(dpomdp)) states")

# pvi_sharedarrays = ParallelValueIterationSolver(max_iterations=100, belres=1e-3, verbose=true, n_procs=N_PROCS)

# pvi_notshared = ParallelValueIterationNotSharedSolver(max_iterations=100, belres=1e-3, verbose=true, n_procs=N_PROCS)

t_shared = @elapsed solve(pvi_sharedarrays, dpomdp)
t_notshared = @elapsed solve(pvi_notshared, dpomdp)
@sprintf("With states as SharedArray: %.4f", t_shared)
@sprintf("With states as regular array: %.4f", t_notshared)