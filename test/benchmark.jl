N_PROCS = 4
addprocs(N_PROCS)


using POMDPs
using POMDPModels
using SubHunt
using DiscreteValueIteration

dpomdp = DSubHuntPOMDP(SubHuntPOMDP(), 1.0)
dpomdp = GridWorld(sx=500, sy=500)

pvi_sharedarrays = ParallelValueIterationSolver(max_iterations=100, belres=1e-3, verbose=true, n_procs=N_PROCS)

pvi_notshared = ParallelValueIterationNotSharedSolver(max_iterations=100, belres=1e-3, verbose=true, n_procs=N_PROCS)

t_shared = @elapsed solve(pvi_sharedarrays, dpomdp)
t_notshared = @elapsed solve(pvi_notshared, dpomdp)

n_states(dpomdp)