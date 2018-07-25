N_PROCS = 28
addprocs(N_PROCS)

using POMDPs
using POMDPModels
using SubHunt
using DiscreteValueIteration


dpomdp = DSubHuntPOMDP(SubHuntPOMDP(), 1.0)
println("Starting benchmark on Discrete SubHunt POMDP with $(n_states(dpomdp)) states")

pvi_sharedarrays = ParallelValueIterationSolver(max_iterations=100, belres=1e-3, verbose=true, n_procs=N_PROCS)

pvi_notshared = ParallelValueIterationNotSharedSolver(max_iterations=100, belres=1e-3, verbose=true, n_procs=N_PROCS)

pvi_macro = ParallelValueIterationMacroSolver(max_iterations=100, belres=1e-3, verbose=true, n_procs=N_PROCS)

vi = ValueIterationSolver(max_iterations=100, belres=1e-3, verbose=true)

t_shared = @elapsed solve(pvi, dpomdp)
t_notshared = @elapsed solve(pvi_notshared, dpomdp)
t_macro = @elapsed solve(pvi_macro, dpomdp)
t_vi = @elapsed solve(vi, dpomdp)

@printf("With states as SharedArray: %.4f", t_shared)
@printf("With states as regular array: %.4f", t_notshared)
@printf("With @parallel: %.4f", t_macro)