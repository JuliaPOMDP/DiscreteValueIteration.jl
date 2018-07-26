N_PROCS = 56
addprocs(N_PROCS)

using POMDPs
using POMDPModels
using SubHunt
using DiscreteValueIteration


dpomdp = DSubHuntPOMDP(SubHuntPOMDP(), 1.0)
println("Starting benchmark on Discrete SubHunt POMDP with $(n_states(dpomdp)) states")

pvi_sharedarrays = ParallelValueIterationSolver(max_iterations=30, belres=0., verbose=true, n_procs=N_PROCS)
println("Starting parallel vi with states as shared array")
@time solve(pvi_sharedarrays, dpomdp)

pvi_notshared = ParallelValueIterationNotSharedSolver(max_iterations=30, belres=0., verbose=true, n_procs=N_PROCS)
println("Starting parallel vi with states as regular array")
@time solve(pvi_notshared, dpomdp)

pvi_macro = ParallelValueIterationMacroSolver(max_iterations=30, belres=0., verbose=true, n_procs=N_PROCS)
println("Starting parallel vi with @parallel")
solve(pvi_macro, dpomdp)

vi = ValueIterationSolver(max_iterations=30, belres=0., verbose=true)
println("Starting serial vi")
@time solve(vi, dpomdp)
